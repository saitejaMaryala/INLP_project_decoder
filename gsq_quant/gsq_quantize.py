"""
Gradient-Sensitivity Quantization (GSQ)
----------------------------------------
Fast, targeted mixed-precision quantization using gradient magnitudes
to identify sensitive layers — no optimization loop required.

Pipeline:
  1. Forward + backward pass on ~50 calibration samples
  2. Score each Linear layer by gradient magnitude
  3. Assign INT8 to top-k% sensitive layers, INT4 to the rest
  4. One-shot direct rounding quantization
  5. Save quantized weights + sensitivity metadata
"""

import os
import json
import torch
import torch.nn as nn
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class GSQConfig:
    model_path: str
    output_path: str
    n_calibration_samples: int = 50
    max_seq_len: int = 512
    sensitive_ratio: float = 0.25          # top 25% layers → INT8
    sensitive_bits: int = 8
    robust_bits: int = 4
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    dataset_split: str = "train"
    calibration_seed: int = 42


# ---------------------------------------------------------------------------
# Step 1: Calibration data
# ---------------------------------------------------------------------------

def get_calibration_tokens(
    tokenizer,
    cfg: GSQConfig,
    device: torch.device,
) -> list[torch.Tensor]:
    """
    Load a small slice of wikitext-2 and tokenize into fixed-length chunks.
    Returns a list of (1, seq_len) token tensors on `device`.
    """
    ds = load_dataset(cfg.dataset_name, cfg.dataset_config, split=cfg.dataset_split)
    text = "\n\n".join(ds["text"])

    torch.manual_seed(cfg.calibration_seed)
    enc = tokenizer(text, return_tensors="pt")
    all_ids = enc["input_ids"][0]           # (total_tokens,)

    n_tok = cfg.max_seq_len
    max_start = len(all_ids) - n_tok
    starts = torch.randperm(max_start)[: cfg.n_calibration_samples]

    samples = []
    for s in starts:
        chunk = all_ids[s : s + n_tok].unsqueeze(0).to(device)
        samples.append(chunk)

    print(f"[GSQ] Prepared {len(samples)} calibration chunks of length {n_tok}")
    return samples


# ---------------------------------------------------------------------------
# Step 2: Gradient-magnitude sensitivity map
# ---------------------------------------------------------------------------

def compute_sensitivity_map(
    model: nn.Module,
    samples: list[torch.Tensor],
    cfg: GSQConfig,
) -> dict[str, float]:
    """
    Run forward+backward on calibration samples.
    For every Linear layer, accumulate the mean absolute gradient of its weight.
    Returns {layer_name: sensitivity_score}.
    """
    # Temporarily enable gradients
    for p in model.parameters():
        p.requires_grad_(True)

    # Accumulate |grad| sums and counts
    grad_accum: dict[str, torch.Tensor] = {}

    print(f"[GSQ] Running {len(samples)} calibration passes …")
    for i, input_ids in enumerate(samples):
        model.zero_grad()

        # Teacher-forcing: labels = input_ids shifted inside the model
        outputs = model(input_ids=input_ids, labels=input_ids)
        loss = outputs.loss
        loss.backward()

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and module.weight.grad is not None:
                score = module.weight.grad.abs().mean().item()
                grad_accum[name] = grad_accum.get(name, 0.0) + score

        if (i + 1) % 10 == 0:
            print(f"  … {i + 1}/{len(samples)} samples done")

    model.zero_grad()
    # Freeze again
    for p in model.parameters():
        p.requires_grad_(False)

    # Average over samples
    n = len(samples)
    sensitivity_map = {name: total / n for name, total in grad_accum.items()}

    print(f"[GSQ] Computed sensitivity scores for {len(sensitivity_map)} Linear layers")
    return sensitivity_map


# ---------------------------------------------------------------------------
# Step 3: Assign bit-widths
# ---------------------------------------------------------------------------

def assign_bit_widths(
    sensitivity_map: dict[str, float],
    cfg: GSQConfig,
) -> dict[str, int]:
    """
    Rank layers by sensitivity score (descending).
    Top `sensitive_ratio` fraction → INT8, rest → INT4.
    Returns {layer_name: n_bits}.
    """
    ranked = sorted(sensitivity_map.items(), key=lambda x: x[1], reverse=True)
    n_sensitive = max(1, int(len(ranked) * cfg.sensitive_ratio))

    bit_assignment: dict[str, int] = {}
    for rank, (name, score) in enumerate(ranked):
        bits = cfg.sensitive_bits if rank < n_sensitive else cfg.robust_bits
        bit_assignment[name] = bits

    n8 = sum(1 for b in bit_assignment.values() if b == cfg.sensitive_bits)
    n4 = len(bit_assignment) - n8
    print(f"[GSQ] Bit-width assignment: {n8} layers → INT{cfg.sensitive_bits}, "
          f"{n4} layers → INT{cfg.robust_bits}")
    return bit_assignment


# ---------------------------------------------------------------------------
# Step 4: One-shot quantization helpers
# ---------------------------------------------------------------------------

def quantize_tensor(tensor: torch.Tensor, n_bits: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Symmetric per-channel (output-channel) min-max quantization.
    Returns (quantized_weight_int8, scale, zero_point=0).

    We store everything as int8 regardless of n_bits to keep it simple;
    the scale encodes the effective dynamic range for 4-bit or 8-bit.
    """
    qmin = -(2 ** (n_bits - 1))
    qmax = 2 ** (n_bits - 1) - 1

    # Per output-channel scale
    w = tensor.float()
    abs_max = w.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8)
    scale = abs_max / qmax                          # (out_channels, 1)

    w_quant = (w / scale).round().clamp(qmin, qmax).to(torch.int8)
    return w_quant, scale.squeeze(1).half(), torch.zeros(tensor.shape[0], dtype=torch.int8)


def dequantize_tensor(
    w_int: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    original_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Reconstruct FP16 weights from quantized representation."""
    w = w_int.float() * scale.float().unsqueeze(1)
    return w.to(original_dtype)


# ---------------------------------------------------------------------------
# Step 5: Apply quantization to model in-place
# ---------------------------------------------------------------------------

def apply_gsq_quantization(
    model: nn.Module,
    bit_assignment: dict[str, int],
) -> dict[str, dict]:
    """
    Walk every Linear layer in `model`.
    - If it's in bit_assignment: quantize → dequantize (simulate quant),
      store quant metadata in a side dict.
    - Mutates model weights in-place (replaces with reconstructed FP16).
    Returns quant_state dict for saving.
    """
    quant_state: dict[str, dict] = {}

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if name not in bit_assignment:
            continue

        n_bits = bit_assignment[name]
        orig_dtype = module.weight.dtype
        w = module.weight.data

        w_int, scale, zero_point = quantize_tensor(w, n_bits)
        w_reconstructed = dequantize_tensor(w_int, scale, zero_point, orig_dtype)

        module.weight.data = w_reconstructed

        quant_state[name] = {
            "n_bits": n_bits,
            "scale": scale.cpu().tolist(),
            "zero_point": zero_point.cpu().tolist(),
            "shape": list(w.shape),
        }

    print(f"[GSQ] Quantized {len(quant_state)} layers in-place (simulated quant)")
    return quant_state


# ---------------------------------------------------------------------------
# Step 6: Save
# ---------------------------------------------------------------------------

def save_gsq_model(
    model: nn.Module,
    tokenizer,
    quant_state: dict,
    sensitivity_map: dict[str, float],
    bit_assignment: dict[str, int],
    cfg: GSQConfig,
) -> None:
    out = Path(cfg.output_path)
    out.mkdir(parents=True, exist_ok=True)

    # Save model weights + tokenizer
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))

    # Save GSQ metadata
    metadata = {
        "quantization_method": "GSQ",
        "config": asdict(cfg),
        "sensitivity_map": {k: float(v) for k, v in sensitivity_map.items()},
        "bit_assignment": bit_assignment,
        "quant_state": quant_state,
    }
    with open(out / "gsq_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[GSQ] Saved quantized model + metadata → {out}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_gsq(cfg: GSQConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[GSQ] Using device: {device}")

    # ── Load model ──────────────────────────────────────────────────────────
    print(f"[GSQ] Loading model from {cfg.model_path} …")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Step 1: Calibration data ─────────────────────────────────────────────
    samples = get_calibration_tokens(tokenizer, cfg, device)

    # ── Step 2: Sensitivity map ──────────────────────────────────────────────
    sensitivity_map = compute_sensitivity_map(model, samples, cfg)

    # ── Step 3: Bit-width assignment ─────────────────────────────────────────
    bit_assignment = assign_bit_widths(sensitivity_map, cfg)

    # ── Step 4+5: One-shot quantization ──────────────────────────────────────
    quant_state = apply_gsq_quantization(model, bit_assignment)

    # ── Step 6: Save ─────────────────────────────────────────────────────────
    save_gsq_model(model, tokenizer, quant_state, sensitivity_map, bit_assignment, cfg)


if __name__ == "__main__":
    cfg = GSQConfig(
        model_path="hf_models/Llama-3.1-8B-Instruct",
        output_path="hf_models/Llama-3.1-8B-Instruct-GSQ",
        n_calibration_samples=50,
        max_seq_len=512,
        sensitive_ratio=0.25,
        sensitive_bits=8,
        robust_bits=4,
    )
    run_gsq(cfg)