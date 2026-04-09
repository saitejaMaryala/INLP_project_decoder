"""
Gradient-Sensitivity Quantization (GSQ)
----------------------------------------
Saves a fully self-contained checkpoint.

What is saved  (output_path/)
──────────────────────────────
  config.json          ~  a few KB   (HF architecture definition only, NO weights)
  tokenizer files      ~  1 MB
  gsq_weights.pt       ~  4-5 GB     (ALL tensors needed to run the model)
    ├─ quantized Linear layers  → int8 weight + fp16 scale
    └─ everything else          → fp16  (embeddings, norms, lm_head …)
  gsq_metadata.json    ~  a few MB   (sensitivity scores, bit assignments)

At load time
────────────
  1. Build empty model skeleton from config.json  (zero RAM for weights)
  2. Load gsq_weights.pt
  3. Swap Linear → Int8Linear using the int8 tensors
  4. Load remaining fp16 tensors into the skeleton
  → Original 16 GB model is NEVER needed again.

Size targets for Llama-3.1-8B
──────────────────────────────
  robust_bits=4, sensitive_bits=8  →  ~4.5 GB  (default, mixed)
  robust_bits=4, sensitive_bits=4  →  ~4.0 GB  (all INT4)
  robust_bits=8, sensitive_bits=8  →  ~7.0 GB  (all INT8)
"""

import os, json, torch, torch.nn as nn
from pathlib import Path
from dataclasses import dataclass, asdict
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from datasets import load_dataset

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GSQConfig:
    model_path: str
    output_path: str
    n_calibration_samples: int = 50
    max_seq_len: int = 512
    sensitive_ratio: float = 0.25   # top fraction → sensitive_bits, rest → robust_bits
    sensitive_bits: int = 8
    robust_bits: int = 4
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    dataset_split: str = "train"
    calibration_seed: int = 42


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Calibration data
# ─────────────────────────────────────────────────────────────────────────────

def get_calibration_tokens(tokenizer, cfg, device):
    ds      = load_dataset(cfg.dataset_name, cfg.dataset_config, split=cfg.dataset_split)
    text    = "\n\n".join(ds["text"])
    torch.manual_seed(cfg.calibration_seed)
    all_ids = tokenizer(text, return_tensors="pt")["input_ids"][0]
    starts  = torch.randperm(len(all_ids) - cfg.max_seq_len)[:cfg.n_calibration_samples]
    samples = [all_ids[s: s + cfg.max_seq_len].unsqueeze(0).to(device) for s in starts]
    print(f"[GSQ] {len(samples)} calibration chunks × {cfg.max_seq_len} tokens")
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Sensitivity map
# ─────────────────────────────────────────────────────────────────────────────

def compute_sensitivity_map(model, samples, cfg):
    for p in model.parameters():
        p.requires_grad_(True)

    accum = {}
    print(f"[GSQ] {len(samples)} calibration passes …")
    for i, ids in enumerate(samples):
        model.zero_grad()
        model(input_ids=ids, labels=ids).loss.backward()
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Linear) and mod.weight.grad is not None:
                accum[name] = accum.get(name, 0.0) + mod.weight.grad.abs().mean().item()
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(samples)}")

    model.zero_grad()
    for p in model.parameters():
        p.requires_grad_(False)

    sens = {k: v / len(samples) for k, v in accum.items()}
    print(f"[GSQ] Sensitivity computed for {len(sens)} layers")
    return sens


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Bit-width assignment
# ─────────────────────────────────────────────────────────────────────────────

def assign_bit_widths(sensitivity_map, cfg):
    ranked      = sorted(sensitivity_map.items(), key=lambda x: x[1], reverse=True)
    n_sensitive = max(1, int(len(ranked) * cfg.sensitive_ratio))
    bits = {n: (cfg.sensitive_bits if i < n_sensitive else cfg.robust_bits)
            for i, (n, _) in enumerate(ranked)}
    n8 = sum(1 for b in bits.values() if b == cfg.sensitive_bits)
    print(f"[GSQ] {n8} → INT{cfg.sensitive_bits}  |  {len(bits)-n8} → INT{cfg.robust_bits}")
    return bits


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Build the unified weight store
# ─────────────────────────────────────────────────────────────────────────────

def quantize_tensor(w: torch.Tensor, n_bits: int):
    qmax = (2 ** (n_bits - 1)) - 1
    qmin = -(2 ** (n_bits - 1))
    wf = w.float()

    # ROW-wise scaling: one scale per output neuron [out_features, 1]
    amax = wf.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    scale = amax / qmax
    w_int = torch.round(wf / scale).clamp(qmin, qmax).to(torch.int8)

    # Save scale as [out_features] — squeeze the keepdim
    return w_int, scale.squeeze(1).half()


def build_weight_store(model, bit_assignment):
    """
    Returns a single dict with every tensor the model needs:

    For each quantized Linear  → key = layer name (e.g. "model.layers.0.self_attn.q_proj")
      { "quantized": True, "bits": 8, "w": int8, "s": fp16, "b": fp16|None }

    For everything else        → key = full state_dict key (e.g. "model.embed_tokens.weight")
      { "quantized": False, "data": fp16 }
    """
    store = {}

    # Set of layer names (not state_dict keys) that will be quantized
    quant_layer_names = {
        name for name, mod in model.named_modules()
        if isinstance(mod, nn.Linear) and name in bit_assignment
    }

    # Quantized layers
    for name, mod in model.named_modules():
        if name not in quant_layer_names:
            continue
        n_bits        = bit_assignment[name]
        w_int8, scale = quantize_tensor(mod.weight.data, n_bits)
        bias          = mod.bias.data.cpu().half() if mod.bias is not None else None
        store[name]   = {"quantized": True, "bits": n_bits,
                         "w": w_int8.cpu(), "s": scale.cpu(), "b": bias}

    # State-dict keys that belong to quantized layers (skip — already stored above)
    skip_keys = set()
    for name in quant_layer_names:
        skip_keys.add(name + ".weight")
        skip_keys.add(name + ".bias")

    # Everything else (embeddings, norms, lm_head if not quantized, …)
    for key, tensor in model.state_dict().items():
        if key in skip_keys:
            continue
        store[key] = {
            "quantized": False,
            "data": tensor.cpu().half() if tensor.is_floating_point() else tensor.cpu(),
        }

    n_q   = sum(1 for v in store.values() if v["quantized"])
    n_fp  = len(store) - n_q
    n_bytes = sum(
        (v["w"].numel() + v["s"].numel() * 2 + (v["b"].numel() * 2 if v["b"] is not None else 0))
        if v["quantized"] else v["data"].numel() * (1 if not v["data"].is_floating_point() else 2)
        for v in store.values()
    )
    print(f"[GSQ] Store: {n_q} quantized, {n_fp} fp16 tensors — "
          f"estimated {n_bytes/1e9:.2f} GB")
    return store


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Save (architecture config only + weight store + tokenizer)
# ─────────────────────────────────────────────────────────────────────────────

def save_gsq_checkpoint(tokenizer, hf_config, weight_store,
                        sensitivity_map, bit_assignment, cfg):
    out = Path(cfg.output_path)
    out.mkdir(parents=True, exist_ok=True)

    # Architecture definition only — a few KB of JSON, zero weights
    hf_config.save_pretrained(str(out))

    # Tokenizer
    tokenizer.save_pretrained(str(out))

    # All weights in one file
    torch.save(weight_store, out / "gsq_weights.pt")

    # Metadata
    with open(out / "gsq_metadata.json", "w") as f:
        json.dump({
            "quantization_method": "GSQ",
            "config": asdict(cfg),
            "sensitivity_map": {k: float(v) for k, v in sensitivity_map.items()},
            "bit_assignment": bit_assignment,
        }, f, indent=2)

    size_gb = (out / "gsq_weights.pt").stat().st_size / 1e9
    print(f"[GSQ] gsq_weights.pt   : {size_gb:.2f} GB")
    print(f"[GSQ] config.json      : architecture only, no weights")
    print(f"[GSQ] Checkpoint → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_gsq(cfg: GSQConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[GSQ] Device : {device}")
    print(f"[GSQ] Loading {cfg.model_path} …")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Save architecture config before loading weights (it's just JSON)
    hf_config = AutoConfig.from_pretrained(cfg.model_path)

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_path, torch_dtype=torch.float16,
        device_map="auto", trust_remote_code=True,
    )
    model.eval()

    samples         = get_calibration_tokens(tokenizer, cfg, device)
    sensitivity_map = compute_sensitivity_map(model, samples, cfg)
    bit_assignment  = assign_bit_widths(sensitivity_map, cfg)
    weight_store    = build_weight_store(model, bit_assignment)

    # Drop the FP16 model from VRAM before writing to disk
    del model
    torch.cuda.empty_cache()
    print("[GSQ] FP16 model freed from VRAM")

    save_gsq_checkpoint(tokenizer, hf_config, weight_store,
                        sensitivity_map, bit_assignment, cfg)


if __name__ == "__main__":
    cfg = GSQConfig(
        model_path      = "hf_models/Llama-3.1-8B-Instruct",
        output_path     = "hf_models/Llama-3.1-8B-GSQ-int8",
        n_calibration_samples = 50,
        max_seq_len     = 512,
        sensitive_ratio = 0.25,
        sensitive_bits  = 8,
        robust_bits     = 4,
    )
    run_gsq(cfg)