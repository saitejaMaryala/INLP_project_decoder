"""
GSQ Model Loader
----------------
Loads a self-contained GSQ checkpoint. The original 16 GB FP16 model
is never needed.

Fixes vs previous version
──────────────────────────
  • load_state_dict(..., assign=True)  — meta tensors are REPLACED by the
    checkpoint tensors rather than copied into them (copying from/into a
    meta tensor is a no-op, which is why weights were silently ignored).
  • No dispatch_model on a meta-device skeleton — we materialize on CPU
    first, then move to the target device via HF's device_map logic.
"""

import os, json, warnings
import torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from typing import Optional
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Suppress the no-op meta-copy warning — we handle it correctly with assign=True
warnings.filterwarnings("ignore", message=".*copying from a non-meta.*")


# ─────────────────────────────────────────────────────────────────────────────
# Int8Linear
# ─────────────────────────────────────────────────────────────────────────────

class Int8Linear(nn.Module):
    """
    Drop-in nn.Linear replacement with INT8 weight storage.
    Dequantizes on-the-fly:  w_fp = weight_int8 * scale[:, None]
    """
    def __init__(self, w_int8: torch.Tensor, scale: torch.Tensor,
                 bias: Optional[torch.Tensor], n_bits: int):
        super().__init__()
        self.n_bits = n_bits
        self.register_buffer("weight_int8", w_int8)
        self.register_buffer("scale", scale)
        self.register_buffer("bias", bias.half() if bias is not None else None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # scale is [out_features], unsqueeze to [out_features, 1] for broadcast
        w = self.weight_int8.float() * self.scale.float().unsqueeze(1)
        out = F.linear(x.float(), w,
                       self.bias.float() if self.bias is not None else None)
        return out.to(x.dtype)

    def extra_repr(self):
        o, i = self.weight_int8.shape
        return f"in={i}, out={o}, bits={self.n_bits}, bias={self.bias is not None}"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_parent(model, dotted_name):
    parts, parent = dotted_name.split("."), model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def _print_summary(metadata):
    cfg  = metadata.get("config", {})
    ba   = metadata.get("bit_assignment", {})
    n8   = sum(1 for b in ba.values() if b == cfg.get("sensitive_bits", 8))
    n4   = len(ba) - n8
    avg  = (n8 * cfg.get("sensitive_bits", 8) + n4 * cfg.get("robust_bits", 4)) / max(len(ba), 1)
    print("=" * 54)
    print("  GSQ Model")
    print("=" * 54)
    print(f"  Base model      : {cfg.get('model_path', 'N/A')}")
    print(f"  Calib samples   : {cfg.get('n_calibration_samples')}")
    print(f"  Sensitive ratio : {cfg.get('sensitive_ratio', 0):.0%}")
    print(f"  INT{cfg.get('sensitive_bits',8)} layers    : {n8:>4}  ({n8/max(len(ba),1):.1%})")
    print(f"  INT{cfg.get('robust_bits',4)} layers    : {n4:>4}  ({n4/max(len(ba),1):.1%})")
    print(f"  Avg bit-width   : {avg:.2f}")
    print("=" * 54)


# ─────────────────────────────────────────────────────────────────────────────
# Core loader
# ─────────────────────────────────────────────────────────────────────────────

def load_gsq_model(gsq_path: str, device_map: str = "auto"):
    """
    Parameters
    ----------
    gsq_path   : directory produced by gsq_quantize.py
    device_map : "auto" | "cuda:0" | "cpu"

    Returns
    -------
    model, tokenizer, metadata
    """
    path = Path(gsq_path)

    # ── Metadata ──────────────────────────────────────────────────────────────
    with open(path / "gsq_metadata.json") as f:
        metadata = json.load(f)
    _print_summary(metadata)
    bit_assignment = metadata["bit_assignment"]

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    print("[GSQ] Loading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(str(path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Step 1: empty skeleton on meta device (~0 RAM) ────────────────────────
    print("[GSQ] Building empty model skeleton from config.json …")
    hf_config = AutoConfig.from_pretrained(str(path))
    with torch.device("meta"):
        skeleton = AutoModelForCausalLM.from_config(hf_config, trust_remote_code=True)

    # skeleton = skeleton.to(torch.float16)
    skeleton = skeleton.to_empty(device="cpu")
    skeleton = skeleton.to(torch.float16)

    # ── Step 2: load weight store ─────────────────────────────────────────────
    print("[GSQ] Loading gsq_weights.pt …")
    store = torch.load(path / "gsq_weights.pt", map_location="cpu", weights_only=False)

    # ── Step 3: inject Int8Linear for quantized layers ────────────────────────
    print("[GSQ] Injecting Int8Linear modules …")
    n_patched = 0
    for key, entry in store.items():
        if not entry["quantized"]:
            continue
        try:
            parent, attr = _get_parent(skeleton, key)
        except AttributeError:
            print(f"  [warn] {key} not in skeleton — skipping")
            continue
        setattr(parent, attr, Int8Linear(
            w_int8=entry["w"], scale=entry["s"],
            bias=entry["b"], n_bits=entry["bits"],
        ))
        n_patched += 1
    print(f"[GSQ] Patched {n_patched} layers with Int8Linear")

    # ── Step 4: build full state dict and load with assign=True ───────────────
    # assign=True replaces meta tensors with real ones instead of trying
    # to copy into them (which is a no-op and the source of the silent bug).
    print("[GSQ] Loading all tensors into skeleton (assign=True) …")
    state_dict = {}

    # fp16 tensors (embeddings, norms, lm_head if unquantized, …)
    for k, v in store.items():
        if not v["quantized"]:
            state_dict[k] = v["data"]

    # Int8Linear buffers — must match the buffer names registered above
    for layer_name, entry in store.items():
        if not entry["quantized"]:
            continue
        state_dict[f"{layer_name}.weight_int8"] = entry["w"]
        state_dict[f"{layer_name}.scale"]       = entry["s"]
        if entry["b"] is not None:
            state_dict[f"{layer_name}.bias"]    = entry["b"]

    missing, unexpected = skeleton.load_state_dict(state_dict, strict=False, assign=True)

    # Filter noise: Int8Linear has no "weight" key, so lm_head.weight may
    # appear missing if lm_head was quantized — that's expected.
    real_missing = [k for k in missing
                    if not any(k.startswith(ln + ".weight") for ln in bit_assignment)]
    real_unexpected = [k for k in unexpected
                       if not any(k.startswith(ln) for ln in bit_assignment)]
    if real_missing:
        print(f"  [warn] Missing   : {real_missing[:5]}")
    if real_unexpected:
        print(f"  [warn] Unexpected: {real_unexpected[:5]}")

    # ── Step 5: move to device ────────────────────────────────────────────────
    # The skeleton is now fully materialized on CPU.
    # Use HF's from_pretrained device placement logic via a small shim.
    print(f"[GSQ] Moving to device ({device_map}) …")

    if device_map == "cpu" or not torch.cuda.is_available():
        model = skeleton.to("cpu")
    elif device_map == "auto":
        # Manually place on GPU(s) using accelerate
        from accelerate import dispatch_model, infer_auto_device_map
        # skeleton is on CPU at this point — safe to dispatch
        device_map_dict = infer_auto_device_map(
            skeleton,
            max_memory={i: torch.cuda.get_device_properties(i).total_memory
                        for i in range(torch.cuda.device_count())},
            no_split_module_classes=["LlamaDecoderLayer"],
        )
        model = dispatch_model(skeleton, device_map=device_map_dict)
    else:
        model = skeleton.to(device_map)

    model.eval()
    print("[GSQ] Model ready.\n")
    return model, tokenizer, metadata


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def generate(model, tokenizer, prompt, max_new_tokens=100,
             temperature=0.8, do_sample=True) -> str:
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

        print("nan:", torch.isnan(logits).any())
        print("inf:", torch.isinf(logits).any())
        print("min:", logits.min())
        print("max:", logits.max())

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def get_most_sensitive_layers(metadata, top_k=10):
    sm = metadata.get("sensitivity_map", {})
    return sorted(sm.items(), key=lambda x: x[1], reverse=True)[:top_k]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    GSQ_PATH = "hf_models/Llama-3.1-8B-GSQ-int8"

    model, tokenizer, metadata = load_gsq_model(GSQ_PATH, device_map="auto")

    print("Top-10 most sensitive layers:")
    for name, score in get_most_sensitive_layers(metadata, top_k=10):
        bits = metadata["bit_assignment"].get(name, "?")
        print(f"  [{bits}-bit]  {name:<60}  score={score:.6f}")

    prompt = "Once upon a time"
    print(f"\nPrompt: {prompt!r}\n")
    print(generate(model, tokenizer, prompt, max_new_tokens=80))