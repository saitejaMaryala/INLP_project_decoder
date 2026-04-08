"""
GSQ Model Loader
----------------
Load a model saved by gsq_quantize.py and run inference.

The saved model has weights already reconstructed to FP16
(simulated quantization), so loading is identical to a standard
HuggingFace load — we just also read gsq_metadata.json to surface
stats and allow optional exact re-quantization.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM


# ---------------------------------------------------------------------------
# Core loader
# ---------------------------------------------------------------------------

def load_gsq_model(model_path: str, device_map: str = "auto"):
    """
    Load a GSQ-quantized model.

    Parameters
    ----------
    model_path : str
        Path to the directory produced by gsq_quantize.py.
    device_map : str
        HuggingFace device_map argument. Use "auto" to spread across GPUs,
        or "cuda:0" / "cpu" for a specific device.

    Returns
    -------
    model       : AutoModelForCausalLM  (weights in FP16)
    tokenizer   : AutoTokenizer
    metadata    : dict  (GSQ config, sensitivity map, bit assignment)
    """
    path = Path(model_path)

    # ── Load metadata ────────────────────────────────────────────────────────
    meta_path = path / "gsq_metadata.json"
    metadata = {}
    if meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)
        _print_gsq_summary(metadata)
    else:
        print(f"[GSQ Loader] Warning: gsq_metadata.json not found in {path}. "
              "Loading weights as-is.")

    # ── Load model & tokenizer ───────────────────────────────────────────────
    print(f"[GSQ Loader] Loading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(str(path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[GSQ Loader] Loading model weights (device_map={device_map!r}) …")
    model = AutoModelForCausalLM.from_pretrained(
        str(path),
        torch_dtype=torch.float16,
        device_map=device_map,
        trust_remote_code=True,
    )
    model.eval()

    print("[GSQ Loader] Model ready.\n")
    return model, tokenizer, metadata


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_gsq_summary(metadata: dict) -> None:
    cfg = metadata.get("config", {})
    ba  = metadata.get("bit_assignment", {})

    n8 = sum(1 for b in ba.values() if b == cfg.get("sensitive_bits", 8))
    n4 = sum(1 for b in ba.values() if b == cfg.get("robust_bits",    4))
    total = len(ba)

    # Effective average bits
    avg_bits = (n8 * cfg.get("sensitive_bits", 8) + n4 * cfg.get("robust_bits", 4)) / max(total, 1)

    print("=" * 54)
    print("  GSQ Model Summary")
    print("=" * 54)
    print(f"  Base model      : {cfg.get('model_path', 'N/A')}")
    print(f"  Calib samples   : {cfg.get('n_calibration_samples', 'N/A')}")
    print(f"  Sensitive ratio : {cfg.get('sensitive_ratio', 'N/A'):.0%}")
    print(f"  INT8 layers     : {n8:>4}  ({n8/max(total,1):.1%})")
    print(f"  INT4 layers     : {n4:>4}  ({n4/max(total,1):.1%})")
    print(f"  Avg bit-width   : {avg_bits:.2f}")
    print("=" * 54)


def get_most_sensitive_layers(metadata: dict, top_k: int = 10) -> list[tuple[str, float]]:
    """Return the top-k most sensitive layer names and their scores."""
    sm = metadata.get("sensitivity_map", {})
    ranked = sorted(sm.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


# ---------------------------------------------------------------------------
# Inference helper (mirrors your existing generate loop)
# ---------------------------------------------------------------------------

def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    do_sample: bool = True,
) -> str:
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Main — drop-in replacement for your existing load_model script
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    GSQ_MODEL_PATH = "hf_models/Llama-3.1-8B-Instruct-GSQ"

    model, tokenizer, metadata = load_gsq_model(GSQ_MODEL_PATH, device_map="auto")

    # ── Optional: inspect top sensitive layers ───────────────────────────────
    print("\nTop-10 most sensitive layers:")
    for name, score in get_most_sensitive_layers(metadata, top_k=10):
        bits = metadata["bit_assignment"].get(name, "?")
        print(f"  [{bits}-bit]  {name:<60}  score={score:.6f}")

    # ── Inference ────────────────────────────────────────────────────────────
    prompt = "Once upon a time"
    print(f"\nPrompt: {prompt!r}\n")
    result = generate(model, tokenizer, prompt, max_new_tokens=80)
    print("Generated:")
    print(result)