import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from awq import AutoAWQForCausalLM


def load_model(model_path):
    if "AWQ" in model_path:
        model = AutoAWQForCausalLM.from_quantized(
            model_path,
            fuse_layers=True,
            trust_remote_code=True,
            device_map="auto"          # spreads across all visible GPUs
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",         # spreads layers across all 4 GPUs
            torch_dtype=torch.float16,
            trust_remote_code=True,
            attn_implementation="sdpa",  # native PyTorch fast attention
        )

    # Pad token fix (common issue with Llama)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model.eval()
    return model, tokenizer