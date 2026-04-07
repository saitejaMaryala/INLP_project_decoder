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
        )

    model.eval()
    return model, tokenizer

if __name__ == "__main__":
    model_path = "/home/sai.teja/gsq_decoder/hf_models/Llama-3.1-8B-Instruct"
    model, tokenizer = load_model(model_path)
    print("Model loaded successfully")

    prompt = "Once upon a time"
    # get the device of the model
    device = next(model.parameters()).device

    # move inputs to that device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.8,
            pad_token_id=tokenizer.eos_token_id,  # important
            eos_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(generated_text)