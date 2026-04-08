from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = "hf_models/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"

tokenizer = AutoTokenizer.from_pretrained(model_path)

model = AutoAWQForCausalLM.from_quantized(
    model_path,
    fuse_layers=True,
    device_map={"":0}
)

prompt = "Once upon a time"
device = next(model.parameters()).device
inputs = tokenizer(prompt, return_tensors="pt").to(device)

outputs = model.generate(**inputs, max_new_tokens=50)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))