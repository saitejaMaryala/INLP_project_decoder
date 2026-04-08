import argparse
import json
import numpy as np
import torch
import os
from tqdm import tqdm

from utils.load_model import load_model
from gsq_quant.gsq_load import load_gsq_model


def get_logprob(model, tokenizer, text):
    device = next(model.parameters()).device
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    # Return total sequence log-probability
    return -outputs.loss.item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--dataset_path", default="benchmark_datasets/stereo_set/stereo_set.json")
    args = parser.parse_args()

    # model, tokenizer = load_model(args.model_path)
    model, tokenizer, metadata = load_gsq_model(args.model_path)
    data = json.load(open(args.dataset_path))

    stereo_scores = []
    anti_scores = []
    prefer_stereo = 0

    for sample in tqdm(data["data"]["intrasentence"], desc='Evaluating StereoSet'):
        sentences = sample["sentences"]
        
        stereo = None
        anti = None
        for s in sentences:
            if s["gold_label"] == "stereotype":
                stereo = s["sentence"]
            elif s["gold_label"] == "anti-stereotype":
                anti = s["sentence"]
                
        if not stereo or not anti:
            continue

        s1 = get_logprob(model, tokenizer, stereo)
        s2 = get_logprob(model, tokenizer, anti)

        stereo_scores.append(s1)
        anti_scores.append(s2)

        if s1 > s2:
            prefer_stereo += 1

    N = len(stereo_scores)
    stereotype_score = prefer_stereo / N if N > 0 else 0

    mu_more = np.mean(stereo_scores)
    mu_less = np.mean(anti_scores)
    
    var_more = np.var(stereo_scores, ddof=1)
    var_less = np.var(anti_scores, ddof=1)
    sigma = np.sqrt((var_more + var_less) / 2)

    cohen_d = (mu_more - mu_less) / sigma if sigma != 0 else 0

    result = {
        "stereotype_score": stereotype_score,
        "cohen_d": float(cohen_d),
        "total_samples": N
    }

    model_name = os.path.basename(os.path.normpath(args.model_path))
    os.makedirs("results/stereoset", exist_ok=True)
    save_path = f"results/stereoset/{model_name}.json"

    with open(save_path, "w") as f:
        json.dump(result, f, indent=4)

    print(f"[{model_name}] StereoSet Results: {result}")


if __name__ == "__main__":
    main()