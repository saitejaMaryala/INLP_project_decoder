import argparse
import json
import torch
import numpy as np
import os
from datasets import load_from_disk
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

from utils.load_model import load_model
from gsq_quant.gsq_load import load_gsq_model

def build_prompts(sample):
    """Build the base prompt and all 4 choice-appended prompts for one sample."""
    question = sample["question"]
    choices = sample["choices"]

    prompt = f"Question: {question}\n"
    for i, c in enumerate(choices):
        prompt += f"{chr(65 + i)}. {c}\n"
    prompt += "Answer:"

    # Return prompt+choice strings for A, B, C, D
    return [f"{prompt} {ch}" for ch in ["A", "B", "C", "D"]], sample["answer"]


def score_choices_batched(model, tokenizer, choice_texts, batch_size=8):
    """
    Score all choice_texts (N*4 strings) in mini-batches.
    Returns a list of log-likelihoods, one per string.
    """
    all_scores = []

    for i in range(0, len(choice_texts), batch_size):
        batch_texts = choice_texts[i: i + batch_size]

        # Tokenize with padding
        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )

        # input_ids = enc["input_ids"].to(model.device)
        # attention_mask = enc["attention_mask"].to(model.device)

        device = next(model.parameters()).device
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )
            # outputs.loss is mean NLL over non-padding tokens
            # We need per-sample log-likelihoods, so compute manually
            logits = outputs.logits  # (B, T, V)

        # Shift for causal LM: predict token[t] from token[t-1]
        shift_logits = logits[:, :-1, :].contiguous()   # (B, T-1, V)
        shift_labels = input_ids[:, 1:].contiguous()     # (B, T-1)
        shift_mask = attention_mask[:, 1:].contiguous()  # (B, T-1)

        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        token_losses = loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).view(shift_labels.size())  # (B, T-1)

        # Zero out padding positions and sum
        token_losses = token_losses * shift_mask
        seq_lengths = shift_mask.sum(dim=1)                     # (B,)
        seq_log_likelihoods = -token_losses.sum(dim=1) / seq_lengths  # mean NLL → negate

        all_scores.extend(seq_log_likelihoods.cpu().float().tolist())

    return all_scores


def evaluate_mmlu(model, tokenizer, dataset, inference_batch_size=32):
    """
    Batch across samples: collect (inference_batch_size * 4) strings,
    score them together, then compute accuracy.
    """
    correct = 0
    total = 0

    # Collect all prompts in chunks of `inference_batch_size` samples
    sample_buffer = []
    choice_buffer = []   # flat list of (sample_idx, choice_idx, text)

    def flush(sample_buffer, choice_buffer):
        nonlocal correct, total
        if not choice_buffer:
            return

        texts = [t for _, _, t in choice_buffer]
        scores = score_choices_batched(model, tokenizer, texts, batch_size=inference_batch_size * 4)

        # Group back into 4 choices per sample
        idx = 0
        for s_idx, answer in sample_buffer:
            sample_scores = scores[idx: idx + 4]
            pred_idx = int(np.argmax(sample_scores))
            if pred_idx == answer:
                correct += 1
            total += 1
            idx += 4

    sample_buffer = []
    choice_buffer = []

    for global_idx, sample in enumerate(tqdm(dataset, desc="Evaluating MMLU")):
        choice_texts, answer = build_prompts(sample)
        sample_buffer.append((global_idx, answer))
        for c_idx, text in enumerate(choice_texts):
            choice_buffer.append((global_idx, c_idx, text))

        # Flush when buffer is full
        if len(sample_buffer) >= inference_batch_size:
            flush(sample_buffer, choice_buffer)
            sample_buffer = []
            choice_buffer = []

    # Flush remaining
    flush(sample_buffer, choice_buffer)

    return correct / total if total > 0 else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--dataset_path", default="benchmark_datasets/mmlu")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Samples per batch (tune based on VRAM)")
    args = parser.parse_args()

    # model, tokenizer = load_model(args.model_path)
    model, tokenizer, metadata = load_gsq_model(args.model_path)

    # Fix padding token fo LLAMA 
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    test_path = os.path.join(args.dataset_path, "test")
    dataset = load_from_disk(test_path)

    acc = evaluate_mmlu(model, tokenizer, dataset, inference_batch_size=args.batch_size)

    model_name = os.path.basename(os.path.normpath(args.model_path))
    result = {"accuracy": float(acc)}

    os.makedirs("results/mmlu", exist_ok=True)
    save_path = f"results/mmlu/{model_name}.json"
    with open(save_path, "w") as f:
        json.dump(result, f, indent=4)

    print(f"[{model_name}] MMLU Results: {result}")


if __name__ == "__main__":
    main()