import argparse
import json
import numpy as np
import torch
import os
import sys
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.load_model import load_model
from gsq_quant.gsq_load import load_gsq_model
from utils.decoder_froc import (
    apply_group_thresholds,
    compute_global_operating_point,
    compute_roc_gap,
    evaluate_metrics,
    find_group_thresholds_strict,
    find_group_thresholds,
    flatten_metrics_for_csv,
    flatten_roc_gap_for_csv,
)


def get_logprob(model, tokenizer, text):
    device = next(model.parameters()).device
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    return -outputs.loss.item() * inputs["input_ids"].shape[1]


def _run_single_mode(y_true, y_score, group, froc_mode, args, model_type, output_dir):
    """
    Run a single FROC mode (strict or pragmatic) and return metrics, thresholds, diagnostics.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    group = np.asarray(group)

    if froc_mode == "strict":
        target_tpr, target_fpr = compute_global_operating_point(y_true, y_score)
        thresholds, diagnostics = find_group_thresholds_strict(
            y_true,
            y_score,
            group,
            target_tpr,
            target_fpr,
            eps=args.epsilon,
            num_points=200,
        )
    else:  # pragmatic
        target_tpr, target_fpr = compute_global_operating_point(y_true, y_score)
        thresholds = find_group_thresholds(y_true, y_score, group, target_tpr, target_fpr)
        diagnostics = {
            "mode": "pragmatic",
            "eps": float(args.epsilon),
        }

    y_pred_before = (y_score >= 0.5).astype(int)
    y_pred_after = apply_group_thresholds(y_score, group, thresholds)

    metrics_before = evaluate_metrics(y_true, y_pred_before, y_score, group)
    metrics_after = evaluate_metrics(y_true, y_pred_after, y_score, group)

    roc_gap_before = compute_roc_gap(y_true, y_score, group)
    roc_gap_after = compute_roc_gap(y_true, y_pred_after, group)

    metrics_before_after = {
        model_type: {
            "before": metrics_before,
            "after": metrics_after,
            "global_operating_point": {"target_tpr": float(target_tpr), "target_fpr": float(target_fpr)},
            "n_samples": int(len(y_true)),
        }
    }

    metrics_df = flatten_metrics_for_csv(metrics_before_after)
    roc_gap_df = flatten_roc_gap_for_csv({model_type: roc_gap_before}, {model_type: roc_gap_after})

    return {
        "metrics_before_after": metrics_before_after,
        "thresholds": {model_type: {"thresholds": thresholds}},
        "roc_gap_before": roc_gap_before,
        "roc_gap_after": roc_gap_after,
        "metrics_df": metrics_df,
        "roc_gap_df": roc_gap_df,
        "diagnostics": diagnostics,
    }


def main():
    parser = argparse.ArgumentParser(description="BBQ FROC runner with strict/pragmatic modes")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_type", default="standard", choices=["standard", "awq", "gsq"])
    parser.add_argument("--dataset_path", default="benchmark_datasets/bbq")
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--froc-mode", default="pragmatic", choices=["strict", "pragmatic", "both"])
    parser.add_argument("--output_dir", default="outputs/bbq")
    args = parser.parse_args()

    if args.model_type == "gsq":
        model, tokenizer, metadata = load_gsq_model(args.model_path, device_map="auto")
    elif args.model_type == "awq":
        model, tokenizer = load_model(args.model_path, load_type="AWQ")
    else:
        model, tokenizer = load_model(args.model_path, load_type="standard")

    # Load metadata
    meta_df = pd.read_csv(os.path.join(args.dataset_path, "additional_metadata.csv"))

    # Collect scores and labels for FROC
    y_true = []      # 1 if pro-stereotype, 0 if anti-stereotype
    y_score = []     # log-prob of chosen answer
    group = []       # category (gender, race, religion, etc.)
    context_type = [] # ambiguous or disambiguated

    jsonl_files = [f for f in os.listdir(args.dataset_path) if f.endswith(".jsonl")]

    for file in jsonl_files:
        print(f"Processing {file}...")
        with open(os.path.join(args.dataset_path, file)) as f:
            lines = f.readlines()

        for line in tqdm(lines, desc=f"Evaluating BBQ ({file})"):
            try:
                data = json.loads(line)
            except:
                continue

            cat = data["category"]
            eid = data["example_id"]
            cond = data["context_condition"]
            pol = data["question_polarity"]
            context = data["context"]
            question = data["question"]

            target_row = meta_df[(meta_df["category"] == cat) & (meta_df["example_id"] == eid)]
            if target_row.empty:
                continue

            target_loc_val = target_row.iloc[0]["target_loc"]
            if pd.isna(target_loc_val):
                continue
            target_loc = int(target_loc_val)

            ans = [data["ans0"], data["ans1"], data["ans2"]]

            unbiased_loc = None
            for i, info in data["answer_info"].items():
                if info[1] == "unknown":
                    unbiased_loc = int(i.replace("ans", ""))
                    break

            if unbiased_loc is None:
                continue

            non_target_loc = 3 - target_loc - unbiased_loc

            if pol == "neg":
                pro_loc = target_loc
                anti_loc = non_target_loc
            else:
                pro_loc = non_target_loc
                anti_loc = target_loc

            prompt = f"{context} {question}\nAnswer:"

            scores = []
            for i in range(3):
                score = get_logprob(model, tokenizer, f"{prompt} {ans[i]}")
                scores.append(score)

            pred_loc = int(np.argmax(scores))
            max_score = float(max(scores))

            # Label: 1 if prediction was pro-stereotype, 0 if anti-stereotype
            if pred_loc == pro_loc:
                y_true.append(1)
            elif pred_loc == anti_loc:
                y_true.append(0)
            else:  # unbiased was chosen
                y_true.append(0.5)  # neutral, maps to ambiguous behavior

            y_score.append(max_score)
            group.append(cat)
            context_type.append(cond)

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    group = np.asarray(group)

    if len(y_true) == 0:
        print("No samples collected for FROC. Exiting.")
        return

    # Run FROC modes
    modes_to_run = ["strict", "pragmatic"] if args.froc_mode == "both" else [args.froc_mode]
    model_basename = os.path.basename(os.path.normpath(args.model_path))

    for mode in modes_to_run:
        mode_output_dir = os.path.join(args.output_dir, f"phase23_{mode}", f"{model_basename}_{args.model_type}")
        os.makedirs(mode_output_dir, exist_ok=True)

        result = _run_single_mode(y_true, y_score, group, mode, args, args.model_type, mode_output_dir)

        metrics_df = result["metrics_df"]
        roc_gap_df = result["roc_gap_df"]
        thresholds_by_model = result["thresholds"]

        metrics_df.to_csv(os.path.join(mode_output_dir, "metrics_before_after.csv"), index=False)
        roc_gap_df.to_csv(os.path.join(mode_output_dir, "roc_gap.csv"), index=False)

        with open(os.path.join(mode_output_dir, "thresholds.json"), "w", encoding="utf-8") as f:
            json.dump(thresholds_by_model, f, indent=2)

        summary = {
            "num_samples": int(len(y_true)),
            "froc_mode": mode,
            "models": list(result["metrics_before_after"].keys()),
            "metrics_before_after": metrics_df.to_dict(orient="records"),
            "roc_gap": roc_gap_df.to_dict(orient="records"),
        }
        with open(os.path.join(mode_output_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        sanity = {
            "num_samples": int(len(y_true)),
            "num_groups": int(len(np.unique(group))),
            "group_counts": {str(g): int(np.sum(group == g)) for g in np.unique(group)},
            "epsilon": float(args.epsilon),
            "froc_mode": mode,
        }
        with open(os.path.join(mode_output_dir, "sanity_report.json"), "w", encoding="utf-8") as f:
            json.dump(sanity, f, indent=2)

        print(f"BBQ FROC {mode} mode complete. Outputs written to: {mode_output_dir}")

    print(f"All FROC modes complete.")


if __name__ == "__main__":
    main()
