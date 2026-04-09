import argparse
import json
import re
import os
import sys
import torch
import numpy as np
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

occupations = [
    "janitor", "accountant", "chief", "assistant", "carpenter", "teacher", "lawyer",
    "laborer", "designer", "cook", "clerk", "analyst", "cashier", "guard", "writer",
    "housekeeper", "ceo", "hairdresser", "cleaner", "counselor", "developer", "manager",
    "mover", "editor", "farmer", "attendant", "baker", "receptionist", "tailor", "driver",
    "salesperson", "mechanic", "librarian", "physician", "sheriff", "nurse", "supervisor", "auditor",
    "secretary", "construction worker"
]


def load_file(path):
    with open(path) as f:
        lines = f.readlines()
    return lines


def get_logprob(model, tokenizer, text):
    device = next(model.parameters()).device
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    return -outputs.loss.item() * inputs["input_ids"].shape[1]


def collect_winobias_scores(model, tokenizer, lines, stereotype_type):
    """
    Collect y_true (correctness), y_score (logprob difference), and group (gender).
    Returns: (y_true, y_score, gender_group)
    """
    y_true_list = []
    y_score_list = []
    gender_groups = []

    for line in tqdm(lines, desc=f"Evaluating WinoBias ({stereotype_type})"):
        text = re.sub(r'^\d+\s+', '', line).strip()
        if not text:
            continue

        brackets = re.findall(r'\[([^\]]+)\]', text)
        if len(brackets) < 2:
            continue

        target_answer = brackets[0]
        pronoun = brackets[-1]

        clean_text = text.replace('[', '').replace(']', '')

        found_occ = []
        for occ in occupations:
            if re.search(r'\b(?:the|a|an)\s+' + occ + r'\b', clean_text, re.IGNORECASE):
                found_occ.append(occ)

        if len(found_occ) < 2:
            continue

        opt_A = "The " + found_occ[0]
        opt_B = "The " + found_occ[1]

        prompt = f"Context: {clean_text}\nQuestion: Who does '{pronoun}' refer to?\nAnswer:"

        score_A = get_logprob(model, tokenizer, f"{prompt} {opt_A}")
        score_B = get_logprob(model, tokenizer, f"{prompt} {opt_B}")

        pred = opt_A if score_A > score_B else opt_B
        is_correct = 1 if pred.lower() == target_answer.lower() else 0

        # Score: difference in log-probs (positive if predicted correctly)
        score_diff = score_A - score_B if pred == opt_A else score_B - score_A

        y_true_list.append(is_correct)
        y_score_list.append(score_diff)

        # Group by gender
        if pronoun.lower() in ["he", "his", "him"]:
            gender_groups.append("male")
        elif pronoun.lower() in ["she", "her", "hers"]:
            gender_groups.append("female")

    return y_true_list, y_score_list, gender_groups


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

    y_pred_before = (y_score >= 0.0).astype(int)
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
    parser = argparse.ArgumentParser(description="WinoBias FROC runner with strict/pragmatic modes")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_type", default="standard", choices=["standard", "awq", "gsq"])
    parser.add_argument("--dataset_dir", default="benchmark_datasets/wino_bias")
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--froc-mode", default="pragmatic", choices=["strict", "pragmatic", "both"])
    parser.add_argument("--output_dir", default="outputs/winobias")
    args = parser.parse_args()

    if args.model_type == "gsq":
        model, tokenizer, metadata = load_gsq_model(args.model_path, device_map="auto")
    elif args.model_type == "awq":
        model, tokenizer = load_model(args.model_path, load_type="AWQ")
    else:
        model, tokenizer = load_model(args.model_path, load_type="standard")

    pro1 = load_file(f"{args.dataset_dir}/pro_stereotyped_type1.test")
    pro2 = load_file(f"{args.dataset_dir}/pro_stereotyped_type2.test")
    anti1 = load_file(f"{args.dataset_dir}/anti_stereotyped_type1.test")
    anti2 = load_file(f"{args.dataset_dir}/anti_stereotyped_type2.test")

    print("Collecting WinoBias scores...")
    p1_true, p1_score, p1_gender = collect_winobias_scores(model, tokenizer, pro1, "pro_type1")
    p2_true, p2_score, p2_gender = collect_winobias_scores(model, tokenizer, pro2, "pro_type2")
    a1_true, a1_score, a1_gender = collect_winobias_scores(model, tokenizer, anti1, "anti_type1")
    a2_true, a2_score, a2_gender = collect_winobias_scores(model, tokenizer, anti2, "anti_type2")

    # Combine all data
    y_true = np.array(p1_true + p2_true + a1_true + a2_true)
    y_score = np.array(p1_score + p2_score + a1_score + a2_score)
    gender_group = np.array(p1_gender + p2_gender + a1_gender + a2_gender)

    if len(y_true) == 0:
        print("No samples collected for FROC. Exiting.")
        return

    # Run FROC modes
    modes_to_run = ["strict", "pragmatic"] if args.froc_mode == "both" else [args.froc_mode]
    model_basename = os.path.basename(os.path.normpath(args.model_path))

    for mode in modes_to_run:
        mode_output_dir = os.path.join(args.output_dir, f"phase23_{mode}", f"{model_basename}_{args.model_type}")
        os.makedirs(mode_output_dir, exist_ok=True)

        result = _run_single_mode(y_true, y_score, gender_group, mode, args, args.model_type, mode_output_dir)

        metrics_df = result["metrics_df"]
        roc_gap_df = result["roc_gap_df"]
        thresholds_by_model = result["thresholds"]

        metrics_df.to_csv(os.path.join(mode_output_dir, "metrics_before_after.csv"), index=False)
        roc_gap_df.to_csv(os.path.join(mode_output_dir, "roc_gap.csv"), index=False)

        with open(os.path.join(mode_output_dir, "thresholds.json"), "w", encoding="utf-8") as f:
            json.dump(thresholds_by_model, f, indent=2)

        summary = {
            "num_samples": int(len(y_true)),
            "num_pro_stereotyped": int(len(p1_true) + len(p2_true)),
            "num_anti_stereotyped": int(len(a1_true) + len(a2_true)),
            "froc_mode": mode,
            "models": list(result["metrics_before_after"].keys()),
            "metrics_before_after": metrics_df.to_dict(orient="records"),
            "roc_gap": roc_gap_df.to_dict(orient="records"),
        }
        with open(os.path.join(mode_output_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        sanity = {
            "num_samples": int(len(y_true)),
            "num_groups": int(len(np.unique(gender_group))),
            "group_counts": {str(g): int(np.sum(gender_group == g)) for g in np.unique(gender_group)},
            "epsilon": float(args.epsilon),
            "froc_mode": mode,
        }
        with open(os.path.join(mode_output_dir, "sanity_report.json"), "w", encoding="utf-8") as f:
            json.dump(sanity, f, indent=2)

        print(f"WinoBias FROC {mode} mode complete. Outputs written to: {mode_output_dir}")

    print(f"All FROC modes complete.")


if __name__ == "__main__":
    main()
