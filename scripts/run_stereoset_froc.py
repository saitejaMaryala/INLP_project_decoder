import argparse
import json
import os
import sys
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.decoder_froc import (
    apply_group_thresholds,
    compute_roc_gap,
    evaluate_metrics,
    flatten_metrics_for_csv,
    flatten_roc_gap_for_csv,
    load_decoder_model,
    quantize_model_int8,
)


def get_logprob(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    return -outputs.loss.item()


def safe_auc(y_true, y_score):
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return 0.5


def generate_roc_points(y_true, y_scores, k=100):
    """
    Creates a Piece-wise Linear Approximation (PLA) of ROC.
    Returns dict containing thresholds and aligned ROC points.
    """
    y_true = np.asarray(y_true).astype(int)
    y_scores = np.asarray(y_scores).astype(float)

    if y_true.size == 0:
        thresholds = np.linspace(1.0, 0.0, k)
        fpr = np.zeros(k, dtype=float)
        tpr = np.zeros(k, dtype=float)
        fpr[0], tpr[0] = 0.0, 0.0
        fpr[-1], tpr[-1] = 1.0, 1.0
        return {"thresholds": thresholds, "fpr": fpr, "tpr": tpr}

    score_min = float(np.min(y_scores))
    score_max = float(np.max(y_scores))
    if np.isclose(score_min, score_max):
        score_max = score_min + 1e-12
    thresholds = np.linspace(score_max, score_min, k)

    fpr_values = []
    tpr_values = []
    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tn = np.sum((y_true == 0) & (y_pred == 0))

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fpr_values.append(float(fpr))
        tpr_values.append(float(tpr))

    fpr = np.asarray(fpr_values, dtype=float)
    tpr = np.asarray(tpr_values, dtype=float)

    order = np.argsort(fpr)
    fpr = fpr[order]
    tpr = tpr[order]
    thresholds = thresholds[order]

    fpr = np.concatenate(([0.0], fpr, [1.0]))
    tpr = np.concatenate(([0.0], tpr, [1.0]))
    thresholds = np.concatenate(([score_max + 1e-12], thresholds, [score_min - 1e-12]))

    return {"thresholds": thresholds, "fpr": fpr, "tpr": tpr}


def calculate_area(q_from, q_to):
    dx = abs(float(q_from[0]) - float(q_to[0]))
    dy = abs(float(q_from[1]) - float(q_to[1]))
    return 0.5 * dx * dy


def _move_toward_with_l1_limit(q_priv, q_dis, epsilon):
    fpr_p, tpr_p = float(q_priv[0]), float(q_priv[1])
    fpr_d, tpr_d = float(q_dis[0]), float(q_dis[1])

    if epsilon < 0:
        epsilon = 0.0

    delta_f = fpr_p - fpr_d
    delta_t = tpr_p - tpr_d
    current_dist = abs(delta_f) + abs(delta_t)
    if current_dist <= epsilon:
        return (fpr_p, tpr_p)

    remain = current_dist - epsilon
    shift_f = min(abs(delta_f), remain)
    remain -= shift_f
    shift_t = min(abs(delta_t), remain)

    new_fpr = fpr_p - np.sign(delta_f) * shift_f
    new_tpr = tpr_p - np.sign(delta_t) * shift_t
    return (float(np.clip(new_fpr, 0.0, 1.0)), float(np.clip(new_tpr, 0.0, 1.0)))


def apply_froc_transport(roc_privileged, roc_disadvantaged, epsilon):
    """
    Transports a higher ROC curve toward a lower baseline within epsilon L1 distance.
    """
    fair_points = []
    for i in range(len(roc_disadvantaged)):
        q_priv = roc_privileged[i]
        q_dis = roc_disadvantaged[i]
        current_dist = abs(q_priv[0] - q_dis[0]) + abs(q_priv[1] - q_dis[1])

        if current_dist <= epsilon:
            fair_points.append((float(q_priv[0]), float(q_priv[1])))
            continue

        up_shift = _move_toward_with_l1_limit((q_priv[0], q_dis[1]), q_dis, epsilon)
        left_shift = _move_toward_with_l1_limit((q_dis[0], q_priv[1]), q_dis, epsilon)

        if calculate_area(q_priv, up_shift) < calculate_area(q_priv, left_shift):
            fair_points.append(up_shift)
        else:
            fair_points.append(left_shift)

    fair_points = np.asarray(fair_points, dtype=float)
    order = np.argsort(fair_points[:, 0])
    fair_points = fair_points[order]
    fair_points[:, 1] = np.maximum.accumulate(fair_points[:, 1])
    return fair_points


def derive_threshold_from_transport(roc_dict, fair_points):
    original = np.column_stack([roc_dict["fpr"], roc_dict["tpr"]])
    dist = np.sum((original - fair_points) ** 2, axis=1)
    idx = int(np.argmin(dist)) if dist.size else 0
    threshold = float(roc_dict["thresholds"][idx])
    if not np.isfinite(threshold):
        finite = roc_dict["thresholds"][np.isfinite(roc_dict["thresholds"])]
        threshold = float(np.mean(finite)) if finite.size else 0.5
    return threshold


def froc_predict(score, fair_threshold):
    """
    Converts score into prediction using fair threshold.
    """
    return int(float(score) >= float(fair_threshold))


def build_stereoset_binary_records(data, subset="intrasentence", group_field="target", use_context=False, max_samples=None):
    records = []
    samples = data["data"][subset]

    for sample in samples:
        stereo = None
        anti = None
        for sentence in sample["sentences"]:
            if sentence.get("gold_label") == "stereotype":
                stereo = sentence.get("sentence")
            elif sentence.get("gold_label") == "anti-stereotype":
                anti = sentence.get("sentence")

        if not stereo or not anti:
            continue

        group_value = str(sample.get(group_field, "unknown"))
        context = sample.get("context", "") if use_context else ""
        stereo_text = f"{context} {stereo}".strip()
        anti_text = f"{context} {anti}".strip()

        records.append(
            {
                "id": sample.get("id"),
                "group": group_value,
                "stereotype_text": stereo_text,
                "anti_text": anti_text,
            }
        )

        if max_samples is not None and len(records) >= max_samples:
            break

    return records


def score_records(model, tokenizer, records):
    y_true = []
    y_score = []
    group = []

    for row in tqdm(records, desc="Scoring StereoSet"):
        logp_stereo = get_logprob(model, tokenizer, row["stereotype_text"])
        logp_anti = get_logprob(model, tokenizer, row["anti_text"])

        max_logp = max(logp_stereo, logp_anti)
        p_stereo = np.exp(logp_stereo - max_logp) / (np.exp(logp_stereo - max_logp) + np.exp(logp_anti - max_logp))
        p_anti = 1.0 - p_stereo

        y_true.extend([1, 0])
        y_score.extend([float(p_stereo), float(p_anti)])
        group.extend([row["group"], row["group"]])

    return np.asarray(y_true, dtype=int), np.asarray(y_score, dtype=float), np.asarray(group, dtype=str)


def learn_froc_thresholds(y_true, y_score, group, epsilon=0.05, k=100, disadvantaged_group=None):
    unique_groups = np.unique(group)
    if unique_groups.size < 2:
        return {str(unique_groups[0]): 0.5} if unique_groups.size == 1 else {}

    auc_per_group = {}
    roc_per_group = {}
    for g in unique_groups:
        mask = group == g
        auc_per_group[str(g)] = safe_auc(y_true[mask], y_score[mask])
        roc_per_group[str(g)] = generate_roc_points(y_true[mask], y_score[mask], k=k)

    if disadvantaged_group is None:
        disadvantaged_group = min(auc_per_group, key=auc_per_group.get)
    elif disadvantaged_group not in auc_per_group:
        raise ValueError(f"Requested disadvantaged group '{disadvantaged_group}' not present in data.")

    dis_roc = roc_per_group[disadvantaged_group]
    dis_points = np.column_stack([dis_roc["fpr"], dis_roc["tpr"]])

    thresholds = {}
    thresholds[disadvantaged_group] = derive_threshold_from_transport(dis_roc, dis_points)

    for g in unique_groups:
        g = str(g)
        if g == disadvantaged_group:
            continue

        priv_roc = roc_per_group[g]
        priv_points = np.column_stack([priv_roc["fpr"], priv_roc["tpr"]])

        n = min(len(priv_points), len(dis_points))
        transported = apply_froc_transport(priv_points[:n], dis_points[:n], epsilon=epsilon)
        threshold = derive_threshold_from_transport(
            {
                "fpr": priv_roc["fpr"][:n],
                "tpr": priv_roc["tpr"][:n],
                "thresholds": priv_roc["thresholds"][:n],
            },
            transported,
        )
        thresholds[g] = threshold

    return thresholds


def evaluate_before_after(y_true, y_score, group, thresholds):
    y_pred_before = (y_score >= 0.5).astype(int)
    y_pred_after = apply_group_thresholds(y_score, group, thresholds)

    metrics_before = evaluate_metrics(y_true, y_pred_before, y_score, group)
    metrics_after = evaluate_metrics(y_true, y_pred_after, y_score, group)

    return metrics_before, metrics_after, y_pred_after


def plot_roc_curves_by_group(y_true, y_score, group, title, save_path):
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    for g in np.unique(group):
        mask = group == g
        roc = generate_roc_points(y_true[mask], y_score[mask], k=120)
        ax.plot(roc["fpr"], roc["tpr"], label=str(g), linewidth=2)
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="StereoSet FROC runner with geometric ROC transport")
    parser.add_argument("--model_name", default="gpt2")
    parser.add_argument("--dataset_path", default="benchmark_datasets/stereo_set/stereo_set.json")
    parser.add_argument("--subset", default="intrasentence", choices=["intrasentence", "intersentence"])
    parser.add_argument("--group_field", default="target", choices=["target", "bias_type"])
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--max_samples", type=int, default=400)
    parser.add_argument("--smoke_test", action="store_true", help="Run a fast sanity pass with a small sample size")
    parser.add_argument("--smoke_samples", type=int, default=20, help="Number of StereoSet pairs for smoke test")
    parser.add_argument("--use_context", action="store_true")
    parser.add_argument("--disadvantaged_group", default=None)
    parser.add_argument("--output_dir", default="outputs/decoder_phase5/stereoset")
    args = parser.parse_args()

    if args.smoke_test:
        args.max_samples = int(args.smoke_samples)
        if args.max_samples <= 0:
            raise ValueError("smoke_samples must be positive.")

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = build_stereoset_binary_records(
        data,
        subset=args.subset,
        group_field=args.group_field,
        use_context=args.use_context,
        max_samples=args.max_samples,
    )
    if len(records) == 0:
        raise ValueError("No valid stereotype/anti-stereotype pairs found in selected StereoSet subset.")

    tokenizer_fp32, model_fp32 = load_decoder_model(args.model_name, device=args.device, torch_dtype=torch.float32)
    y_true, y_score_fp32, group = score_records(model_fp32, tokenizer_fp32, records)

    results = {
        "fp32": {"y_true": y_true, "y_score": y_score_fp32, "group": group}
    }

    if args.device == "cuda" and torch.cuda.is_available():
        try:
            tokenizer_fp16, model_fp16 = load_decoder_model(args.model_name, device="cuda", torch_dtype=torch.float16)
            _, y_score_fp16, _ = score_records(model_fp16, tokenizer_fp16, records)
            results["fp16"] = {"y_true": y_true, "y_score": y_score_fp16, "group": group}
        except Exception as exc:
            warnings.warn(f"FP16 variant skipped: {exc}")

    tokenizer_int8, model_int8 = load_decoder_model(args.model_name, device="cpu", torch_dtype=torch.float32)
    model_int8 = quantize_model_int8(model_int8)
    _, y_score_int8, _ = score_records(model_int8, tokenizer_int8, records)
    results["int8"] = {"y_true": y_true, "y_score": y_score_int8, "group": group}

    metrics_before_after = {}
    thresholds_by_model = {}
    roc_gap_before = {}
    roc_gap_after = {}

    for model_key, payload in results.items():
        y_true_m = payload["y_true"]
        y_score_m = payload["y_score"]
        group_m = payload["group"]

        thresholds = learn_froc_thresholds(
            y_true_m,
            y_score_m,
            group_m,
            epsilon=args.epsilon,
            k=args.k,
            disadvantaged_group=args.disadvantaged_group,
        )

        metrics_before, metrics_after, y_pred_after = evaluate_before_after(y_true_m, y_score_m, group_m, thresholds)
        metrics_before_after[model_key] = {
            "before": metrics_before,
            "after": metrics_after,
            "global_operating_point": {"target_tpr": float("nan"), "target_fpr": float("nan")},
            "n_samples": int(len(y_true_m)),
        }
        thresholds_by_model[model_key] = {"thresholds": thresholds}

        roc_gap_before[model_key] = compute_roc_gap(y_true_m, y_score_m, group_m)
        roc_gap_after[model_key] = compute_roc_gap(y_true_m, y_pred_after, group_m)

        plot_roc_curves_by_group(
            y_true_m,
            y_score_m,
            group_m,
            title=f"StereoSet ROC by Group ({model_key}, before FROC)",
            save_path=os.path.join(args.output_dir, f"roc_curves_{model_key}_before.png"),
        )
        plot_roc_curves_by_group(
            y_true_m,
            y_pred_after,
            group_m,
            title=f"StereoSet ROC by Group ({model_key}, after FROC)",
            save_path=os.path.join(args.output_dir, f"roc_curves_{model_key}_after.png"),
        )

    metrics_df = flatten_metrics_for_csv(metrics_before_after)
    roc_gap_df = flatten_roc_gap_for_csv(roc_gap_before, roc_gap_after)

    metrics_df.to_csv(os.path.join(args.output_dir, "metrics_before_after.csv"), index=False)
    roc_gap_df.to_csv(os.path.join(args.output_dir, "roc_gap.csv"), index=False)

    with open(os.path.join(args.output_dir, "thresholds.json"), "w", encoding="utf-8") as f:
        json.dump(thresholds_by_model, f, indent=2)

    summary = {
        "num_pairs": len(records),
        "smoke_test": bool(args.smoke_test),
        "models": list(results.keys()),
        "metrics_before_after": metrics_df.to_dict(orient="records"),
        "roc_gap": roc_gap_df.to_dict(orient="records"),
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    sanity = {
        "num_pairs": int(len(records)),
        "num_binary_samples": int(len(y_true)),
        "num_groups": int(len(np.unique(group))),
        "group_counts": {str(g): int(np.sum(group == g)) for g in np.unique(group)},
        "smoke_test": bool(args.smoke_test),
        "epsilon": float(args.epsilon),
        "k": int(args.k),
    }
    with open(os.path.join(args.output_dir, "sanity_report.json"), "w", encoding="utf-8") as f:
        json.dump(sanity, f, indent=2)

    print(f"StereoSet FROC complete. Outputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()