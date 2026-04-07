import json
import math
import os
import warnings
from itertools import combinations
from typing import Iterable, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
except ImportError:  # pragma: no cover - safe fallback when fairlearn is unavailable.
    demographic_parity_difference = None
    equalized_odds_difference = None


DEFAULT_CANDIDATE_TOKENS = [" toxic", " non-toxic"]


def _get_model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _normalize_group_array(group: Sequence) -> np.ndarray:
    return np.asarray([str(item) for item in group])


def _safe_roc_curve(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    if y_true.size == 0:
        return np.asarray([0.0, 1.0]), np.asarray([0.0, 1.0]), np.asarray([float("inf"), float("-inf")])

    unique_labels = np.unique(y_true)
    if unique_labels.size < 2:
        if unique_labels[0] == 1:
            min_score = float(np.min(y_score)) if y_score.size else 0.0
            return (
                np.asarray([0.0, 0.0, 1.0]),
                np.asarray([0.0, 1.0, 1.0]),
                np.asarray([float("inf"), min_score + 1e-12, min_score - 1e-12]),
            )
        max_score = float(np.max(y_score)) if y_score.size else 1.0
        return (
            np.asarray([0.0, 1.0, 1.0]),
            np.asarray([0.0, 0.0, 1.0]),
            np.asarray([float("inf"), max_score + 1e-12, max_score - 1e-12]),
        )

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    return np.asarray(fpr), np.asarray(tpr), np.asarray(thresholds)


def _normalize_threshold(threshold: float, scores: np.ndarray) -> float:
    if np.isfinite(threshold):
        return float(threshold)
    if threshold > 0:
        return float(np.max(scores) + 1e-12) if scores.size else float(1.0)
    return float(np.min(scores) - 1e-12) if scores.size else float(0.0)


def _selection_rate(y_pred: np.ndarray) -> float:
    if y_pred.size == 0:
        return 0.0
    return float(np.mean(y_pred.astype(float)))


def _fallback_demographic_parity_difference(y_true, y_pred, sensitive_features) -> float:
    groups = _normalize_group_array(sensitive_features)
    y_pred = np.asarray(y_pred).astype(int)
    unique_groups = np.unique(groups)
    if unique_groups.size < 2:
        return 0.0
    rates = []
    for group_name in unique_groups:
        mask = groups == group_name
        rates.append(_selection_rate(y_pred[mask]))
    return float(np.max(rates) - np.min(rates))


def _fallback_equalized_odds_difference(y_true, y_pred, sensitive_features) -> float:
    groups = _normalize_group_array(sensitive_features)
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    unique_groups = np.unique(groups)
    if unique_groups.size < 2:
        return 0.0

    per_group = []
    for group_name in unique_groups:
        mask = groups == group_name
        group_true = y_true[mask]
        group_pred = y_pred[mask]

        positives = group_true == 1
        negatives = group_true == 0
        tpr = float(np.mean(group_pred[positives] == 1)) if np.any(positives) else 0.0
        fpr = float(np.mean(group_pred[negatives] == 1)) if np.any(negatives) else 0.0
        per_group.append((tpr, fpr))

    max_gap = 0.0
    for (tpr_a, fpr_a), (tpr_b, fpr_b) in combinations(per_group, 2):
        max_gap = max(max_gap, abs(tpr_a - tpr_b), abs(fpr_a - fpr_b))
    return float(max_gap)


def load_decoder_model(model_name="gpt2", device="cuda", torch_dtype=None):
    """
    Returns:
        tokenizer, model
    """
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token or tokenizer.sep_token

    model_kwargs = {"trust_remote_code": True}
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype

    if device == "cuda":
        model_kwargs["device_map"] = "auto"
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        model = model.to(device)

    model.eval()
    return tokenizer, model


def build_prompt(text):
    """
    Returns:
        formatted prompt string
    """
    return f"Text: {text}\nQuestion: Is this text toxic? Answer:"


def get_token_logprobs(model, tokenizer, prompt, candidate_tokens):
    """
    Returns:
        dict[token] = log probability
    """
    device = _get_model_device(model)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not prompt_ids:
        prompt_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else [0]

    logprobs = {}
    for candidate in candidate_tokens:
        candidate_ids = tokenizer.encode(candidate, add_special_tokens=False)
        full_ids = prompt_ids + candidate_ids
        input_ids = torch.tensor([full_ids], device=device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids)

        logits = outputs.logits
        token_log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
        gathered = token_log_probs.gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)

        prompt_token_count = len(prompt_ids)
        candidate_logprob = gathered[:, prompt_token_count - 1 :].sum().item() if candidate_ids else 0.0
        logprobs[candidate] = float(candidate_logprob)

    return logprobs


def compute_score(logprobs):
    """
    Returns:
        scalar score = P("toxic")
    """
    if not logprobs:
        return 0.5

    toxic_key = None
    for key in logprobs:
        normalized = key.strip().lower().replace(" ", "")
        if normalized == "toxic":
            toxic_key = key
            break

    if toxic_key is None:
        toxic_key = next(iter(logprobs))

    values = np.asarray(list(logprobs.values()), dtype=float)
    max_logprob = float(np.max(values))
    numerator = math.exp(float(logprobs[toxic_key]) - max_logprob)
    denominator = float(np.sum(np.exp(values - max_logprob)))
    if denominator == 0.0:
        return 0.5
    return float(numerator / denominator)


def compute_scores_batch(model, tokenizer, texts):
    """
    Returns:
        numpy array of scores
    """
    scores = []
    for text in texts:
        prompt = build_prompt(text)
        logprobs = get_token_logprobs(model, tokenizer, prompt, DEFAULT_CANDIDATE_TOKENS)
        scores.append(compute_score(logprobs))
    return np.asarray(scores, dtype=float)


def compute_global_operating_point(y_true, y_score, threshold=0.5):
    """
    Returns:
        target_tpr, target_fpr
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    y_pred = (y_score >= threshold).astype(int)

    tp = float(np.sum((y_true == 1) & (y_pred == 1)))
    fp = float(np.sum((y_true == 0) & (y_pred == 1)))
    fn = float(np.sum((y_true == 1) & (y_pred == 0)))
    tn = float(np.sum((y_true == 0) & (y_pred == 0)))

    target_tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    target_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return float(target_tpr), float(target_fpr)


def compute_group_roc(y_true, y_score, group):
    """
    Returns:
        dict[group] = (fpr, tpr, thresholds)
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    group = _normalize_group_array(group)

    roc_data = {}
    for group_name in np.unique(group):
        mask = group == group_name
        roc_data[str(group_name)] = _safe_roc_curve(y_true[mask], y_score[mask])
    return roc_data


def find_group_thresholds(y_true, y_score, group, target_tpr, target_fpr):
    """
    Returns:
        dict[group] = optimal_threshold
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    group = _normalize_group_array(group)

    thresholds = {}
    for group_name in np.unique(group):
        mask = group == group_name
        fpr, tpr, candidate_thresholds = _safe_roc_curve(y_true[mask], y_score[mask])
        distances = (tpr - target_tpr) ** 2 + (fpr - target_fpr) ** 2
        best_idx = int(np.argmin(distances)) if distances.size else 0
        best_threshold = candidate_thresholds[best_idx] if candidate_thresholds.size else 0.5
        thresholds[str(group_name)] = _normalize_threshold(float(best_threshold), y_score[mask])
    return thresholds


def apply_group_thresholds(y_score, group, thresholds_per_group):
    """
    Returns:
        y_pred_froc (binary predictions)
    """
    y_score = np.asarray(y_score).astype(float)
    group = _normalize_group_array(group)
    y_pred = np.zeros_like(y_score, dtype=int)

    for idx, (score, group_name) in enumerate(zip(y_score, group)):
        threshold = thresholds_per_group.get(str(group_name), 0.5)
        y_pred[idx] = int(score >= threshold)

    return y_pred


def evaluate_metrics(y_true, y_pred, y_score, group):
    """
    Returns:
        dict with accuracy, f1, auc, dpd, eod
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_score = np.asarray(y_score).astype(float)
    group = _normalize_group_array(group)

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError:
        auc = float("nan")

    try:
        if demographic_parity_difference is None:
            raise ImportError
        dpd = demographic_parity_difference(y_true=y_true, y_pred=y_pred, sensitive_features=group)
    except Exception:
        dpd = _fallback_demographic_parity_difference(y_true, y_pred, group)

    try:
        if equalized_odds_difference is None:
            raise ImportError
        eod = equalized_odds_difference(y_true=y_true, y_pred=y_pred, sensitive_features=group)
    except Exception:
        eod = _fallback_equalized_odds_difference(y_true, y_pred, group)

    return {
        "accuracy": float(accuracy),
        "f1": float(f1),
        "auc": float(auc) if np.isfinite(auc) else float("nan"),
        "dpd": float(dpd),
        "eod": float(eod),
    }


def evaluate_metrics_after_froc(y_true, y_pred, y_score, group):
    return evaluate_metrics(y_true, y_pred, y_score, group)


def froc_pipeline(results):
    """
    For each model:
        - compute global target
        - compute thresholds
        - apply thresholds
        - compute metrics before/after

    Returns:
        metrics_before_after
        thresholds_per_model
    """
    metrics_before_after = {}
    thresholds_per_model = {}

    for model_name, payload in results.items():
        y_true = np.asarray(payload["y_true"]).astype(int)
        y_score = np.asarray(payload["y_score"]).astype(float)
        group = _normalize_group_array(payload["group"])

        target_tpr, target_fpr = compute_global_operating_point(y_true, y_score)
        thresholds = find_group_thresholds(y_true, y_score, group, target_tpr, target_fpr)

        y_pred_before = (y_score >= 0.5).astype(int)
        y_pred_after = apply_group_thresholds(y_score, group, thresholds)

        metrics_before = evaluate_metrics(y_true, y_pred_before, y_score, group)
        metrics_after = evaluate_metrics_after_froc(y_true, y_pred_after, y_score, group)

        metrics_before_after[model_name] = {
            "before": metrics_before,
            "after": metrics_after,
            "global_operating_point": {
                "target_tpr": float(target_tpr),
                "target_fpr": float(target_fpr),
            },
            "n_samples": int(len(y_true)),
        }
        thresholds_per_model[model_name] = {
            "global_operating_point": {
                "target_tpr": float(target_tpr),
                "target_fpr": float(target_fpr),
            },
            "thresholds": thresholds,
        }

    return metrics_before_after, thresholds_per_model


def interpolate_roc(fpr, tpr, num_points=100):
    """
    Returns:
        common_fpr, interpolated_tpr
    """
    common_fpr = np.linspace(0.0, 1.0, num_points)
    fpr = np.asarray(fpr, dtype=float)
    tpr = np.asarray(tpr, dtype=float)
    if fpr.size == 0 or tpr.size == 0:
        return common_fpr, np.zeros_like(common_fpr)

    order = np.argsort(fpr)
    fpr = fpr[order]
    tpr = tpr[order]
    interpolated_tpr = np.interp(common_fpr, fpr, tpr)
    return common_fpr, interpolated_tpr


def compute_roc_gap(y_true, y_score, group):
    """
    Returns:
        scalar ROC gap (float)
    """
    group = _normalize_group_array(group)
    unique_groups = np.unique(group)
    if unique_groups.size < 2:
        return 0.0

    roc_data = compute_group_roc(y_true, y_score, group)
    interpolated = []
    for group_name in unique_groups:
        fpr, tpr, _ = roc_data[str(group_name)]
        _, tpr_interp = interpolate_roc(fpr, tpr)
        interpolated.append(tpr_interp)

    if len(interpolated) < 2:
        return 0.0

    gaps = []
    for idx_a, idx_b in combinations(range(len(interpolated)), 2):
        gaps.append(float(np.mean((interpolated[idx_a] - interpolated[idx_b]) ** 2)))
    return float(np.mean(gaps)) if gaps else 0.0


def compute_roc_gap_after_froc(y_true, y_score, group, thresholds):
    """
    Applies thresholds and recomputes ROC gap
    """
    y_pred_froc = apply_group_thresholds(y_score, group, thresholds)
    return compute_roc_gap(y_true, y_pred_froc, group)


def roc_analysis_pipeline(results, thresholds_per_model):
    """
    Returns:
        roc_gap_before
        roc_gap_after
    """
    roc_gap_before = {}
    roc_gap_after = {}

    for model_name, payload in results.items():
        y_true = np.asarray(payload["y_true"]).astype(int)
        y_score = np.asarray(payload["y_score"]).astype(float)
        group = _normalize_group_array(payload["group"])

        roc_gap_before[model_name] = float(compute_roc_gap(y_true, y_score, group))
        roc_gap_after[model_name] = float(
            compute_roc_gap_after_froc(
                y_true,
                y_score,
                group,
                thresholds_per_model[model_name]["thresholds"],
            )
        )

    return roc_gap_before, roc_gap_after


def _metrics_dict_to_frame(metrics):
    rows = []
    for model_name, payload in metrics.items():
        for phase in ["before", "after"]:
            row = {"model": model_name, "phase": phase}
            row.update(payload[phase])
            row["target_tpr"] = payload["global_operating_point"]["target_tpr"]
            row["target_fpr"] = payload["global_operating_point"]["target_fpr"]
            row["n_samples"] = payload["n_samples"]
            rows.append(row)
    return pd.DataFrame(rows)


def plot_fairness_comparison(metrics):
    """Plot before vs after metrics for each model."""
    frame = _metrics_dict_to_frame(metrics)
    metric_names = ["accuracy", "f1", "auc", "dpd", "eod"]
    model_names = list(frame["model"].unique())
    phase_order = ["before", "after"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    axes = axes.flatten()

    for axis_index, metric_name in enumerate(metric_names):
        axis = axes[axis_index]
        for phase in phase_order:
            phase_frame = frame[frame["phase"] == phase]
            values = [phase_frame[phase_frame["model"] == model][metric_name].iloc[0] for model in model_names]
            offset = -0.18 if phase == "before" else 0.18
            x = np.arange(len(model_names)) + offset
            axis.bar(x, values, width=0.32, label=phase)

        axis.set_title(metric_name.upper())
        axis.set_xticks(np.arange(len(model_names)))
        axis.set_xticklabels(model_names, rotation=20, ha="right")
        axis.grid(axis="y", alpha=0.25)
        axis.legend()

    axes[-1].axis("off")
    fig.suptitle("Fairness and Utility Before vs After FROC")
    return fig


def plot_roc_curves(y_true, y_score, group, title):
    """Plot ROC curves for each sensitive group."""
    roc_data = compute_group_roc(y_true, y_score, group)
    fig, axis = plt.subplots(figsize=(8, 6), constrained_layout=True)

    for group_name, (fpr, tpr, _) in roc_data.items():
        axis.plot(fpr, tpr, linewidth=2, label=f"Group {group_name}")

    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.set_title(title)
    axis.legend()
    axis.grid(alpha=0.25)
    return fig


def plot_roc_gap(roc_gap_before, roc_gap_after):
    """Compare ROC gap before and after FROC."""
    model_names = list(roc_gap_before.keys())
    x = np.arange(len(model_names))

    fig, axis = plt.subplots(figsize=(8, 6), constrained_layout=True)
    axis.bar(x - 0.18, [roc_gap_before[m] for m in model_names], width=0.36, label="before")
    axis.bar(x + 0.18, [roc_gap_after[m] for m in model_names], width=0.36, label="after")
    axis.set_xticks(x)
    axis.set_xticklabels(model_names, rotation=20, ha="right")
    axis.set_ylabel("ROC Gap")
    axis.set_title("ROC Gap Before vs After FROC")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    return fig


def load_decoder_data(data_path: str) -> pd.DataFrame:
    """Load a text/label/group dataset from CSV, JSON, or JSONL."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    lower_path = data_path.lower()
    if lower_path.endswith(".csv"):
        frame = pd.read_csv(data_path)
    elif lower_path.endswith(".jsonl"):
        frame = pd.read_json(data_path, lines=True)
    elif lower_path.endswith(".json"):
        with open(data_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict) and {"text", "label", "group"}.issubset(payload.keys()):
            frame = pd.DataFrame(payload)
        elif isinstance(payload, list):
            frame = pd.DataFrame(payload)
        else:
            raise ValueError("JSON input must contain either a list of records or text/label/group arrays.")
    else:
        raise ValueError("Unsupported input format. Use CSV, JSON, or JSONL.")

    required_columns = {"text", "label", "group"}
    missing_columns = required_columns - set(frame.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    frame = frame.loc[:, ["text", "label", "group"]].copy()
    frame["text"] = frame["text"].astype(str)
    frame["label"] = pd.to_numeric(frame["label"], errors="coerce")
    frame["group"] = frame["group"].astype(str)
    frame = frame.dropna(subset=["label", "text", "group"])
    frame["label"] = frame["label"].astype(int)
    return frame.reset_index(drop=True)


def quantize_model_int8(model):
    """
    Returns:
        quantized model
    """
    model = model.to("cpu")
    try:
        quantized_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        quantized_model.eval()
        return quantized_model
    except Exception as exc:  # pragma: no cover - best-effort fallback for unsupported checkpoints.
        warnings.warn(f"INT8 quantization failed, falling back to the CPU model: {exc}")
        model.eval()
        return model


def flatten_metrics_for_csv(metrics_before_after):
    rows = []
    for model_name, payload in metrics_before_after.items():
        for phase in ["before", "after"]:
            row = {
                "model": model_name,
                "phase": phase,
                "n_samples": payload["n_samples"],
                "target_tpr": payload["global_operating_point"]["target_tpr"],
                "target_fpr": payload["global_operating_point"]["target_fpr"],
            }
            row.update(payload[phase])
            rows.append(row)
    return pd.DataFrame(rows)


def flatten_roc_gap_for_csv(roc_gap_before, roc_gap_after):
    rows = []
    for model_name in roc_gap_before:
        rows.append(
            {
                "model": model_name,
                "roc_gap_before": float(roc_gap_before[model_name]),
                "roc_gap_after": float(roc_gap_after[model_name]),
            }
        )
    return pd.DataFrame(rows)