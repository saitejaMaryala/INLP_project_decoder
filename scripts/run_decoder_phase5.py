import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.decoder_froc import (
    compute_scores_batch,
    flatten_metrics_for_csv,
    flatten_roc_gap_for_csv,
    froc_pipeline,
    load_decoder_data,
    load_decoder_model,
    plot_fairness_comparison,
    plot_roc_curves,
    plot_roc_gap,
    quantize_model_int8,
    roc_analysis_pipeline,
    threshold_invariance_check,
)

from utils.load_model import load_model
from gsq_quant.gsq_load import load_gsq_model


def _score_model(tokenizer, model, texts):
    scores = compute_scores_batch(model, tokenizer, texts)
    return np.asarray(scores, dtype=float)


def _save_figure(fig, path):
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.clf()


def _build_results_frame(metrics_before_after):
    frame = flatten_metrics_for_csv(metrics_before_after)
    return frame.sort_values(["model", "phase"]).reset_index(drop=True)


def _write_verification_report(report_path, mode_name, metrics_frame, roc_gap_frame, transport_diagnostics, froc_eps):
    lines = []
    lines.append(f"# Phase 2/3 Verification Report ({mode_name})")
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    lines.append(f"- froc_mode: {mode_name}")
    lines.append(f"- froc_eps: {float(froc_eps)}")
    lines.append("")
    lines.append("## Model Summary")
    lines.append("")

    model_names = sorted(metrics_frame["model"].unique().tolist())
    for model_name in model_names:
        model_before = metrics_frame[(metrics_frame["model"] == model_name) & (metrics_frame["phase"] == "before")].iloc[0]
        model_after = metrics_frame[(metrics_frame["model"] == model_name) & (metrics_frame["phase"] == "after")].iloc[0]
        roc_row = roc_gap_frame[roc_gap_frame["model"] == model_name].iloc[0]

        lines.append(f"### {model_name}")
        lines.append("")
        lines.append(
            "- ROC gap: "
            f"{float(roc_row['roc_gap_before']):.8f} -> {float(roc_row['roc_gap_after']):.8f}"
        )
        lines.append(
            "- Accuracy: "
            f"{float(model_before['accuracy']):.6f} -> {float(model_after['accuracy']):.6f}"
        )
        lines.append(
            "- F1: "
            f"{float(model_before['f1']):.6f} -> {float(model_after['f1']):.6f}"
        )
        lines.append(
            "- DPD: "
            f"{float(model_before['dpd']):.6f} -> {float(model_after['dpd']):.6f}"
        )
        lines.append(
            "- EOD: "
            f"{float(model_before['eod']):.6f} -> {float(model_after['eod']):.6f}"
        )

        diagnostics = transport_diagnostics.get(model_name, {})
        if diagnostics:
            lines.append(
                "- Transport: "
                f"disadvantaged_group={diagnostics.get('disadvantaged_group')}, "
                f"max_l1_after={diagnostics.get('max_l1_after')}"
            )
        lines.append("")

    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def _run_single_mode(
    mode_name,
    output_dir,
    results,
    skip_plots,
    froc_eps,
    strict_num_points,
    invariance_deltas,
):
    os.makedirs(output_dir, exist_ok=True)

    metrics_before_after, thresholds_per_model, transport_diagnostics = froc_pipeline(
        results,
        froc_mode=mode_name,
        froc_eps=froc_eps,
        strict_num_points=strict_num_points,
    )
    roc_gap_before, roc_gap_after = roc_analysis_pipeline(results, thresholds_per_model)

    metrics_frame = _build_results_frame(metrics_before_after)
    roc_gap_frame = flatten_roc_gap_for_csv(roc_gap_before, roc_gap_after)

    metrics_path = os.path.join(output_dir, "metrics_before_after.csv")
    roc_gap_path = os.path.join(output_dir, "roc_gap.csv")
    thresholds_path = os.path.join(output_dir, "thresholds.json")
    transport_path = os.path.join(output_dir, "transport_diagnostics.json")
    threshold_invariance_path = os.path.join(output_dir, "threshold_invariance.csv")
    report_path = os.path.join(output_dir, "phase23_verification_report.md")

    metrics_frame.to_csv(metrics_path, index=False)
    roc_gap_frame.to_csv(roc_gap_path, index=False)

    with open(thresholds_path, "w", encoding="utf-8") as handle:
        json.dump(thresholds_per_model, handle, indent=2)

    with open(transport_path, "w", encoding="utf-8") as handle:
        json.dump(transport_diagnostics, handle, indent=2)

    invariance_frames = []
    for model_name, payload in results.items():
        invariance_frame = threshold_invariance_check(
            payload["y_true"],
            payload["y_score"],
            payload["group"],
            thresholds_per_model[model_name]["thresholds"],
            deltas=invariance_deltas,
        )
        invariance_frame.insert(0, "model", model_name)
        invariance_frames.append(invariance_frame)

    if invariance_frames:
        pd.concat(invariance_frames, ignore_index=True).to_csv(threshold_invariance_path, index=False)
    else:
        pd.DataFrame(columns=["model", "delta", "accuracy", "f1", "auc", "dpd", "eod", "roc_gap"]).to_csv(
            threshold_invariance_path,
            index=False,
        )

    _write_verification_report(
        report_path,
        mode_name,
        metrics_frame,
        roc_gap_frame,
        transport_diagnostics,
        froc_eps,
    )

    if not skip_plots:
        fairness_fig = plot_fairness_comparison(metrics_before_after)
        _save_figure(fairness_fig, os.path.join(output_dir, "fairness_comparison.png"))

        roc_gap_fig = plot_roc_gap(roc_gap_before, roc_gap_after)
        _save_figure(roc_gap_fig, os.path.join(output_dir, "roc_gap_comparison.png"))

        for model_name, payload in results.items():
            roc_fig = plot_roc_curves(
                payload["y_true"],
                payload["y_score"],
                payload["group"],
                title=f"ROC Curves: {model_name}",
            )
            _save_figure(roc_fig, os.path.join(output_dir, f"roc_curves_{model_name}.png"))

    summary = {
        "froc_mode": mode_name,
        "froc_eps": float(froc_eps),
        "metrics_before_after": metrics_frame.to_dict(orient="records"),
        "roc_gap": roc_gap_frame.to_dict(orient="records"),
        "thresholds": thresholds_per_model,
        "transport_diagnostics": transport_diagnostics,
    }

    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Run decoder FROC Phase 5 pipeline")
    parser.add_argument("--data_path", default="/home/sai.teja/gsq_decoder/benchmark_datasets/stereo_set/stereo_set.json", help="CSV, JSON, or JSONL with text/label/group columns")
    parser.add_argument("--model_name", default="gpt2", help="Hugging Face causal LM name or path")
    parser.add_argument("--output_dir", default="./output_decoder_phase5")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--skip_plots", action="store_true")
    parser.add_argument(
        "--froc-mode",
        default="both",
        choices=["strict", "pragmatic", "both"],
        help="FROC mode to run",
    )
    parser.add_argument("--froc-eps", type=float, default=0.02, help="L1 transport budget used in strict mode")
    parser.add_argument("--strict-num-points", type=int, default=200, help="Interpolation grid size used in strict mode")
    parser.add_argument(
        "--invariance-deltas",
        default="-0.02,-0.01,0,0.01,0.02",
        help="Comma-separated threshold perturbations for invariance checks",
    )
    parser.add_argument(
        "--model_type",
        default="standard",
        choices=["standard", "awq", "gsq"],
        help="Type of model to load"
    )   
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if "stereo_set" in args.data_path.lower():
        with open(args.data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        flat_records = []
        for subset in ["intrasentence", "intersentence"]:
            if subset in data.get("data", {}):
                for sample in data["data"][subset]:
                    stereo = None
                    anti = None
                    for sentence in sample.get("sentences", []):
                        if sentence.get("gold_label") == "stereotype":
                            stereo = sentence.get("sentence")
                        elif sentence.get("gold_label") == "anti-stereotype":
                            anti = sentence.get("sentence")
                    if stereo and anti:
                        group_value = str(sample.get("target", "unknown"))
                        flat_records.append({"text": stereo, "label": 1, "group": group_value})
                        flat_records.append({"text": anti, "label": 0, "group": group_value})
        frame = pd.DataFrame(flat_records)
    else:
        frame = load_decoder_data(args.data_path)
        
    if args.max_samples is not None:
        frame = frame.head(args.max_samples).reset_index(drop=True)

    texts = frame["text"].tolist()
    labels = frame["label"].to_numpy(dtype=int)
    groups = frame["group"].astype(str).to_numpy()

    results = {}

    if args.model_type == "gsq":
        model, tokenizer, metadata = load_gsq_model(args.model_name, device_map="auto")

    elif args.model_type == "awq":
        model, tokenizer = load_model(args.model_name, load_type="AWQ")

    else:  # standard
        model, tokenizer = load_model(args.model_name, load_type="standard")

    model_type_name = args.model_type

    results[model_type_name] = {
        "y_true": labels,
        "y_score": _score_model(tokenizer, model, texts),
        "group": groups,
    }

    invariance_deltas = []
    for token in str(args.invariance_deltas).split(","):
        token = token.strip()
        if not token:
            continue
        invariance_deltas.append(float(token))
    if not invariance_deltas:
        invariance_deltas = [-0.02, -0.01, 0.0, 0.01, 0.02]

    model_basename = os.path.basename(os.path.normpath(args.model_name))
    if args.froc_mode == "both":
        mode_to_dir = {
            "strict": os.path.join(args.output_dir, "phase23_strict", model_basename),
            "pragmatic": os.path.join(args.output_dir, "phase23_pragmatic", model_basename),
        }
    else:
        mode_to_dir = {args.froc_mode: os.path.join(args.output_dir, model_basename)}

    for mode_name, mode_output_dir in mode_to_dir.items():
        _run_single_mode(
            mode_name=mode_name,
            output_dir=mode_output_dir,
            results=results,
            skip_plots=args.skip_plots,
            froc_eps=args.froc_eps,
            strict_num_points=args.strict_num_points,
            invariance_deltas=invariance_deltas,
        )

    print(f"Saved decoder Phase 5 outputs to {args.output_dir}")


if __name__ == "__main__":
    main()