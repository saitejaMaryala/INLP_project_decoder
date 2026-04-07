import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
import torch

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
)


def _score_model(tokenizer, model, texts):
    scores = compute_scores_batch(model, tokenizer, texts)
    return np.asarray(scores, dtype=float)


def _save_figure(fig, path):
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.clf()


def _build_results_frame(metrics_before_after):
    frame = flatten_metrics_for_csv(metrics_before_after)
    return frame.sort_values(["model", "phase"]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Run decoder FROC Phase 5 pipeline")
    parser.add_argument("--data_path", required=True, help="CSV, JSON, or JSONL with text/label/group columns")
    parser.add_argument("--model_name", default="gpt2", help="Hugging Face causal LM name or path")
    parser.add_argument("--output_dir", default="outputs/decoder_phase5")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--skip_plots", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    frame = load_decoder_data(args.data_path)
    if args.max_samples is not None:
        frame = frame.head(args.max_samples).reset_index(drop=True)

    texts = frame["text"].tolist()
    labels = frame["label"].to_numpy(dtype=int)
    groups = frame["group"].astype(str).to_numpy()

    results = {}

    tokenizer_fp32, model_fp32 = load_decoder_model(args.model_name, device=args.device)
    results["fp32"] = {
        "y_true": labels,
        "y_score": _score_model(tokenizer_fp32, model_fp32, texts),
        "group": groups,
    }

    if torch.cuda.is_available() and args.device == "cuda":
        try:
            tokenizer_fp16, model_fp16 = load_decoder_model(args.model_name, device=args.device, torch_dtype=torch.float16)
            results["fp16"] = {
                "y_true": labels,
                "y_score": _score_model(tokenizer_fp16, model_fp16, texts),
                "group": groups,
            }
        except Exception as exc:
            warnings.warn(f"FP16 loading failed, skipping the FP16 variant: {exc}")
    else:
        print("FP16 requested but CUDA is unavailable; skipping the FP16 variant.")

    tokenizer_int8, model_int8 = load_decoder_model(args.model_name, device="cpu")
    model_int8 = quantize_model_int8(model_int8)
    results["int8"] = {
        "y_true": labels,
        "y_score": _score_model(tokenizer_int8, model_int8, texts),
        "group": groups,
    }

    metrics_before_after, thresholds_per_model = froc_pipeline(results)
    roc_gap_before, roc_gap_after = roc_analysis_pipeline(results, thresholds_per_model)

    metrics_frame = _build_results_frame(metrics_before_after)
    roc_gap_frame = flatten_roc_gap_for_csv(roc_gap_before, roc_gap_after)

    metrics_path = os.path.join(args.output_dir, "metrics_before_after.csv")
    roc_gap_path = os.path.join(args.output_dir, "roc_gap.csv")
    thresholds_path = os.path.join(args.output_dir, "thresholds.json")

    metrics_frame.to_csv(metrics_path, index=False)
    roc_gap_frame.to_csv(roc_gap_path, index=False)

    with open(thresholds_path, "w", encoding="utf-8") as handle:
        json.dump(thresholds_per_model, handle, indent=2)

    if not args.skip_plots:
        fairness_fig = plot_fairness_comparison(metrics_before_after)
        _save_figure(fairness_fig, os.path.join(args.output_dir, "fairness_comparison.png"))

        roc_gap_fig = plot_roc_gap(roc_gap_before, roc_gap_after)
        _save_figure(roc_gap_fig, os.path.join(args.output_dir, "roc_gap_comparison.png"))

        for model_name, payload in results.items():
            roc_fig = plot_roc_curves(
                payload["y_true"],
                payload["y_score"],
                payload["group"],
                title=f"ROC Curves: {model_name}",
            )
            _save_figure(roc_fig, os.path.join(args.output_dir, f"roc_curves_{model_name}.png"))

    summary = {
        "metrics_before_after": metrics_frame.to_dict(orient="records"),
        "roc_gap": roc_gap_frame.to_dict(orient="records"),
        "thresholds": thresholds_per_model,
    }

    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Saved decoder Phase 5 outputs to {args.output_dir}")


if __name__ == "__main__":
    main()