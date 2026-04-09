# Repository Results Analysis

Date: 2026-04-09

This report analyzes only the result artifacts that exist in this repository.

## Scope

Datasets present in this repository results folder:

- MMLU
- StereoSet
- BBQ
- WinoBias

Compared model variants:

- Llama-3.1-8B-Instruct (baseline)
- Llama-3.1-8B-Instruct-GSQ
- Meta-Llama-3.1-8B-Instruct-AWQ-INT4

## Source Files

- results/mmlu/Llama-3.1-8B-Instruct.json
- results/mmlu/Llama-3.1-8B-Instruct-GSQ.json
- results/mmlu/Meta-Llama-3.1-8B-Instruct-AWQ-INT4.json
- results/stereoset/Llama-3.1-8B-Instruct.json
- results/stereoset/Llama-3.1-8B-Instruct-GSQ.json
- results/stereoset/Meta-Llama-3.1-8B-Instruct-AWQ-INT4.json
- results/bbq/Llama-3.1-8B-Instruct.json
- results/bbq/Llama-3.1-8B-Instruct-GSQ.json
- results/bbq/Meta-Llama-3.1-8B-Instruct-AWQ-INT4.json
- results/winobias/Llama-3.1-8B-Instruct.json
- results/winobias/Llama-3.1-8B-Instruct-GSQ.json
- results/winobias/Meta-Llama-3.1-8B-Instruct-AWQ-INT4.json

## Raw Metrics

### MMLU (higher accuracy is better)

| Model | Accuracy |
|---|---:|
| Llama-3.1-8B-Instruct | 0.649765 |
| Llama-3.1-8B-Instruct-GSQ | 0.567939 |
| Meta-Llama-3.1-8B-Instruct-AWQ-INT4 | 0.633101 |

### StereoSet (closer to target is better)

Targets:

- stereotype_score target = 0.5
- cohen_d target = 0

| Model | stereotype_score | abs(stereotype_score - 0.5) | cohen_d | abs(cohen_d) |
|---|---:|---:|---:|---:|
| Llama-3.1-8B-Instruct | 0.665242 | 0.165242 | 0.175952 | 0.175952 |
| Llama-3.1-8B-Instruct-GSQ | 0.654321 | 0.154321 | 0.140830 | 0.140830 |
| Meta-Llama-3.1-8B-Instruct-AWQ-INT4 | 0.661918 | 0.161918 | 0.177431 | 0.177431 |

### BBQ (closer to zero bias is better)

| Model | ambiguous_bias | disambiguated_bias |
|---|---:|---:|
| Llama-3.1-8B-Instruct | 0.067414 | 0.026799 |
| Llama-3.1-8B-Instruct-GSQ | 0.068142 | 0.038439 |
| Meta-Llama-3.1-8B-Instruct-AWQ-INT4 | 0.037711 | 0.026791 |

### WinoBias (lower bias values are better)

| Model | historical_bias | population_bias | acc_pro | acc_anti | acc_male | acc_female |
|---|---:|---:|---:|---:|---:|---:|
| Llama-3.1-8B-Instruct | 24.6212 | 5.1475 | 89.2677 | 64.6465 | 79.5455 | 74.3980 |
| Llama-3.1-8B-Instruct-GSQ | 27.2727 | 2.8824 | 87.3737 | 60.1010 | 75.2525 | 72.3701 |
| Meta-Llama-3.1-8B-Instruct-AWQ-INT4 | 21.4646 | 0.7461 | 80.8081 | 59.3434 | 70.4545 | 69.7085 |

## Delta vs Baseline

### MMLU

- GSQ: -0.081826 absolute accuracy vs baseline.
- AWQ-INT4: -0.016664 absolute accuracy vs baseline.

Interpretation:

- AWQ-INT4 preserves capability much better than GSQ on MMLU.

### StereoSet

- GSQ improves closeness to 0.5 stereotype score by 0.010921 and improves abs(cohen_d) by 0.035122.
- AWQ-INT4 improves closeness to 0.5 stereotype score slightly by 0.003324, but worsens abs(cohen_d) by 0.001480.

Interpretation:

- GSQ is the strongest of the two quantized variants on StereoSet fairness indicators.
- AWQ-INT4 is mixed on StereoSet: small gain on stereotype score, slight regression on effect size.

### BBQ

- GSQ: ambiguous_bias worsens by +0.000728 and disambiguated_bias worsens by +0.011640.
- AWQ-INT4: ambiguous_bias improves by -0.029703 and disambiguated_bias improves marginally by -0.000008.

Interpretation:

- AWQ-INT4 is clearly better than both baseline and GSQ on BBQ ambiguous bias.
- GSQ regresses on both BBQ bias metrics.

### WinoBias

- GSQ: historical_bias worsens by +2.6515, population_bias improves by -2.2650.
- AWQ-INT4: historical_bias improves by -3.1566, population_bias improves by -4.4014.

Utility trend in WinoBias accuracy fields:

- GSQ reduces both pro and anti accuracies vs baseline.
- AWQ-INT4 further reduces both pro and anti accuracies vs baseline.

Interpretation:

- AWQ-INT4 gives the strongest bias reduction on WinoBias metrics, but with a larger accuracy drop.
- GSQ gives partial bias improvement (population_bias) while worsening historical_bias.

## Overall Ranking by Objective

If capability on MMLU is primary:

1. Baseline
2. AWQ-INT4
3. GSQ

If fairness on BBQ and WinoBias is primary:

1. AWQ-INT4
2. Baseline
3. GSQ

If StereoSet fairness indicators are primary:

1. GSQ
2. AWQ-INT4 (mixed)
3. Baseline

## Practical Takeaway

No single variant dominates every metric.

- AWQ-INT4 is the strongest fairness-oriented choice on BBQ and WinoBias while largely preserving MMLU compared with GSQ.
- GSQ is strongest on StereoSet indicators but shows larger capability loss and weaker BBQ/WinoBias consistency.
- Baseline remains best for pure capability but is not best on most fairness metrics.

For a balanced deployment recommendation from these results, AWQ-INT4 is the most consistent compromise across capability and fairness, with the caveat of lower WinoBias accuracy fields.

## StereoSet FROC Output Analysis (from outputs folder)

FROC artifacts currently present in this repository:

- outputs/stereoset/Llama-3.1-8B-Instruct_standard/roc_gap.csv
- outputs/stereoset/Llama-3.1-8B-Instruct-GSQ_gsq/roc_gap.csv
- outputs/stereoset/Meta-Llama-3.1-8B-Instruct-AWQ-INT4_awq/roc_gap.csv
- outputs/stereoset/Llama-3.1-8B-Instruct_standard/metrics_before_after.csv
- outputs/stereoset/Llama-3.1-8B-Instruct-GSQ_gsq/metrics_before_after.csv
- outputs/stereoset/Meta-Llama-3.1-8B-Instruct-AWQ-INT4_awq/metrics_before_after.csv
- outputs/stereoset/Llama-3.1-8B-Instruct_standard/thresholds.json
- outputs/stereoset/Llama-3.1-8B-Instruct-GSQ_gsq/thresholds.json
- outputs/stereoset/Meta-Llama-3.1-8B-Instruct-AWQ-INT4_awq/thresholds.json

### ROC-gap results

| Run | ROC gap (before) | ROC gap (after) | Delta |
|---|---:|---:|---:|
| standard | 0.0420598709 | 0.0000000000 | -0.0420598709 |
| gsq | 0.0400424215 | 0.0000000000 | -0.0400424215 |
| awq | 0.0417565746 | 0.0000000000 | -0.0417565746 |

All three runs report complete ROC-gap collapse to 0.0 after thresholding.

### Before/after metric behavior under FROC

| Run | Accuracy | F1 | AUC | DPD | EOD |
|---|---|---|---|---|---|
| standard | 0.6625 -> 0.5000 | 0.6625 -> 0.0000 | 0.7035718750 -> 0.7035718750 | 0.0000 -> 0.0000 | 0.3956043956 -> 0.0000 |
| gsq | 0.6425 -> 0.5000 | 0.6425 -> 0.0000 | 0.6874687500 -> 0.6874687500 | 0.0000 -> 0.0000 | 0.3000000000 -> 0.0000 |
| awq | 0.6550 -> 0.5000 | 0.6550 -> 0.0000 | 0.7024625000 -> 0.7024625000 | 0.0000 -> 0.0000 | 0.3956043956 -> 0.0000 |

Interpretation:

- The post-processing achieves perfect ROC-gap alignment and zero EOD in these runs.
- This comes with severe utility collapse (F1 drops to 0.0 and accuracy converges to 0.5 in all runs).
- AUC remains unchanged because ranking quality in scores is unchanged; only thresholded decisions changed.

### Run conditions and comparability

All three runs share identical sanity settings:

- `num_pairs = 400`
- `num_binary_samples = 800`
- `num_groups = 19`
- `epsilon = 0.05`
- `k = 100`

This makes the observed differences attributable to model variant (standard vs GSQ vs AWQ) rather than sampling or configuration drift.

### Final takeaway for FROC in this repository

For the available StereoSet FROC outputs, fairness alignment metrics improve to the extreme endpoint (ROC gap/EOD to 0), but operational utility degrades substantially. This indicates the current thresholding policy is over-aggressive for deployment and should be treated as a fairness-stress baseline rather than a production operating point.

## Strict vs Pragmatic Differentiation

This repository currently does not contain paired strict/pragmatic output bundles (for example, no `phase23_strict` and `phase23_pragmatic` artifact directories are present in tracked outputs).

Given the artifacts that do exist under `outputs/stereoset`, the observed FROC behavior is:

- complete ROC-gap alignment (`roc_gap_after = 0.0`),
- complete EOD collapse to `0.0`,
- major utility loss (`accuracy -> 0.5`, `f1 -> 0.0`).

That behavior matches a strict-style operating point (high fairness-alignment pressure, low utility retention) rather than a pragmatic-style operating point.

Pragmatic-style expectation, by contrast, would typically be:

- partial ROC-gap reduction (not necessarily to exactly `0.0`),
- meaningful utility retention,
- less extreme threshold effects.

So the practical differentiation from currently available artifacts is:

1. Strict-like profile in this repo outputs: strong fairness alignment, severe utility degradation.
2. Pragmatic profile: not empirically present yet in tracked output artifacts, so no direct numeric strict-vs-pragmatic delta table can be computed from current files.

## BBQ and WinoBias FROC Support

Two new FROC runners have been implemented for BBQ and WinoBias datasets with full strict/pragmatic mode support:

### BBQ FROC Pipeline

**Script:** `scripts/run_bbq_froc.py`

**Design:**
- Converts BBQ predictions to a scoring problem: y_true = 1 if pro-stereotype predicted, 0 if anti-stereotype
- Groups by category (gender, race, religion, etc.)
- Applies geometric ROC transport (Algorithm 1) for strict mode or direct threshold matching (pragmatic mode)

**Usage:**
```bash
PYTHONPATH=. python scripts/run_bbq_froc.py \
  --model_path gpt2 \
  --froc-mode both \
  --epsilon 0.05 \
  --output_dir outputs/bbq
```

**Output structure:**
- `outputs/bbq/phase23_strict/<model>/` and `outputs/bbq/phase23_pragmatic/<model>/`
- Each contains: metrics_before_after.csv, roc_gap.csv, thresholds.json, summary.json, sanity_report.json

### WinoBias FROC Pipeline

**Script:** `scripts/run_winobias_froc.py`

**Design:**
- Converts WinoBias predictions to binary scores: y_true = 1 if coreference resolution correct, 0 if incorrect
- Groups by gender (male, female)
- Applies the same geometric ROC transport framework for fairness correction
- Combines data from all four stereotype type conditions (pro/anti × type1/type2)

**Usage:**
```bash
PYTHONPATH=. python scripts/run_winobias_froc.py \
  --model_path gpt2 \
  --froc-mode both \
  --epsilon 0.05 \
  --output_dir outputs/winobias
```

**Output structure:**
- `outputs/winobias/phase23_strict/<model>/` and `outputs/winobias/phase23_pragmatic/<model>/`
- Each contains: metrics_before_after.csv, roc_gap.csv, thresholds.json, summary.json, sanity_report.json

### Implementation Details

Both runners follow the same pattern:

1. **Data Collection**: Load all test examples and compute log-probability scores for predictions
2. **FROC Application**:
   - **Strict mode**: Identifies disadvantaged group (lowest AUC), then transports privileged-group ROC curves to stay within epsilon L1-distance using the full Algorithm 1 (CutShift/UpShift/LeftShift decisions)
   - **Pragmatic mode**: Finds single operating point (global target TPR/FPR) and matches each group's threshold to that point
3. **Output Generation**: Generate metrics (accuracy, F1, AUC, DPD, EOD) before/after, ROC gaps before/after, and per-group thresholds

Both respect the `--froc-mode {strict, pragmatic, both}` flag and `--epsilon` parameter for transport budget control.
