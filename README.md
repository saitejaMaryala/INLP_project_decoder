## Decoder Phase 5: FROC Pipeline

Run the decoder fairness post-processing pipeline on any dataset that exposes `text`, `label`, and `group` columns or arrays.

```bash
PYTHONPATH=. python scripts/run_decoder_phase5.py \
	--data_path path/to/decoder_data.json \
	--model_name gpt2 \
	--froc-mode both \
	--froc-eps 0.02 \
	--output_dir outputs/decoder_phase5
```

When `--froc-mode both` is used, mode-specific directories are generated:

- `outputs/decoder_phase5/phase23_strict/`
- `outputs/decoder_phase5/phase23_pragmatic/`

Each mode directory contains:

- `phase23_verification_report.md`
- `metrics_before_after.csv`
- `roc_gap.csv`
- `thresholds.json`
- `transport_diagnostics.json`
- `threshold_invariance.csv`
- `summary.json`
- plots (`fairness_comparison.png`, `roc_gap_comparison.png`, and per-model ROC figures)

### Strict vs pragmatic mode

- `--froc-mode pragmatic`: deterministic threshold matching to a global operating point.
- `--froc-mode strict`: disadvantaged-group-aware transport-style threshold targeting with an L1 budget controlled by `--froc-eps`.
- `--froc-mode both`: runs strict and pragmatic in a single command and writes separate artifact bundles.

### Reproducible wrapper (archive + rerun)

Use the PowerShell wrapper to archive previous outputs and regenerate strict/pragmatic bundles in one command:

```powershell
.\run_froc_phase23_pipeline.ps1 -DataPath path/to/decoder_data.json -ModelName gpt2 -FrocEps 0.02
```

It validates the presence of key artifacts (`phase23_verification_report.md`, `transport_diagnostics.json`, and `threshold_invariance.csv`) for both modes.

### Static consistency verification (no execution)

The pipeline wiring has been statically verified in code without executing scripts in this environment.

Verified alignment:
- `README.md` commands and artifact expectations
- `scripts/run_decoder_phase5.py` CLI flags and output writers
- `utils/decoder_froc.py` strict/pragmatic mode logic and threshold invariance helper
- `run_froc_phase23_pipeline.ps1` wrapper behavior and required artifact checks

Teammate run-time validation checklist:
1. Run `run_froc_phase23_pipeline.ps1` with dataset path and model name.
2. Confirm both folders exist: `outputs/decoder_phase5/phase23_strict/` and `outputs/decoder_phase5/phase23_pragmatic/`.
3. Confirm each folder contains:
	- `phase23_verification_report.md`
	- `transport_diagnostics.json`
	- `threshold_invariance.csv`
	- `metrics_before_after.csv`
	- `roc_gap.csv`
	- `thresholds.json`
	- `summary.json`

### Validated FROC results currently in repository

Current FROC output artifacts are available for StereoSet runs under:

- `outputs/stereoset/Llama-3.1-8B-Instruct_standard/`
- `outputs/stereoset/Llama-3.1-8B-Instruct-GSQ_gsq/`
- `outputs/stereoset/Meta-Llama-3.1-8B-Instruct-AWQ-INT4_awq/`

Validated ROC-gap results:

| Run | ROC gap (before) | ROC gap (after) |
|---|---:|---:|
| standard | 0.0420598709 | 0.0000000000 |
| gsq | 0.0400424215 | 0.0000000000 |
| awq | 0.0417565746 | 0.0000000000 |

Validated metric shifts from `metrics_before_after.csv`:

| Run | Accuracy (before -> after) | F1 (before -> after) | AUC (before -> after) | DPD (before -> after) | EOD (before -> after) |
|---|---|---|---|---|---|
| standard | 0.6625 -> 0.5000 | 0.6625 -> 0.0000 | 0.7035718750 -> 0.7035718750 | 0.0000 -> 0.0000 | 0.3956043956 -> 0.0000 |
| gsq | 0.6425 -> 0.5000 | 0.6425 -> 0.0000 | 0.6874687500 -> 0.6874687500 | 0.0000 -> 0.0000 | 0.3000000000 -> 0.0000 |
| awq | 0.6550 -> 0.5000 | 0.6550 -> 0.0000 | 0.7024625000 -> 0.7024625000 | 0.0000 -> 0.0000 | 0.3956043956 -> 0.0000 |

Run setup (from `sanity_report.json`):

- `num_pairs = 400`
- `num_binary_samples = 800`
- `num_groups = 19`
- `epsilon = 0.05`
- `k = 100`

## StereoSet FROC Script

Use this script when you want direct FROC on StereoSet with geometric ROC transport (no pre-quantized checkpoint required).

**Single mode (pragmatic):**
```bash
PYTHONPATH=. python scripts/run_stereoset_froc.py \
	--model_name gpt2 \
	--dataset_path benchmark_datasets/stereo_set/stereo_set.json \
	--subset intrasentence \
	--group_field target \
	--epsilon 0.05 \
	--max_samples 400 \
	--froc-mode pragmatic \
	--output_dir outputs/stereoset
```

**Paired strict/pragmatic modes** (generates both in a single run):
```bash
PYTHONPATH=. python scripts/run_stereoset_froc.py \
	--model_name gpt2 \
	--dataset_path benchmark_datasets/stereo_set/stereo_set.json \
	--subset intrasentence \
	--group_field target \
	--epsilon 0.05 \
	--max_samples 400 \
	--froc-mode both \
	--output_dir outputs/stereoset
```

With `--froc-mode both`, outputs are organized as:
- `outputs/stereoset/phase23_strict/<model>_<type>/` — strict mode artifacts
- `outputs/stereoset/phase23_pragmatic/<model>_<type>/` — pragmatic mode artifacts

Each folder contains:
- `metrics_before_after.csv` — before/after fairness metrics
- `roc_gap.csv` — ROC gap before/after
- `thresholds.json` — per-group thresholds
- `summary.json` — run metadata
- `sanity_report.json` — dataset and epsilon info
- `roc_curves_*_before.png` and `roc_curves_*_after.png` — ROC curve plots

Key notes:
- The script builds binary pairs from stereotype vs anti-stereotype sentences.
- INT8 is created at runtime with dynamic quantization (`torch.quantization.quantize_dynamic`).
- `--froc-mode strict`: uses transport-aware threshold targeting with L1-budget control.
- `--froc-mode pragmatic`: uses direct operating-point matching (deterministic).
- `--froc-mode both`: runs both modes sequentially to phase23_strict/ and phase23_pragmatic/ folders.

### Quick smoke test (20 pairs)

Run this first to verify environment, model loading, and output generation:

```bash
python scripts/run_stereoset_froc.py \
	--model_name gpt2 \
	--smoke_test \
	--smoke_samples 20 \
	--output_dir outputs/decoder_phase5/stereoset_smoke
```

### Full run after training is done

1. Ensure your teammate has the final checkpoint path (local or Hugging Face model id).
2. From project root, install dependencies:

```bash
pip install -r requirements.txt
```

3. Run FROC on StereoSet with the trained checkpoint:

```bash
python scripts/run_stereoset_froc.py \
	--model_name path/to/trained/checkpoint \
	--dataset_path benchmark_datasets/stereo_set/stereo_set.json \
	--subset intrasentence \
	--group_field target \
	--epsilon 0.05 \
	--k 100 \
	--max_samples 400 \
	--output_dir outputs/decoder_phase5/stereoset_final
```

4. Optional: CPU-only run (if no GPU available):

```bash
python scripts/run_stereoset_froc.py \
	--model_name path/to/trained/checkpoint \
	--device cpu \
	--output_dir outputs/decoder_phase5/stereoset_cpu
```

5. Share these files with the team:
- outputs/decoder_phase5/stereoset_final/metrics_before_after.csv
- outputs/decoder_phase5/stereoset_final/roc_gap.csv
- outputs/decoder_phase5/stereoset_final/thresholds.json
- outputs/decoder_phase5/stereoset_final/sanity_report.json
- outputs/decoder_phase5/stereoset_final/summary.json

./evaluate.sh original_models/Llama-3.1-8B-Instruct

Runs the four dataset evaluation scripts sequentially for a provided model_path

## BBQ FROC Script

Use this script to apply strict/pragmatic FROC fairness correction to BBQ (Bias Benchmark for Question Answering) results.

**Single mode (pragmatic):**
```bash
PYTHONPATH=. python scripts/run_bbq_froc.py \
	--model_path gpt2 \
	--model_type standard \
	--dataset_path benchmark_datasets/bbq \
	--epsilon 0.05 \
	--froc-mode pragmatic \
	--output_dir outputs/bbq
```

**Paired strict/pragmatic modes:**
```bash
PYTHONPATH=. python scripts/run_bbq_froc.py \
	--model_path gpt2 \
	--model_type standard \
	--dataset_path benchmark_datasets/bbq \
	--epsilon 0.05 \
	--froc-mode both \
	--output_dir outputs/bbq
```

With `--froc-mode both`, outputs are organized as:
- `outputs/bbq/phase23_strict/<model>_<type>/` — strict mode artifacts
- `outputs/bbq/phase23_pragmatic/<model>_<type>/` — pragmatic mode artifacts

Each folder contains:
- `metrics_before_after.csv` — before/after fairness metrics
- `roc_gap.csv` — ROC gap before/after
- `thresholds.json` — per-category thresholds
- `summary.json` — run metadata
- `sanity_report.json` — dataset and epsilon info

Key notes:
- BBQ labels: 1 = pro-stereotype prediction, 0 = anti-stereotype prediction
- Groups: category (gender, race, religion)
- `--froc-mode strict`: uses transport-aware threshold targeting with L1-budget control.
- `--froc-mode pragmatic`: uses direct operating-point matching (deterministic).
- `--froc-mode both`: runs both modes sequentially to phase23_strict/ and phase23_pragmatic/ folders.

## WinoBias FROC Script

Use this script to apply strict/pragmatic FROC fairness correction to WinoBias (gender bias in coreference resolution) results.

**Single mode (pragmatic):**
```bash
PYTHONPATH=. python scripts/run_winobias_froc.py \
	--model_path gpt2 \
	--model_type standard \
	--dataset_dir benchmark_datasets/wino_bias \
	--epsilon 0.05 \
	--froc-mode pragmatic \
	--output_dir outputs/winobias
```

**Paired strict/pragmatic modes:**
```bash
PYTHONPATH=. python scripts/run_winobias_froc.py \
	--model_path gpt2 \
	--model_type standard \
	--dataset_dir benchmark_datasets/wino_bias \
	--epsilon 0.05 \
	--froc-mode both \
	--output_dir outputs/winobias
```

With `--froc-mode both`, outputs are organized as:
- `outputs/winobias/phase23_strict/<model>_<type>/` — strict mode artifacts
- `outputs/winobias/phase23_pragmatic/<model>_<type>/` — pragmatic mode artifacts

Each folder contains:
- `metrics_before_after.csv` — before/after fairness metrics
- `roc_gap.csv` — ROC gap before/after
- `thresholds.json` — per-gender thresholds
- `summary.json` — run metadata
- `sanity_report.json` — dataset and epsilon info

Key notes:
- WinoBias labels: 1 = correct coreference resolution, 0 = incorrect
- Groups: gender (male, female)
- Uses pro-stereotyped and anti-stereotyped sentence pairs to measure bias correction
- `--froc-mode strict`: uses transport-aware threshold targeting with L1-budget control.
- `--froc-mode pragmatic`: uses direct operating-point matching (deterministic).
- `--froc-mode both`: runs both modes sequentially to phase23_strict/ and phase23_pragmatic/ folders.

bash run_all_models.sh

Runs the evaluation script for all models in the original_models directory