## Decoder Phase 5: FROC Pipeline

Run the decoder fairness post-processing pipeline on any dataset that exposes `text`, `label`, and `group` columns or arrays.

```bash
PYTHONPATH=. python scripts/run_decoder_phase5.py \
	--data_path path/to/decoder_data.json \
	--model_name gpt2 \
	--output_dir outputs/decoder_phase5
```

The runner saves `metrics_before_after.csv`, `roc_gap.csv`, `thresholds.json`, `summary.json`, and plots in the output directory.

## StereoSet FROC Script

Use this script when you want direct FROC on StereoSet with geometric ROC transport (no pre-quantized checkpoint required).

```bash
PYTHONPATH=. python scripts/run_stereoset_froc.py \
	--model_name gpt2 \
	--dataset_path benchmark_datasets/stereo_set/stereo_set.json \
	--subset intrasentence \
	--group_field target \
	--epsilon 0.05 \
	--max_samples 400 \
	--output_dir outputs/decoder_phase5/stereoset
```

Key notes:
- The script builds binary pairs from stereotype vs anti-stereotype sentences.
- INT8 is created at runtime with dynamic quantization (`torch.quantization.quantize_dynamic`).
- Outputs include `metrics_before_after.csv`, `roc_gap.csv`, and `thresholds.json`.

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


bash run_all_models.sh

Runs the evaluation script for all models in the original_models directory