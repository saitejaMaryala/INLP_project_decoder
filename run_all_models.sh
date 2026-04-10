#!/bin/bash

echo "Starting evaluation for all models..."

# echo "[1/3] running evaluation on standard model"
# python scripts/run_decoder_phase5.py --model_name hf_models/Llama-3.1-8B-Instruct --model_type standard

# echo "[2/3] running evaluation on awq model"
# python scripts/run_decoder_phase5.py --model_name hf_models/Meta-Llama-3.1-8B-Instruct-AWQ-INT4/ --model_type awq

# echo "[3/3] running evaluation on gsq model"
# python scripts/run_decoder_phase5.py --model_name hf_models/Llama-3.1-8B-Instruct-GSQ/ --model_type gsq

CUDA_VISIBLE_DEVICES=1

# BBQ paired strict + pragmatic:
# echo "[1/6] running evaluation BBQ on standard model"
# python scripts/run_bbq_froc.py --model_path hf_models/Llama-3.1-8B-Instruct --model_type standard --dataset_path benchmark_datasets/bbq --epsilon 0.05 --froc-mode both --output_dir outputs/bbq

# WinoBias paired strict + pragmatic:
# echo "[2/6] running evaluation WinoBias on standard model"
# python scripts/run_winobias_froc.py --model_path hf_models/Llama-3.1-8B-Instruct --model_type standard --dataset_dir benchmark_datasets/wino_bias --epsilon 0.05 --froc-mode both --output_dir outputs/winobias

# If you want GSQ or AWQ variants too, run the same commands with model-specific args:

# BBQ GSQ:

echo "[3/6] running evaluation BBQ on gsq model"
python scripts/run_bbq_froc.py --model_path hf_models/Llama-3.1-8B-Instruct-GSQ --model_type gsq --dataset_path benchmark_datasets/bbq --epsilon 0.05 --froc-mode both --output_dir outputs/bbq

# WinoBias GSQ:

# echo "[4/6] running evaluation WinoBias on gsq model"
# python scripts/run_winobias_froc.py --model_path hf_models/Llama-3.1-8B-Instruct-GSQ --model_type gsq --dataset_dir benchmark_datasets/wino_bias --epsilon 0.05 --froc-mode both --output_dir outputs/winobias

# BBQ AWQ:
echo "[5/6] running evaluation BBQ on awq model"
python scripts/run_bbq_froc.py --model_path hf_models/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 --model_type awq --dataset_path benchmark_datasets/bbq --epsilon 0.05 --froc-mode both --output_dir outputs/bbq

# WinoBias AWQ:
# echo "[6/6] running evaluation WinoBias on awq model"
# python scripts/run_winobias_froc.py --model_path hf_models/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 --model_type awq --dataset_dir benchmark_datasets/wino_bias --epsilon 0.05 --froc-mode both --output_dir outputs/winobias


echo "Expected output folders:"
echo "outputs/bbq with phase23_strict and phase23_pragmatic subfolders"
echo "outputs/winobias with phase23_strict and phase23_pragmatic subfolders"

echo "All evaluations finished."
