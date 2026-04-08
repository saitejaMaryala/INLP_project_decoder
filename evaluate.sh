#!/bin/bash
# evaluate.sh

MODEL_PATH=$1

if [ -z "$MODEL_PATH" ]; then
    echo "Usage: ./evaluate.sh <model_path>"
    echo "Example: ./evaluate.sh original_models/Llama-3.1-8B-Instruct"
    exit 1
fi

if [ -d "$MODEL_PATH/vllm_quant_model" ]; then
    MODEL_PATH="$MODEL_PATH/vllm_quant_model"
fi

echo "==========================================="
echo "Starting evaluation for: $MODEL_PATH"
echo "Available GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
echo "==========================================="

export CUDA_VISIBLE_DEVICES=1

echo -e "\n[1/4] Evaluating on MMLU..."
PYTHONPATH=. python3 scripts/run_mmlu.py \
    --model_path "$MODEL_PATH" \
    --batch_size 4          # lower to 16 if OOM, raise to 64 if VRAM allows

echo -e "\n[2/4] Evaluating on StereoSet..."
PYTHONPATH=. python3 scripts/run_stereoset.py --model_path "$MODEL_PATH"

echo -e "\n[3/4] Evaluating on BBQ..."
PYTHONPATH=. python3 scripts/run_bbq.py --model_path "$MODEL_PATH"

echo -e "\n[4/4] Evaluating on WinoBias..."
PYTHONPATH=. python3 scripts/run_winobias.py --model_path "$MODEL_PATH"

echo "==========================================="
echo "Evaluation completed for: $MODEL_PATH"
echo "Results saved in results/ directory."
echo "==========================================="
