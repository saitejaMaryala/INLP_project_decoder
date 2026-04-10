#!/bin/bash

echo "Starting evaluation for all models..."

echo "[1/3] running evaluation on standard model"
python scripts/run_decoder_phase5.py --model_name hf_models/Llama-3.1-8B-Instruct --model_type standard

echo "[2/3] running evaluation on awq model"
python scripts/run_decoder_phase5.py --model_name hf_models/Meta-Llama-3.1-8B-Instruct-AWQ-INT4/ --model_type awq

echo "[3/3] running evaluation on gsq model"
python scripts/run_decoder_phase5.py --model_name hf_models/Llama-3.1-8B-Instruct-GSQ/ --model_type gsq

echo "All evaluations finished."
