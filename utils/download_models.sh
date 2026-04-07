#!/bin/bash

# Stop on any error
set -e

# Hugging Face Model IDs
BASE_MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"
AWQ_MODEL_ID="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"

# Local save paths
BASE_LOCAL_PATH="hf_models/Llama-3.1-8B-Instruct"
AWQ_LOCAL_PATH="hf_models/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"

# echo "==========================================="
# echo "1. Installing hf CLI if missing..."
# echo "==========================================="
# pip install -q -U "huggingface_hub[cli]"

echo "==========================================="
echo "2. Downloading Models from Hugging Face..."
echo "==========================================="

# Download Base Model (skip original FP16 weights)
echo "Downloading Base Model: $BASE_MODEL_ID"
hf download $BASE_MODEL_ID --repo-type model --local-dir $BASE_LOCAL_PATH --exclude "original/*"

# Download AWQ Model (skip original FP16 weights if present)

echo "Downloading AWQ Model: $AWQ_MODEL_ID"
hf download $AWQ_MODEL_ID \
   --repo-type model \
   --local-dir $AWQ_LOCAL_PATH \
   --exclude "original/*"

echo "All models downloaded successfully!"