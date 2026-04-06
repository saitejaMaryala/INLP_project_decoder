#!/bin/bash
# run_all_models.sh

source /ssd_scratch/saiteja/miniconda3/bin/activate
conda activate gsq

for m in original_models/*; do
    if [ -d "$m" ]; then
        # Check if it has vllm_quant_model nested directory
        if [ -d "$m/vllm_quant_model" ]; then
            MODEL_PATH="$m/vllm_quant_model"
        else
            MODEL_PATH="$m"
        fi
        
        # Only evaluate if config.json exists
        if [ -f "$MODEL_PATH/config.json" ]; then
            echo "Scheduling evaluation for $MODEL_PATH"
            # It's better to run in background or directly.
            # Assuming you run it serially (this will take a while)
            bash evaluate.sh "$MODEL_PATH"
        fi
    fi
done

echo "All evaluations finished."
