#!/bin/bash

# Example usage script
# Replace paths with your actual model paths

# If using a full model (merged weights):
# python inference_eval.py --model_path /path/to/checkpoint-13560

# If using a LoRA adapter:
# python inference_eval.py \
#   --model_path /path/to/checkpoint-13560 \
#   --base_model_path /path/to/base_model \
#   --test_file /mnt/workspace/majian/vlm_data/input/jsons/test/Chinese.json

# Current default usage (assuming full model at a specific path for demo):
echo "Please edit this script to set your model path."
# python inference_eval.py --model_path /workspace/checkpoints/checkpoint-13560
