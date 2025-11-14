#!/bin/bash

# Script to run inference on test data
# This will load the trained model and run predictions on the test dataset

set -e  # Exit on error

echo "================================================================================"
echo "Running Inference on Test Data"
echo "================================================================================"

# # Activate conda environment
# echo "Activating conda environment: namlnb_ft-vlm"
# source /home/vlai-vqa-info/anaconda3/bin/activate namlnb_ft-vlm

# Navigate to project directory
cd /home/vlai-vqa-info/members/namlnb/fine-tuning-vlm/ft-vlm

# Configuration
BASE_MODEL_ID="Qwen/Qwen2.5-VL-7B-Instruct"
ADAPTER_DIR="/home/vlai-vqa-info/members/namlnb/fine-tuning-vlm/ft-vlm/weight_save_train/Qwen2.5-VL-7B-Instruct/checkpoint-91825"
INPUT_JSON="/home/vlai-vqa-info/members/namlnb/data/multi_image_test_data.json"
OUTPUT_JSON="/home/vlai-vqa-info/members/namlnb/fine-tuning-vlm/ft-vlm/weight_save_train/inference_results_test_merge_inf_only_multi.json"
IMAGES_BASE_DIR="/mnt/VLAI_data/ViInfographicVQA/"

echo ""
echo "Configuration:"
echo "  Base Model: $BASE_MODEL_ID"
echo "  Adapter: $ADAPTER_DIR"
echo "  Input Data: $INPUT_JSON"
echo "  Output File: $OUTPUT_JSON"
echo "  Images Base Dir: $IMAGES_BASE_DIR"
echo ""

# Check if files exist
if [ ! -f "$INPUT_JSON" ]; then
    echo "ERROR: Input JSON file not found: $INPUT_JSON"
    exit 1
fi

if [ ! -d "$ADAPTER_DIR" ]; then
    echo "ERROR: Adapter directory not found: $ADAPTER_DIR"
    exit 1
fi

# Run inference
echo "Starting inference..."
CUDA_VISIBLE_DEVICES=0 python -m src.ft_vlm.inference.run \
    --input_json "$INPUT_JSON" \
    --output_json "$OUTPUT_JSON" \
    --adapter_dir "$ADAPTER_DIR" \
    --base_model_id "$BASE_MODEL_ID" \
    --images_base_dir "$IMAGES_BASE_DIR" \
    --max_new_tokens 128 \
    --temperature 0.2 \
    --top_p 0.9

echo ""
echo "================================================================================"
echo "Inference Complete!"
echo "================================================================================"
echo "Results saved to: $OUTPUT_JSON"
echo ""
echo "To view the results:"
echo "  head -n 50 $OUTPUT_JSON"
echo ""

