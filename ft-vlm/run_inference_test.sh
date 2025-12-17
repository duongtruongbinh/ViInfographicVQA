#!/bin/bash
# Inference script for fine-tuned VLM models
# Configure paths below before running

set -e

# ==============================================================================
# CONFIGURATION - Set these paths before running
# ==============================================================================
BASE_MODEL_ID="${BASE_MODEL_ID:-Qwen/Qwen2.5-VL-7B-Instruct}"
ADAPTER_DIR="${ADAPTER_DIR:-./outputs/Qwen2.5-VL-7B-Instruct}"
INPUT_JSON="${INPUT_JSON:-./data/test.json}"
OUTPUT_JSON="${OUTPUT_JSON:-./results/inference_results.json}"
IMAGES_BASE_DIR="${IMAGES_BASE_DIR:-}"

# Inference parameters
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
TEMPERATURE="${TEMPERATURE:-0.2}"
TOP_P="${TOP_P:-0.9}"
GPU_ID="${GPU_ID:-0}"

# ==============================================================================
# SCRIPT START
# ==============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "================================================================================"
echo "Running Inference"
echo "================================================================================"
echo ""
echo "Configuration:"
echo "  Base Model:      $BASE_MODEL_ID"
echo "  Adapter Dir:     $ADAPTER_DIR"
echo "  Input JSON:      $INPUT_JSON"
echo "  Output JSON:     $OUTPUT_JSON"
echo "  Images Base Dir: ${IMAGES_BASE_DIR:-<not set>}"
echo "  GPU ID:          $GPU_ID"
echo ""

# Validate required files
if [ ! -f "$INPUT_JSON" ]; then
    echo "ERROR: Input JSON not found: $INPUT_JSON"
    echo "Set INPUT_JSON environment variable or update this script."
    exit 1
fi

if [ ! -d "$ADAPTER_DIR" ]; then
    echo "ERROR: Adapter directory not found: $ADAPTER_DIR"
    echo "Set ADAPTER_DIR environment variable or update this script."
    exit 1
fi

# Build command
CMD="CUDA_VISIBLE_DEVICES=$GPU_ID python -m ft_vlm.inference.run"
CMD="$CMD --input_json \"$INPUT_JSON\""
CMD="$CMD --output_json \"$OUTPUT_JSON\""
CMD="$CMD --adapter_dir \"$ADAPTER_DIR\""
CMD="$CMD --base_model_id \"$BASE_MODEL_ID\""
CMD="$CMD --max_new_tokens $MAX_NEW_TOKENS"
CMD="$CMD --temperature $TEMPERATURE"
CMD="$CMD --top_p $TOP_P"

if [ -n "$IMAGES_BASE_DIR" ]; then
    CMD="$CMD --images_base_dir \"$IMAGES_BASE_DIR\""
fi

echo "Running: $CMD"
echo ""
eval "$CMD"

echo ""
echo "================================================================================"
echo "Inference Complete!"
echo "================================================================================"
echo "Results saved to: $OUTPUT_JSON"
