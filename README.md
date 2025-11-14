# ViInfographicVQA

Vietnamese Infographic Visual Question Answering - A comprehensive system for answering questions about Vietnamese infographics using vision-language models.

## Overview

This repository contains a complete pipeline for training and evaluating vision-language models on Vietnamese infographic datasets. The project supports both single-image and multi-image question answering tasks, with fine-tuning capabilities for Qwen2.5-VL models.

## Project Structure

```
ViInfographicVQA/
├── ft-vlm/                    # Fine-tuning module for Qwen2.5-VL models
│   ├── src/ft_vlm/           # Core fine-tuning code
│   │   ├── dataset/          # Dataset loaders (single + multi-image)
│   │   ├── fine_tuning/      # Training scripts with TRL
│   │   ├── inference/        # Inference scripts
│   │   └── model/            # Model wrappers
│   ├── configs/              # Training configuration files
│   ├── weight_save_train/    # Saved model checkpoints
│   ├── tests/                # Test files
│   ├── run_inference_test.sh # Inference convenience script
│   └── README.md             # Detailed fine-tuning documentation
├── src/                      # Main inference and evaluation code
│   └── inference/            # Inference modules
│       ├── single/           # Single-image VQA inference
│       └── multi/            # Multi-image VQA inference
├── data/                     # Dataset files
├── results/                  # Evaluation results
├── requirements.txt          # Main project dependencies
└── README.md                 # This file
```

## Quick Start

### Installation

1. **Main Project Dependencies:**
```bash
pip install -r requirements.txt
```

2. **Fine-tuning Module (ft-vlm):**
```bash
cd ft-vlm
pip install -e .
```

### Fine-tuning Models

For detailed fine-tuning instructions, see [ft-vlm/README.md](ft-vlm/README.md).

**Quick Training Example:**
```bash
cd ft-vlm
CUDA_VISIBLE_DEVICES=0 python -m src.ft_vlm.fine_tuning.train \
    --config configs/train_qwen25vl_7b_multi_image_merge.json
```

**Quick Inference Example:**
```bash
cd ft-vlm
bash run_inference_test.sh
```

### Data Formats

The system supports two data formats:

**Single-Image Format:**
```json
{
  "id": 27333,
  "input": "Theo biểu đồ thông tin, những lợi ích chính của việc Anh gia nhập CPTPP là gì?",
  "image": "/path/to/image.jpg",
  "output": "nâng Tổng sản phẩm quốc nội (GDP)..."
}
```

**Multi-Image Format:**
```json
{
  "question_id": "272",
  "image_paths": [
    "/path/to/image1.jpg",
    "/path/to/image2.jpg",
    "/path/to/image3.jpg"
  ],
  "question": "Số lượng trường học được thống kê trong năm học 2021-2022 và năm học 2020-2021 lần lượt là bao nhiêu?",
  "answer": "645; 2.543"
}
```

The system automatically detects the format based on the presence of `image_paths` field.

## Documentation

- **Fine-tuning Guide**: See [ft-vlm/README.md](ft-vlm/README.md) for comprehensive documentation on:
  - Training configuration
  - Data format details
  - Multi-GPU training
  - Inference parameters
  - Troubleshooting
  - Technical details

## Key Components

### Fine-tuning Module (ft-vlm)

The `ft-vlm` module provides:
- **Training**: Supervised fine-tuning using TRL (Transformer Reinforcement Learning)
- **LoRA/QLoRA**: Parameter-efficient fine-tuning support
- **Multi-image Training**: Handle multiple images per sample
- **Inference**: Batch inference with automatic image resizing

### Main Inference Module (src/inference)

The main inference module provides:
- **Single-image VQA**: Answer questions about individual infographics
- **Multi-image VQA**: Answer questions requiring multiple images
- **Model Abstraction**: Support for different VQA model architectures

## Configuration

### Training Configuration

Training configurations are stored in `ft-vlm/configs/`. Key parameters include:

- Model selection (Qwen2.5-VL-3B or 7B)
- LoRA parameters (rank, alpha, dropout)
- Training hyperparameters (learning rate, batch size, epochs)
- Image processing (max image size, resizing)
- Dataset paths

See [ft-vlm/README.md](ft-vlm/README.md) for detailed configuration options.

### Inference Configuration

Inference can be configured via command-line arguments:
- `--max_image_size`: Control image resizing (default: 1280px)
- `--max_new_tokens`: Maximum tokens to generate
- `--temperature`: Sampling temperature
- `--top_p`: Nucleus sampling parameter

## Requirements

### Main Project
- Python >= 3.10
- PyTorch >= 2.0.0
- Transformers >= 4.40.0
- See `requirements.txt` for full list

### Fine-tuning Module
- Additional dependencies for TRL, PEFT, QLoRA
- See `ft-vlm/pyproject.toml` for full list

## Usage Examples

### Training a Model

```bash
# Single GPU training
cd ft-vlm
CUDA_VISIBLE_DEVICES=2 python -m src.ft_vlm.fine_tuning.train \
    --config configs/train_qwen25vl_7b_multi_image_merge.json

# Multi-GPU training
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --standalone --nnodes=1 --nproc_per_node=3 \
    -m ft_vlm.fine_tuning.train \
    --config configs/train_qwen25vl_7b_local.json
```

### Running Inference

```bash
cd ft-vlm
python -m src.ft_vlm.inference.run \
    --input_json /path/to/test.json \
    --output_json ./results.json \
    --adapter_dir ./weight_save_train/Qwen2.5-VL-7B-Instruct/checkpoint-91825 \
    --base_model_id Qwen/Qwen2.5-VL-7B-Instruct \
    --images_base_dir /path/to/images/ \
    --max_new_tokens 128 \
    --temperature 0.2 \
    --top_p 0.9
```


