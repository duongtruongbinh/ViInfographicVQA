# ViInfographicVQA: A Benchmark for Single and Multi-image Visual Question Answering on Vietnamese Infographics

[![arXiv](https://img.shields.io/badge/arXiv-2512.12424-b31b1b.svg)](https://arxiv.org/abs/2512.12424)
[![Hugging Face Datasets](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-ViInfographicVQA-yellow)](https://huggingface.co/datasets/duytranus/ViInfographicVQA)

This repository contains the official implementation and evaluation code for the paper **"ViInfographicVQA: A Benchmark for Single and Multi-image Visual Question Answering on Vietnamese Infographics"**.

## 📰 News
* **[2025/12]** 🔥 Our paper has been accepted to **AI4Research Workshop @ AAAI 2026**!
* **[2025/12]** We released the [ViInfographicVQA dataset](https://huggingface.co/datasets/duytranus/ViInfographicVQA) and the evaluation code.

## 🌟 Overview
**ViInfographicVQA** is the first large-scale benchmark for Vietnamese Infographic VQA, comprising over **6,747 infographics** and **20,409 QA pairs**. It evaluates models on two distinct tasks:
1.  **Single-image QA:** Traditional VQA requiring layout understanding and OCR.
2.  **Multi-image QA:** A novel task requiring cross-image reasoning, evidence synthesis, and arithmetic operations across multiple semantically related infographics.

![Overview](assets/samples.png)

## 📂 Project Structure

```text
ViInfographicVQA/
├── data/                 # Dataset placeholders
├── ft-vlm/               # Fine-tuning module configuration
├── results/              # Evaluation output
├── src/                  # Main source code
│   ├── common/           # Shared utilities and metrics (ANLS, Accuracy)
│   └── inference/        # Inference engines
│       ├── single/       # Single-image benchmarks
│       └── multi/        # Multi-image benchmarks
├── config.py             # Centralized path configuration
├── pyproject.toml        # Project dependencies & build config
└── uv.lock               # Dependency lock file (ensures reproducibility)
```

## 🛠️ Installation

We use [uv](https://github.com/astral-sh/uv) for extremely fast dependency management. Install uv from [here](https://docs.astral.sh/uv/getting-started/installation/).

1. **Clone the repository:**
```bash
git clone https://github.com/duongtruongbinh/ViInfographicVQA.git
cd ViInfographicVQA
```


2. **Sync Environment:**
This command creates a virtual environment and installs all dependencies (including `torch`, `transformers`, `flash-attn` compatible packages) defined in `uv.lock`.
```bash
uv sync
```

*Optional: If you want to use Flash Attention 2 (Recommended for speed):*
```bash
uv pip install flash-attn --no-build-isolation
```


4. **Setup Environment Variables:**
Create a `.env` file in the root directory:
```bash
export VQA_DATA_DIR="./data"
export VQA_IMAGES_DIR="/path/to/your/images"
export VQA_OUTPUT_DIR="./results"
# Optional: Point to local model weights if not using HuggingFace cache
export VQA_MODEL_QWENVL="Qwen/Qwen2.5-VL-7B-Instruct"
export VQA_MODEL_INTERNVL="OpenGVLab/InternVL3_5-8B"
```


## 📊 Dataset Preparation

Download the dataset from [Hugging Face](https://huggingface.co/datasets/duytranus/ViInfographicVQA) and organize it as follows:

```text
data/
├── train.json
├── test.json
├── images/           # Directory containing all infographic images
│   ├── image_01.jpg
│   └── ...
```

## 🚀 Inference & Evaluation

You can run scripts using `uv run` (which automatically uses the virtual environment) or activate the environment first.

### Single-image Evaluation

```bash
uv run python -m src.inference.single.run_inference \
    qwenvl \
    --data_path data/test.json \
    --image_folder /path/to/images \
    --output_dir results/single
```

### Multi-image Evaluation

```bash
uv run python -m src.inference.multi.run_inference \
    qwenvl \
    --data_path data/multi_image_test.json \
    --output_dir results/multi
```

### Calculate Scores (ANLS)

After inference, calculate the Average Normalized Levenshtein Similarity (ANLS) and Accuracy:

```bash
# For single-image results
uv run python -m src.inference.single.calculate_scores --results-dir results/single

# For multi-image results
uv run python -m src.inference.multi.calculate_scores --results-dir results/multi
```

## 🚅 Fine-tuning (Qwen2.5-VL)

The fine-tuning module is integrated into the project. We use `ft-vlm` (built on TRL and PEFT) for efficient instruction tuning.

```bash
# Example: Fine-tune on Multi-image task with LoRA
uv run ft-vlm-train \
    --config ft-vlm/configs/train_qwen25vl_7b_multi_image.json
```

Or using the python module directly:

```bash
uv run python -m src.ft_vlm.fine_tuning.train \
    --config ft-vlm/configs/train_qwen25vl_7b_multi_image.json
```

## 📝 Citation

If you find this code or dataset useful for your research, please cite our paper:

```bibtex
@misc{vandinh2025viinfographicvqa,
  title={ViInfographicVQA: A Benchmark for Single and Multi-image Visual Question Answering on Vietnamese Infographics}, 
  author={Tue-Thu Van-Dinh and Hoang-Duy Tran and Truong-Binh Duong and Mai-Hanh Pham and Binh-Nam Le-Nguyen and Quoc-Thai Nguyen},
  year={2025},
  eprint={2512.12424},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={[https://arxiv.org/abs/2512.12424](https://arxiv.org/abs/2512.12424)}, 
}
```