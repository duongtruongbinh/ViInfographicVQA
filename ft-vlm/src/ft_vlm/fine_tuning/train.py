from __future__ import annotations

import json
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Optional, List, Tuple

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import TrainingArguments
from trl import SFTTrainer

from ft_vlm.dataset.chartqa import load_chartqa_dataset
from ft_vlm.dataset.local_json import load_local_json_dataset
from ft_vlm.dataset.dump_datasets import (
    load_dump_dataset,
    merge_mixed_datasets_with_ratio,
)
from ft_vlm.model.qwen2_vl import QLoRAConfig, build_model_and_processor
import torch
from ft_vlm.fine_tuning.collator import VLMDataCollator


@dataclass
class TrainConfig:
    model_id: str = "Qwen/Qwen2-VL-7B-Instruct"
    output_dir: str = "./outputs/qwen2-vl-7b-trl-sft-chartqa"
    # Disable external reporting by default to avoid requiring credentials in DDP
    report_to: str | list[str] | None = "none"
    # DDP / DataLoader stability on tiny datasets
    dataloader_drop_last: bool = False
    dataloader_num_workers: int = 0
    ddp_find_unused_parameters: bool = False
    ddp_backend: Optional[str] = "nccl"
    max_train_samples: Optional[int] = None

    # Memory-saving overrides
    max_length: Optional[int] = None  # truncate sequence length for processor
    max_image_long_side: Optional[int] = None  # downscale images to cap vision memory
    torch_dtype: Optional[str] = None  # override loading dtype (e.g., "bfloat16")

    # SFT arguments
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    logging_steps: int = 10
    save_steps: int = 200
    save_total_limit: int = 2
    bf16: bool = True
    gradient_checkpointing: bool = True

    # Dataset
    dataset_name: str = (
        "chartqa"  # "chartqa" | "local_json" | "single_dump" | "multi_dump" | "mixed"
    )
    dataset_split: str = "train"
    dataset_config: Optional[str] = None
    local_json_path: Optional[str] = None
    local_json_test_path: Optional[str] = None  # Optional test/eval data for local_json
    images_base_dir: Optional[str] = None

    # Dump dataset directories for single/multi image datasets
    single_dump_dir: Optional[str] = None
    multi_dump_dir: Optional[str] = None

    # Mixed mode controls
    mixed_single_ratio: float = 0.5  # share of single dataset in merged set
    single_sample_size: Optional[int] = None
    multi_sample_size: Optional[int] = None

    # QLoRA
    use_qlora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Optional[List[str]] = None


def _output_dir_for_model(model_id: str, base_dir: Optional[str] = None) -> str:
    """
    Return output directory for model weights.

    The model name is derived from the last path segment of model_id.
    Base directory can be configured via the base_dir parameter or 
    VQA_OUTPUT_DIR environment variable.
    
    Args:
        model_id: Model identifier (e.g., "Qwen/Qwen2.5-VL-7B-Instruct")
        base_dir: Optional base directory override
        
    Returns:
        Path to output directory
    """
    if base_dir is None:
        base_dir = os.environ.get("VQA_OUTPUT_DIR", "./outputs")
    output_base = Path(base_dir)
    model_name = model_id.split("/")[-1].replace(" ", "_")
    return str(output_base / model_name)


def _build_sft_config(cfg: TrainConfig) -> TrainingArguments:
    # Check if we're in a distributed environment
    is_distributed = (
        "RANK" in os.environ 
        or "WORLD_SIZE" in os.environ 
        or "LOCAL_RANK" in os.environ
    )
    
    # Base arguments that work for both single and distributed training
    args_dict = {
        "output_dir": cfg.output_dir,
        "num_train_epochs": cfg.num_train_epochs,
        "per_device_train_batch_size": cfg.per_device_train_batch_size,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "learning_rate": cfg.learning_rate,
        "lr_scheduler_type": cfg.lr_scheduler_type,
        "warmup_ratio": cfg.warmup_ratio,
        "logging_steps": cfg.logging_steps,
        "save_steps": cfg.save_steps,
        "save_total_limit": cfg.save_total_limit,
        "bf16": cfg.bf16,
        "gradient_checkpointing": cfg.gradient_checkpointing,
        "report_to": cfg.report_to,
        "dataloader_drop_last": cfg.dataloader_drop_last,
        "dataloader_num_workers": cfg.dataloader_num_workers,
        "remove_unused_columns": False,
        "disable_tqdm": False,
        "logging_first_step": True,
        "log_level": "info",
    }
    
    # Only add DDP-specific arguments if in distributed mode
    if is_distributed:
        args_dict["ddp_find_unused_parameters"] = cfg.ddp_find_unused_parameters
        if cfg.ddp_backend:
            args_dict["ddp_backend"] = cfg.ddp_backend
    
    return TrainingArguments(**args_dict)


def _build_lora_config(cfg: TrainConfig) -> LoraConfig:
    # Sensible defaults for Qwen2 / Qwen2.5 VL text modules
    default_targets = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    return LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules or default_targets,
        task_type="CAUSAL_LM",
    )


def _build_datasets(cfg: TrainConfig):
    # Returns (train_dataset, eval_dataset|None)
    if cfg.dataset_name.lower() == "chartqa":
        train_dataset = load_chartqa_dataset(
            split=cfg.dataset_split,
            name=cfg.dataset_config,
            sample_size=cfg.max_train_samples,
        )
        return train_dataset, None
    elif cfg.dataset_name.lower() == "local_json":
        if not cfg.local_json_path:
            raise ValueError(
                "For dataset_name=local_json, set local_json_path in config"
            )
        train_dataset = load_local_json_dataset(
            json_path=cfg.local_json_path,
            images_base_dir=cfg.images_base_dir,
            sample_size=cfg.max_train_samples,
        )
        # Load test/eval dataset if provided
        eval_dataset = None
        if cfg.local_json_test_path:
            try:
                eval_dataset = load_local_json_dataset(
                    json_path=cfg.local_json_test_path,
                    images_base_dir=cfg.images_base_dir,
                    sample_size=None,  # Use all test samples
                )
            except Exception as e:
                print(f"Warning: Could not load test data from {cfg.local_json_test_path}: {e}")
                eval_dataset = None
        return train_dataset, eval_dataset
    elif cfg.dataset_name.lower() in ("single_dump", "multi_dump"):
        target_dir = (
            cfg.single_dump_dir
            if cfg.dataset_name.lower() == "single_dump"
            else cfg.multi_dump_dir
        )
        if not target_dir:
            raise ValueError(
                f"For dataset_name={cfg.dataset_name}, set the corresponding *_dump_dir in config"
            )
        # Default images_base_dir to <dataset_dir>/images if not provided
        from pathlib import Path as _P

        images_base_dir = cfg.images_base_dir or str(_P(target_dir) / "images")
        train_dataset = load_dump_dataset(
            dataset_dir=target_dir,
            split=cfg.dataset_split,
            images_base_dir=images_base_dir,
            sample_size=cfg.max_train_samples,
        )
        return train_dataset, None
    elif cfg.dataset_name.lower() == "mixed":
        if not cfg.single_dump_dir or not cfg.multi_dump_dir:
            raise ValueError(
                "For dataset_name=mixed, set single_dump_dir and multi_dump_dir"
            )
        # In mixed mode we independently load train/test for each dataset name
        from datasets import DatasetDict

        from pathlib import Path as _P

        single_images_base_dir = str(_P(cfg.single_dump_dir) / "images")
        multi_images_base_dir = str(_P(cfg.multi_dump_dir) / "images")

        single_train = load_dump_dataset(
            dataset_dir=cfg.single_dump_dir,
            split="train",
            images_base_dir=single_images_base_dir,
            sample_size=cfg.single_sample_size,
        )
        multi_train = load_dump_dataset(
            dataset_dir=cfg.multi_dump_dir,
            split="train",
            images_base_dir=multi_images_base_dir,
            sample_size=cfg.multi_sample_size,
        )

        single_eval = None
        multi_eval = None
        try:
            single_eval = load_dump_dataset(
                dataset_dir=cfg.single_dump_dir,
                split="test",
                images_base_dir=single_images_base_dir,
                sample_size=None,
            )
        except Exception:
            pass
        try:
            multi_eval = load_dump_dataset(
                dataset_dir=cfg.multi_dump_dir,
                split="test",
                images_base_dir=multi_images_base_dir,
                sample_size=None,
            )
        except Exception:
            pass

        train_merged = merge_mixed_datasets_with_ratio(
            single_train, multi_train, ratio_a=cfg.mixed_single_ratio
        )
        if single_eval is not None and multi_eval is not None:
            eval_merged = merge_mixed_datasets_with_ratio(
                single_eval, multi_eval, ratio_a=cfg.mixed_single_ratio
            )
        else:
            eval_merged = None
        return train_merged, eval_merged
    else:
        raise ValueError(f"Unsupported dataset: {cfg.dataset_name}")


def run_training(cfg: TrainConfig) -> None:
    qlora_cfg = QLoRAConfig(use_qlora=cfg.use_qlora)
    # Place the model on the correct device per process to support 4-bit training.
    # For bitsandbytes quantized models, loading on the destined device is required.
    if torch.cuda.is_available():
        try:
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        except Exception:
            local_rank = 0
        torch.cuda.set_device(local_rank)
        device_map_hint = {"": local_rank}
    else:
        device_map_hint = {"": "cpu"}

    model, processor, _ = build_model_and_processor(
        cfg.model_id, qlora_cfg, device_map=device_map_hint
    )

    # Use configured output_dir, or derive from model name if using default
    if cfg.output_dir == "./outputs/qwen2-vl-7b-trl-sft-chartqa":
        # Default value - derive from model name
        cfg.output_dir = _output_dir_for_model(cfg.model_id)

    if cfg.use_qlora:
        # Ensure base model is properly configured for k-bit training
        model = prepare_model_for_kbit_training(model)
        # Respect explicit gradient checkpointing preference
        if cfg.gradient_checkpointing:
            try:
                model.gradient_checkpointing_enable()
            except Exception:
                pass
            try:
                model.enable_input_require_grads()
            except Exception:
                pass
        lora_config = _build_lora_config(cfg)
        model = get_peft_model(model, lora_config)
        # Print trainable parameters to verify LoRA is applied
        model.print_trainable_parameters()

    train_dataset, eval_dataset = _build_datasets(cfg)

    training_args = _build_sft_config(cfg)
    data_collator = VLMDataCollator(
        processor,
        max_length=cfg.max_length,
        images_base_dir=cfg.images_base_dir,
        max_image_long_side=cfg.max_image_long_side,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        data_collator=data_collator,
    )

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    trainer.train()
    trainer.save_model(cfg.output_dir)


def load_train_config(config_path: Optional[str]) -> TrainConfig:
    if config_path is None:
        return TrainConfig()
    with open(config_path, "r") as f:
        data = json.load(f)
    return TrainConfig(**data)


def cli_main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen2-VL on ChartQA with TRL SFT"
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to JSON config file"
    )
    args = parser.parse_args()

    cfg = load_train_config(args.config)
    run_training(cfg)


if __name__ == "__main__":
    cli_main()
