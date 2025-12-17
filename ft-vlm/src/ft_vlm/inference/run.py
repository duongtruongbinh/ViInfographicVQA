"""Inference runner for LoRA-adapted Qwen2.5-VL models."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Optional

import torch
from PIL import Image
from peft import PeftModel

from ft_vlm.model.qwen2_vl import QLoRAConfig, build_model_and_processor

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

logger = logging.getLogger(__name__)


def _resolve_image_path(image: Optional[str], images_base_dir: Optional[str]) -> Optional[str]:
    """
    Resolve image path strictly without basename fallback.

    Args:
        image: Image path (relative or absolute)
        images_base_dir: Base directory for relative paths

    Returns:
        Resolved image path, or None if image is None/invalid
    """
    if not isinstance(image, str):
        return None
    p = Path(image)
    if p.is_absolute():
        return str(p)
    if images_base_dir is None:
        return str(p)
    return str(Path(images_base_dir) / p)


def _load_image(image_path: Optional[str], max_size: int = 1280) -> Optional[Image.Image]:
    """
    Load and optionally resize image.

    Args:
        image_path: Path to image file
        max_size: Maximum dimension for resizing

    Returns:
        PIL Image or None if loading fails
    """
    if image_path is None:
        return None
    try:
        img = Image.open(image_path).convert("RGB")
        width, height = img.size
        max_dim = max(width, height)

        if max_dim > max_size:
            scale = max_size / max_dim
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        return img
    except Exception as e:
        logger.warning("Failed to load image %s: %s", image_path, e)
        return None


def _build_user_messages(
    item: dict[str, Any],
    images_base_dir: Optional[str],
    max_image_size: int = 1280,
) -> tuple[list[dict[str, Any]], list[Image.Image]]:
    """
    Build user messages from dataset item.

    Args:
        item: Dataset item dictionary
        images_base_dir: Base directory for image paths
        max_image_size: Maximum image dimension

    Returns:
        Tuple of (messages list, loaded images list)
    """
    if isinstance(item.get("messages"), list):
        last_user = None
        for turn in item["messages"]:
            if isinstance(turn, dict) and turn.get("role") == "user":
                last_user = turn
        if last_user is not None and isinstance(last_user.get("content"), list):
            content: list[dict[str, Any]] = []
            images: list[Image.Image] = []
            for c in last_user["content"]:
                if isinstance(c, dict) and c.get("type") == "image":
                    img_path = _resolve_image_path(c.get("image"), images_base_dir)
                    img = _load_image(img_path, max_size=max_image_size)
                    if img is not None:
                        images.append(img)
                        content.append({"type": "image", "image": img})
                elif isinstance(c, dict):
                    content.append(c)
            return ([{"role": "user", "content": content}], images)

    question = item.get("question") or item.get("query") or item.get("input") or ""
    templated_question = (
        f"Answer the following question based solely on the image content "
        f"concisely with a single term: {question}. Answer:"
    )

    content: list[dict[str, Any]] = []
    images: list[Image.Image] = []

    if "image_paths" in item and isinstance(item["image_paths"], list):
        for img_path in item["image_paths"]:
            resolved = _resolve_image_path(img_path, images_base_dir)
            img = _load_image(resolved, max_size=max_image_size)
            if img is not None:
                content.append({"type": "image", "image": img})
                images.append(img)
    else:
        image_path = item.get("image") or item.get("image_path") or item.get("img")
        resolved = _resolve_image_path(image_path, images_base_dir)
        img = _load_image(resolved, max_size=max_image_size)
        if img is not None:
            content.append({"type": "image", "image": img})
            images.append(img)

    if len(images) == 0:
        content = [c for c in content if c.get("type") == "text"]

    content.append({"type": "text", "text": templated_question})
    return ([{"role": "user", "content": content}], images)


def _prepare_inputs(
    processor,
    messages: list[dict[str, Any]],
    images: list[Image.Image],
) -> tuple[str, dict]:
    """Prepare processor inputs from messages and images."""
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(
        text=[text],
        images=images if images else None,
        return_tensors="pt",
    )
    return text, inputs


def _create_error_result(item: dict[str, Any], error_code: str) -> dict[str, Any]:
    """Create standardized error result."""
    item_id = item.get("id") or item.get("question_id")
    return {
        "id": item_id,
        "question_id": item.get("question_id", item_id),
        "question": item.get("question") or item.get("query") or item.get("input") or "",
        "prediction": error_code,
        "label": item.get("output") or item.get("answer") or item.get("final_answer") or "",
        "output": error_code,
    }


def run_inference(
    base_model_id: str,
    adapter_dir: str,
    input_json: str,
    output_json: str,
    images_base_dir: Optional[str] = None,
    max_new_tokens: int = 128,
    temperature: float = 0.2,
    top_p: float = 0.9,
    do_sample: bool = True,
    use_quantization: bool = False,
    max_image_size: int = 1280,
) -> None:
    """
    Run inference with LoRA-adapted model.

    Args:
        base_model_id: HuggingFace model ID
        adapter_dir: Directory containing LoRA adapter
        input_json: Path to input JSON file
        output_json: Path to output JSON file
        images_base_dir: Base directory for image paths
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        do_sample: Enable sampling (vs greedy decoding)
        use_quantization: Enable 4-bit quantization
        max_image_size: Maximum image dimension
    """
    qlora_cfg = QLoRAConfig(use_qlora=use_quantization)

    logger.info("Loading model: %s", base_model_id)
    model, _, _ = build_model_and_processor(
        base_model_id, qlora_cfg, device_map="auto"
    )

    try:
        from transformers import Qwen2_5_VLProcessor
        processor = Qwen2_5_VLProcessor.from_pretrained(adapter_dir)
    except Exception:
        from transformers import Qwen2VLProcessor
        processor = Qwen2VLProcessor.from_pretrained(adapter_dir)

    logger.info("Loading adapter: %s", adapter_dir)
    model = PeftModel.from_pretrained(model, adapter_dir)
    model = model.merge_and_unload()
    model.eval()

    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    data_path = Path(input_json)
    raw = json.loads(data_path.read_text())
    items: list[dict[str, Any]]
    if isinstance(raw, dict) and "data" in raw:
        items = raw["data"]
    elif isinstance(raw, list):
        items = raw
    else:
        raise ValueError("Input JSON must be a list or a dict with key 'data'.")

    results: list[dict[str, Any]] = []
    iterator = tqdm(items, total=len(items), desc="Inference", dynamic_ncols=True)

    for idx, item in enumerate(iterator):
        try:
            messages, images = _build_user_messages(item, images_base_dir, max_image_size)
            _, inputs = _prepare_inputs(processor, messages, images)
        except Exception:
            results.append(_create_error_result(item, "[PREP_ERROR]"))
            continue

        if "input_ids" in inputs:
            input_ids = inputs["input_ids"]
            seq_len = input_ids.shape[1]
            max_id = input_ids.max().item()
            min_id = input_ids.min().item()
            vocab_size = model.config.vocab_size

            if seq_len > 28000:
                results.append(_create_error_result(item, "[TOO_LONG]"))
                continue

            if max_id >= vocab_size or min_id < 0:
                results.append(_create_error_result(item, "[INVALID_TOKENS]"))
                continue

        try:
            inputs = {
                k: v.to(model.device) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }
        except Exception:
            results.append(_create_error_result(item, "[DEVICE_ERROR]"))
            continue

        with torch.no_grad():
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "pad_token_id": processor.tokenizer.pad_token_id,
                "eos_token_id": processor.tokenizer.eos_token_id,
            }
            if do_sample:
                gen_kwargs["temperature"] = temperature
                gen_kwargs["top_p"] = top_p

            try:
                generated = model.generate(**inputs, **gen_kwargs)
            except Exception:
                results.append(_create_error_result(item, "[GEN_ERROR]"))
                continue

        input_len = inputs["input_ids"].shape[1]
        gen_ids = generated[0][input_len:]
        pred = processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        label = item.get("output") or item.get("answer") or item.get("final_answer") or ""
        item_id = item.get("id") or item.get("question_id")
        result = {
            "id": item_id,
            "question_id": item.get("question_id", item_id),
            "question": item.get("question") or item.get("query") or item.get("input") or "",
            "prediction": pred,
            "label": label,
            "output": pred,
        }
        results.append(result)

    out_path = Path(output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))


def cli_main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run inference with LoRA-adapted Qwen2.5-VL"
    )
    parser.add_argument(
        "--input_json", type=str, required=True, help="Path to input JSON file"
    )
    parser.add_argument(
        "--output_json", type=str, required=True, help="Path to write output JSON"
    )
    parser.add_argument(
        "--adapter_dir",
        type=str,
        required=True,
        help="Directory containing saved LoRA adapter",
    )
    parser.add_argument(
        "--base_model_id",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="Base HuggingFace model ID",
    )
    parser.add_argument(
        "--images_base_dir",
        type=str,
        default=None,
        help="Base directory for relative image paths",
    )
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument(
        "--do_sample",
        action="store_true",
        default=False,
        help="Enable sampling (default: greedy decoding)",
    )
    parser.add_argument(
        "--use_quantization",
        action="store_true",
        default=False,
        help="Use 4-bit quantization (not recommended with adapter merging)",
    )
    parser.add_argument(
        "--max_image_size",
        type=int,
        default=1280,
        help="Max image dimension for resizing",
    )
    args = parser.parse_args()

    run_inference(
        base_model_id=args.base_model_id,
        adapter_dir=args.adapter_dir,
        input_json=args.input_json,
        output_json=args.output_json,
        images_base_dir=args.images_base_dir,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.do_sample,
        use_quantization=args.use_quantization,
        max_image_size=args.max_image_size,
    )


if __name__ == "__main__":
    cli_main()
