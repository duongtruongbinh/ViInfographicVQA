from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from peft import PeftModel

from ft_vlm.model.qwen2_vl import QLoRAConfig, build_model_and_processor
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x


def _resolve_image_path(image: Optional[str], images_base_dir: Optional[str]) -> Optional[str]:
    if not isinstance(image, str):
        return None
    p = Path(image)
    if p.is_absolute() or images_base_dir is None:
        return str(p)
    candidate = Path(images_base_dir) / p
    if candidate.exists():
        return str(candidate)
    basename_candidate = Path(images_base_dir) / p.name
    if basename_candidate.exists():
        return str(basename_candidate)
    return str(candidate)


def _load_image(image_path: Optional[str], max_size: int = 1280) -> Optional[Image.Image]:
    """Load and optionally resize image to reduce token count"""
    if image_path is None:
        return None
    try:
        img = Image.open(image_path).convert("RGB")
        width, height = img.size
        max_dim = max(width, height)
        
        if max_dim > max_size:
            # Resize proportionally
            scale = max_size / max_dim
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        return img
    except Exception as e:
        print(f"[ERROR] Failed to load image {image_path}: {e}")
        return None


def _build_user_only_messages(item: Dict[str, Any], images_base_dir: Optional[str], max_image_size: int = 1280) -> Tuple[List[Dict[str, Any]], List[Image.Image]]:
    # Prefer explicit messages if provided: take the last user turn only
    if isinstance(item.get("messages"), list):
        last_user = None
        for turn in item["messages"]:
            if isinstance(turn, dict) and turn.get("role") == "user":
                last_user = turn
        if last_user is not None and isinstance(last_user.get("content"), list):
            content: List[Dict[str, Any]] = []
            images: List[Image.Image] = []
            for c in last_user["content"]:
                if isinstance(c, dict) and c.get("type") == "image":
                    img_path = _resolve_image_path(c.get("image"), images_base_dir)
                    img = _load_image(img_path, max_size=max_image_size)
                    if img is not None:
                        images.append(img)
                        content.append({"type": "image", "image": img})
                    else:
                        # Skip images that could not be loaded to avoid placeholder mismatch
                        print(f"[WARN] Could not load image: {img_path}. Skipping this image.")
                elif isinstance(c, dict):
                    content.append(c)
            return ([{"role": "user", "content": content}], images)

    # Fallback: expect flat fields
    # Support both single image (image/image_path) and multiple images (image_paths)
    question = item.get("question") or item.get("query") or item.get("input") or ""
    templated_question = (
        f"Answer the following question based solely on the image content concisely with a single term: {question}. Answer:"
        if question
        else "Answer the following question based solely on the image content concisely with a single term: . Answer:"
    )
    
    content: List[Dict[str, Any]] = []
    images: List[Image.Image] = []
    
    # Handle multi-image format (image_paths as a list)
    if "image_paths" in item and isinstance(item["image_paths"], list):
        for img_path in item["image_paths"]:
            resolved = _resolve_image_path(img_path, images_base_dir)
            img = _load_image(resolved, max_size=max_image_size)
            if img is not None:
                content.append({"type": "image", "image": img})
                images.append(img)
    else:
        # Handle single image format
        image_path = item.get("image") or item.get("image_path") or item.get("img")
        resolved = _resolve_image_path(image_path, images_base_dir)
        img = _load_image(resolved, max_size=max_image_size)
        if img is not None:
            content.append({"type": "image", "image": img})
            images.append(img)
    
    # If no valid images were loaded, ensure we don't include stale image placeholders
    if len(images) == 0:
        content = [c for c in content if c.get("type") == "text"]
    
    content.append({"type": "text", "text": templated_question})
    return ([{"role": "user", "content": content}], images)


def _prepare_processor_inputs(processor, messages: List[Dict[str, Any]], images: List[Image.Image]):
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    # The number of <|image_pad|> placeholders equals len(images) if messages contain images
    # Pass None if images list is empty to avoid creating phantom image tokens
    inputs = processor(
        text=[text],
        images=images if len(images) > 0 else None,
        return_tensors="pt",
    )
    return text, inputs


def run_inference(
    base_model_id: str,
    adapter_dir: str,
    input_json: str,
    output_json: str,
    images_base_dir: Optional[str] = None,
    max_new_tokens: int = 128,
    temperature: float = 0.2,
    top_p: float = 0.9,
    do_sample: bool = True,  # Changed default to True for proper sampling
    use_quantization: bool = False,  # Set to False for inference to avoid corruption
    max_image_size: int = 1280,  # Resize images to reduce token count
) -> None:
    # Load base model WITHOUT quantization (merging adapters with quantized models causes issues)
    qlora_cfg = QLoRAConfig(use_qlora=use_quantization)
    device_map = "auto"
    
    print(f"Loading model: {base_model_id}")
    model, _, _ = build_model_and_processor(
        base_model_id, qlora_cfg, device_map=device_map
    )
    
    # Load processor from adapter directory (has correct tokenizer config from training)
    try:
        from transformers import Qwen2_5_VLProcessor as Qwen25VLProcessor
        processor = Qwen25VLProcessor.from_pretrained(adapter_dir)
    except:
        from transformers import Qwen2VLProcessor
        processor = Qwen2VLProcessor.from_pretrained(adapter_dir)
    
    # Load and merge adapter
    print(f"Loading adapter: {adapter_dir}")
    model = PeftModel.from_pretrained(model, adapter_dir)
    model = model.merge_and_unload()
    model.eval()
    
    # Ensure tokenizer has proper padding token
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
    
    print(f"Model ready on {model.device}")

    data_path = Path(input_json)
    raw = json.loads(data_path.read_text())
    items: List[Dict[str, Any]]
    if isinstance(raw, dict) and "data" in raw:
        items = raw["data"]
    elif isinstance(raw, list):
        items = raw
    else:
        raise ValueError("Input JSON must be a list or a dict with key 'data'.")

    results: List[Dict[str, Any]] = []
    iterator = tqdm(items, total=len(items), desc="Inference", dynamic_ncols=True)
    for idx, item in enumerate(iterator):
        try:
            messages, images = _build_user_only_messages(item, images_base_dir, max_image_size=max_image_size)
            text, inputs = _prepare_processor_inputs(processor, messages, images)
        except Exception as e:
            print(f"[ERROR] Failed to prepare inputs for item {idx}: {e}")
            result = {
                "id": item.get("id") or item.get("question_id"),
                "question_id": item.get("question_id", item.get("id")),
                "question": item.get("question") or item.get("query") or item.get("input") or "",
                "prediction": "[PREP_ERROR]",
                "label": item.get("output") or item.get("answer") or item.get("final_answer") or "",
                "output": "[PREP_ERROR]",
            }
            results.append(result)
            continue
        
        # Validate input_ids BEFORE moving to device
        if "input_ids" in inputs:
            input_ids = inputs["input_ids"]
            seq_len = input_ids.shape[1]
            max_id = input_ids.max().item()
            min_id = input_ids.min().item()
            vocab_size = model.config.vocab_size
            
            # Check for suspiciously long sequences (likely indicates error)
            if seq_len > 28000:
                print(f"[ERROR] Item {idx}: Sequence too long ({seq_len} tokens), skipping...")
                result = {
                    "id": item.get("id") or item.get("question_id"),
                    "question_id": item.get("question_id", item.get("id")),
                    "question": item.get("question") or item.get("query") or item.get("input") or "",
                    "prediction": "[TOO_LONG]",
                    "label": item.get("output") or item.get("answer") or item.get("final_answer") or "",
                    "output": "[TOO_LONG]",
                }
                results.append(result)
                continue
            
            if max_id >= vocab_size or min_id < 0:
                print(f"[ERROR] Item {idx}: Invalid token IDs [{min_id}, {max_id}], skipping...")
                result = {
                    "id": item.get("id") or item.get("question_id"),
                    "question_id": item.get("question_id", item.get("id")),
                    "question": item.get("question") or item.get("query") or item.get("input") or "",
                    "prediction": "[INVALID_TOKENS]",
                    "label": item.get("output") or item.get("answer") or item.get("final_answer") or "",
                    "output": "[INVALID_TOKENS]",
                }
                results.append(result)
                continue

        # Move tensors to model device
        try:
            inputs = {k: (v.to(model.device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
        except Exception as e:
            print(f"[ERROR] Failed to move tensors to device for item {idx}: {e}")
            result = {
                "id": item.get("id") or item.get("question_id"),
                "question_id": item.get("question_id", item.get("id")),
                "question": item.get("question") or item.get("query") or item.get("input") or "",
                "prediction": "[DEVICE_ERROR]",
                "label": item.get("output") or item.get("answer") or item.get("final_answer") or "",
                "output": "[DEVICE_ERROR]",
            }
            results.append(result)
            continue

        with torch.no_grad():
            # Build generation kwargs
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "pad_token_id": processor.tokenizer.pad_token_id,
                "eos_token_id": processor.tokenizer.eos_token_id,
            }
            
            # Only add temperature and top_p if sampling is enabled
            if do_sample:
                gen_kwargs["temperature"] = temperature
                gen_kwargs["top_p"] = top_p
            
            try:
                generated = model.generate(**inputs, **gen_kwargs)
            except Exception as e:
                print(f"ERROR during generation for item {idx}: {e}")
                print(f"  Input shape: {inputs['input_ids'].shape}")
                print(f"  Sample question: {item.get('question', 'N/A')[:100]}...")
                # Add a placeholder result and continue
                result = {
                    "id": item.get("id") or item.get("question_id"),
                    "question_id": item.get("question_id", item.get("id")),
                    "question": item.get("question") or item.get("query") or item.get("input") or "",
                    "prediction": "[ERROR]",
                    "label": item.get("output") or item.get("answer") or item.get("final_answer") or "",
                    "output": "[ERROR]",
                }
                results.append(result)
                continue

        # Remove the prompt part from the generated sequence
        input_len = inputs["input_ids"].shape[1]
        gen_ids = generated[0][input_len:]
        pred = processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        label = item.get("output") or item.get("answer") or item.get("final_answer") or ""
        
        # Support both id and question_id fields
        item_id = item.get("id") or item.get("question_id")
        result = {
            "id": item_id,
            "question_id": item.get("question_id", item_id),  # Include question_id for multi-image format
            "question": item.get("question") or item.get("query") or item.get("input") or "",
            "prediction": pred,
            "label": label,
            "output": pred,  # Keep for backward compatibility
        }
        results.append(result)

    out_path = Path(output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))


def cli_main() -> None:
    parser = argparse.ArgumentParser(description="Run inference with LoRA-adapted Qwen2.5-VL")
    parser.add_argument("--input_json", type=str, required=True, help="Path to input JSON file")
    parser.add_argument("--output_json", type=str, required=True, help="Path to write output JSON")
    parser.add_argument(
        "--adapter_dir",
        type=str,
        default="/home/vlai-vqa-info/members/namlnb/fine-tuning-vlm/ft-vlm/weight_save_train/Qwen2.5-VL-7B-Instruct",
        help="Directory containing saved LoRA adapter",
    )
    parser.add_argument(
        "--base_model_id",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="Base HF model ID",
    )
    parser.add_argument("--images_base_dir", type=str, default=None, help="Optional images base dir for relative paths")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--do_sample", action="store_true", default=False, 
                        help="Enable sampling. Default is greedy decoding (more stable)")
    parser.add_argument("--use_quantization", action="store_true", default=False,
                        help="Use 4-bit quantization (default: False). Not recommended with adapter merging")
    parser.add_argument("--max_image_size", type=int, default=1280,
                        help="Max image dimension (default: 1280). Larger images are resized to reduce token count")
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
        do_sample=bool(args.do_sample),
        use_quantization=bool(args.use_quantization),
        max_image_size=args.max_image_size,
    )


if __name__ == "__main__":
    cli_main()


