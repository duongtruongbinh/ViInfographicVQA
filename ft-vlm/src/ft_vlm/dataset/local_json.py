from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import Dataset


def _normalize_item(
    item: Dict[str, Any], images_base_dir: Optional[str]
) -> Dict[str, Any]:
    # If already in messages format, passthrough but normalize image paths
    if "messages" in item and isinstance(item["messages"], list):
        messages = []
        for turn in item["messages"]:
            if not isinstance(turn, dict):
                continue
            content = []
            for c in turn.get("content", []):
                if isinstance(c, dict) and c.get("type") == "image":
                    img = c.get("image")
                    if (
                        isinstance(img, str)
                        and images_base_dir
                        and not Path(img).is_absolute()
                    ):
                        img = str(Path(images_base_dir) / img)
                    content.append({"type": "image", "image": img})
                else:
                    content.append(c)
            messages.append({"role": turn.get("role", "user"), "content": content})
        return {"messages": messages}

    # Otherwise, expect flat fields: image/image_path/image_paths, question|input, answer|output
    # Handle both single image and multi-image cases
    images = []
    
    # Try to get image_paths (list) first for multi-image support
    image_paths = item.get("image_paths")
    if image_paths and isinstance(image_paths, list):
        # Multi-image case
        for img_path in image_paths:
            if isinstance(img_path, str):
                p = Path(img_path)
                if not p.is_absolute() and images_base_dir:
                    # First try joining the relative path under images_base_dir
                    candidate = Path(images_base_dir) / p
                    if candidate.exists():
                        images.append(str(candidate))
                    else:
                        # Fallback: try basename under images_base_dir
                        basename_candidate = Path(images_base_dir) / p.name
                        if basename_candidate.exists():
                            images.append(str(basename_candidate))
                        else:
                            images.append(str(candidate))
                else:
                    images.append(img_path)
    else:
        # Single image case
        image = item.get("image") or item.get("image_path") or item.get("img")
        if image and isinstance(image, str):
            if images_base_dir:
                p = Path(image)
                if not p.is_absolute():
                    # First try joining the relative path under images_base_dir
                    candidate = Path(images_base_dir) / p
                    if candidate.exists():
                        image = str(candidate)
                    else:
                        # Fallback: try basename under images_base_dir
                        basename_candidate = Path(images_base_dir) / p.name
                        if basename_candidate.exists():
                            image = str(basename_candidate)
                        else:
                            image = str(candidate)
            images.append(image)

    # Support multiple possible keys for text/labels
    question = item.get("question") or item.get("query") or item.get("input") or ""
    # Apply instruction template to the question for training samples
    templated_question = (
        f"Answer the following question based solely on the image content concisely with a single term: {question}. Answer:"
        if question
        else "Answer the following question based solely on the image content concisely with a single term: . Answer:"
    )
    answer = item.get("answer") or item.get("final_answer") or item.get("output") or ""

    # Build content with all images followed by the question text
    image_content = [{"type": "image", "image": img} for img in images]
    
    messages: List[Dict[str, Any]] = [
        {
            "role": "user",
            "content": image_content + [{"type": "text", "text": templated_question}],
        },
        {"role": "assistant", "content": [{"type": "text", "text": answer}]},
    ]
    return {"messages": messages}


def load_local_json_dataset(
    json_path: str,
    images_base_dir: Optional[str] = None,
    sample_size: Optional[int] = None,
) -> Dataset:
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    raw = json.loads(path.read_text())
    if isinstance(raw, dict) and "data" in raw:
        items = raw["data"]
    elif isinstance(raw, list):
        items = raw
    else:
        raise ValueError("JSON must be a list of samples or a dict with 'data' list")

    normalized = [_normalize_item(item, images_base_dir) for item in items]
    if sample_size is not None and sample_size > 0:
        normalized = normalized[:sample_size]

    return Dataset.from_list(normalized)
