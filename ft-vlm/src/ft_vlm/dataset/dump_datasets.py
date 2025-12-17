from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import json
from datasets import Dataset, concatenate_datasets


def _normalize_dump_item(
    item: Dict[str, Any], images_base_dir: Optional[str]
) -> Dict[str, Any]:
    # Accept both single-image and multi-image style records
    # Image(s)
    images: List[Any] = []
    if "images" in item and isinstance(item["images"], list):
        for im in item["images"]:
            images.append(_normalize_image_path(im, images_base_dir))
    else:
        image = item.get("image") or item.get("image_path") or item.get("img")
        if image is not None:
            images.append(_normalize_image_path(image, images_base_dir))

    # Text fields
    question = item.get("question") or item.get("query") or item.get("input") or ""
    answer = item.get("answer") or item.get("final_answer") or item.get("output") or ""

    content: List[Dict[str, Any]] = []
    for im in images:
        content.append({"type": "image", "image": im})
    content.append({"type": "text", "text": question})

    messages = [
        {"role": "user", "content": content},
        {"role": "assistant", "content": [{"type": "text", "text": answer}]},
    ]
    return {"messages": messages}


def _normalize_image_path(image: Any, images_base_dir: Optional[str]) -> str:
    """
    Normalize image path strictly relative to base directory.
    
    Args:
        image: Image path (string or other)
        images_base_dir: Base directory for relative paths
        
    Returns:
        Normalized image path string
    """
    if not isinstance(image, str):
        return str(image)
    p = Path(image)
    if p.is_absolute():
        return str(p)
    if images_base_dir is None:
        return str(p)
    return str(Path(images_base_dir) / p)


def _read_split_json(dataset_dir: str | Path, split: str) -> List[Dict[str, Any]]:
    path = Path(dataset_dir) / f"{split}.json"
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}, got {type(data)}")
    return data


def load_dump_dataset(
    dataset_dir: str,
    split: str = "train",
    images_base_dir: Optional[str] = None,
    sample_size: Optional[int] = None,
) -> Dataset:
    raw = _read_split_json(dataset_dir, split)
    items = [_normalize_dump_item(x, images_base_dir) for x in raw]
    if sample_size is not None and sample_size > 0:
        items = items[:sample_size]
    return Dataset.from_list(items)


def merge_mixed_datasets_with_ratio(
    ds_a: Dataset,
    ds_b: Dataset,
    ratio_a: float = 0.5,
) -> Dataset:
    # Create a merged dataset that approximately follows the desired ratio
    ratio_a = max(0.0, min(1.0, ratio_a))
    if len(ds_a) == 0:
        return ds_b
    if len(ds_b) == 0:
        return ds_a

    # Target lengths based on the larger dataset to avoid heavy duplication
    total_len = len(ds_a) + len(ds_b)
    target_a = max(1, int(total_len * ratio_a))
    target_b = max(1, total_len - target_a)

    def _repeat_to_length(ds: Dataset, target_len: int) -> Dataset:
        if len(ds) >= target_len:
            return ds.select(range(target_len))
        times = target_len // len(ds)
        remainder = target_len % len(ds)
        parts = [ds]
        for _ in range(max(0, times - 1)):
            parts.append(ds)
        if remainder > 0:
            parts.append(ds.select(range(remainder)))
        return concatenate_datasets(parts)

    ds_a_bal = _repeat_to_length(ds_a, target_a)
    ds_b_bal = _repeat_to_length(ds_b, target_b)
    merged = concatenate_datasets([ds_a_bal, ds_b_bal])
    return merged.shuffle(seed=42)
