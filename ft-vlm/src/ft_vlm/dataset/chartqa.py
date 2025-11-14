from __future__ import annotations

from typing import Any, Dict, List, Optional

from datasets import Dataset, load_dataset


SYSTEM_PROMPT = "You are a helpful assistant that answers questions about charts."


def _extract_image_field(example: Dict[str, Any]) -> Any:
    if "image" in example and example["image"] is not None:
        return example["image"]
    if "img" in example and example["img"] is not None:
        return example["img"]
    # Some variants may use "plot" or store image path under "image_path"
    if "plot" in example and example["plot"] is not None:
        return example["plot"]
    if "image_path" in example and example["image_path"] is not None:
        return example["image_path"]
    return None


def _extract_answer_field(example: Dict[str, Any]) -> Optional[str]:
    for key in ("answer", "final_answer", "label", "annotation", "human_answer"):
        val = example.get(key)
        if isinstance(val, str) and val.strip():
            return val
    # Some datasets store answers as list
    val = example.get("answers")
    if isinstance(val, list) and len(val) > 0 and isinstance(val[0], str):
        return val[0]
    return None


def _format_sample_to_messages(example: Dict[str, Any]) -> Dict[str, Any]:
    image = _extract_image_field(example)
    question = example.get("question") or example.get("query") or ""
    answer = _extract_answer_field(example) or ""

    messages: List[Dict[str, Any]] = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": SYSTEM_PROMPT},
            ],
        },
        {
            "role": "user",
            "content": [
                *(([{"type": "image", "image": image}] if image is not None else [])),
                {"type": "text", "text": question},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": answer},
            ],
        },
    ]

    return {"messages": messages}


def load_chartqa_dataset(
    split: str = "train",
    name: Optional[str] = None,
    sample_size: Optional[int] = None,
) -> Dataset:
    """
    Load ChartQA and map to Qwen2-VL chat-style messages suitable for VLM SFT.

    - Each item will contain a "messages" key compatible with Qwen2-VL chat template.
    - Images are kept inline (PIL.Image) by datasets; TRL data collator will process them.
    """

    dataset_name = "HuggingFaceM4/ChartQA"
    ds = load_dataset(dataset_name, name)  # type: ignore[arg-type]
    if split not in ds:
        raise ValueError(f"Requested split '{split}' not found in {list(ds.keys())}")

    mapped: Dataset = ds[split].map(
        _format_sample_to_messages,
        remove_columns=[col for col in ds[split].column_names if col != "messages"],
    )

    if sample_size is not None and sample_size > 0:
        mapped = mapped.select(range(min(sample_size, len(mapped))))

    return mapped
