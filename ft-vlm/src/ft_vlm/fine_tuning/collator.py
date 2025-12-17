from __future__ import annotations

from typing import Any, Dict, List, Optional
from pathlib import Path

import torch
from PIL import Image


class VLMDataCollator:
    def __init__(
        self,
        processor,
        max_length: int | None = None,
        images_base_dir: Optional[str] = None,
        max_image_long_side: Optional[int] = None,
    ) -> None:
        self.processor = processor
        self.max_length = max_length
        self.images_base_dir = images_base_dir
        self.max_image_long_side = max_image_long_side

    def _resize_image_if_needed(self, img: Image.Image) -> Image.Image:
        if self.max_image_long_side is None:
            return img
        w, h = img.size
        long_side = max(w, h)
        if long_side <= self.max_image_long_side:
            return img
        scale = self.max_image_long_side / float(long_side)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        return img.resize((new_w, new_h))

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts: List[str] = []
        images_batch: List[List[Any]] = []
        for feat in features:
            messages = feat["messages"]
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)

            images: List[Any] = []
            for turn in messages:
                for item in turn.get("content", []):
                    if isinstance(item, dict) and item.get("type") == "image":
                        img = item.get("image")
                        if isinstance(img, str):
                            p = Path(img)
                            if not p.is_absolute() and self.images_base_dir:
                                p = Path(self.images_base_dir) / p
                            if not p.exists():
                                raise ValueError(
                                    f"Image file not found: {p}. "
                                    "Ensure all image paths are correct and files exist."
                                )
                            img = Image.open(p).convert("RGB")
                        if isinstance(img, Image.Image):
                            img = self._resize_image_if_needed(img)
                        images.append(img)
            images_batch.append(images)

        # Align number of provided images with the number of image tokens in text
        # Qwen2.5-VL replaces each <|image_pad|> occurrence with a number of tokens derived
        # from the processed images. If there are more placeholders than images provided,
        # the processor raises an IndexError.
        adjusted_images_flat: List[Any] = []
        image_token: str = getattr(self.processor, "image_token", "<|image_pad|>")
        for idx, text in enumerate(texts):
            expected = text.count(image_token)
            provided = len(images_batch[idx])

            if expected <= 0:
                continue

            if provided == 0:
                raise ValueError(
                    f"Sample {idx}: Expected {expected} image(s) but none were provided. "
                    "All images must be present and loadable. Check image paths in your dataset."
                )

            if provided < expected:
                raise ValueError(
                    f"Sample {idx}: Expected {expected} image(s) but only {provided} were provided. "
                    "Ensure all image paths in the dataset are correct and files exist."
                )

            # Use only the expected number of images
            adjusted_images_flat.extend(images_batch[idx][:expected])

        batch = self.processor(
            text=texts,
            images=adjusted_images_flat if any(t.count(image_token) > 0 for t in texts) else images_batch,
            padding=True,
            # IMPORTANT: Do not truncate multimodal sequences; it can desync
            # the number of special image tokens between text and ids.
            truncation=False,
            return_tensors="pt",
        )
        # Labels are input_ids for SFT
        batch["labels"] = batch["input_ids"].clone()
        return batch
