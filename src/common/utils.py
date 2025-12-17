"""Shared utility functions for VQA inference."""

import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image


SYSTEM_PROMPT = (
    "Answer the following question based solely on the image content "
    "concisely with a single term."
)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def extract_clean_model_name(model_path: str) -> str:
    """
    Extract clean model name from path or HuggingFace ID.

    Args:
        model_path: Full path or HuggingFace model ID

    Returns:
        Clean model name without organization prefix
    """
    name = model_path.split("/")[-1]
    if "_" in name:
        parts = name.split("_", 1)
        if len(parts) > 1:
            return parts[1]
    return name


def extract_clean_filename(filename: str) -> str:
    """
    Extract model name from prediction filename.

    Args:
        filename: Prediction filename (e.g., "model_name.json")

    Returns:
        Clean model name
    """
    base = filename.replace(".json", "")
    if "_" in base:
        parts = base.split("_")
        return "_".join(parts[1:]) if len(parts) > 2 else base
    return base


def get_system_prompt() -> str:
    """Get system prompt for VQA models."""
    return SYSTEM_PROMPT


def format_user_input(question: str) -> str:
    """Format user input with question prefix."""
    return f"Question: {question.strip()}"


def parse_answer(response: str) -> str:
    """Extract answer from model response."""
    if "Answer:" in response:
        return response.split("Answer:")[-1].strip()
    return response.strip()


def load_images(
    image_ids: list[str],
    base_img_dir: Optional[str] = None,
) -> list[Image.Image]:
    """
    Load images from IDs or paths.

    Args:
        image_ids: List of image IDs or absolute paths
        base_img_dir: Base directory for relative paths

    Returns:
        List of PIL Image objects

    Raises:
        FileNotFoundError: If image file not found
        ValueError: If relative path without base_img_dir
    """
    from src.config import get_images_dir

    images = []
    resolved_base = base_img_dir or get_images_dir()

    for img_id in image_ids:
        img_str = str(img_id).strip()

        if img_str.startswith("/"):
            img_path = Path(img_str).resolve()
        elif resolved_base:
            img_path = (Path(resolved_base) / f"{img_str}.jpg").resolve()
        else:
            raise ValueError(
                f"Cannot resolve '{img_str}': base_img_dir not set and "
                "VQA_IMAGES_DIR not configured"
            )

        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        images.append(Image.open(img_path).convert("RGB"))

    return images


def get_image_paths(
    image_ids: list[str],
    base_img_dir: Optional[str] = None,
) -> list[str]:
    """
    Get full paths for image IDs.

    Args:
        image_ids: List of image IDs or absolute paths
        base_img_dir: Base directory for relative paths

    Returns:
        List of absolute image paths

    Raises:
        FileNotFoundError: If image file not found
        ValueError: If relative path without base_img_dir
    """
    from src.config import get_images_dir

    paths = []
    resolved_base = base_img_dir or get_images_dir()

    for img_id in image_ids:
        img_str = str(img_id).strip()

        if img_str.startswith("/"):
            img_path = Path(img_str).resolve()
        elif resolved_base:
            img_path = (Path(resolved_base) / f"{img_str}.jpg").resolve()
        else:
            raise ValueError(
                f"Cannot resolve '{img_str}': base_img_dir not set and "
                "VQA_IMAGES_DIR not configured"
            )

        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        paths.append(str(img_path))

    return paths
