"""
Centralized configuration for model paths and data directories.

Paths are resolved from environment variables with sensible defaults.
Environment variables allow portability across different machines.
"""

import os
from pathlib import Path
from typing import Optional


def _get_env_path(env_var: str, default: Optional[str] = None) -> Optional[str]:
    """Get path from environment variable, returning None if not set and no default."""
    value = os.environ.get(env_var)
    if value:
        return value
    return default


# Base directories (configured via environment variables)
MODELS_BASE_DIR = _get_env_path("VQA_MODELS_DIR")
DATA_BASE_DIR = _get_env_path("VQA_DATA_DIR")
IMAGES_BASE_DIR = _get_env_path("VQA_IMAGES_DIR")
OUTPUT_BASE_DIR = _get_env_path("VQA_OUTPUT_DIR", "./outputs")

# Model identifiers - use HuggingFace IDs or local paths
MODEL_PATHS = {
    "qwenvl": _get_env_path("VQA_MODEL_QWENVL", "Qwen/Qwen2.5-VL-7B-Instruct"),
    "internvl": _get_env_path("VQA_MODEL_INTERNVL", "OpenGVLab/InternVL3_5-8B"),
    "phi": _get_env_path("VQA_MODEL_PHI", "microsoft/Phi-4-multimodal-instruct"),
    "minicpm": _get_env_path("VQA_MODEL_MINICPM", "openbmb/MiniCPM-o-2-6"),
    "molmo": _get_env_path("VQA_MODEL_MOLMO", "allenai/Molmo-7B-D-0924"),
    "ovis": _get_env_path("VQA_MODEL_OVIS", "AIDC-AI/Ovis2.5-9B"),
    "videollama": _get_env_path("VQA_MODEL_VIDEOLLAMA", "DAMO-NLP-SG/VideoLLaMA3-7B-Image"),
    "llava": _get_env_path("VQA_MODEL_LLAVA", "unsloth/llava-v1.6-mistral-7b-hf-bnb-4bit"),
    "aya_vision": _get_env_path("VQA_MODEL_AYA_VISION", "CohereLabs/aya-vision-8b"),
}

def get_model_path(model_key: str) -> str:
    """
    Get the model path for a given model key.
    
    Args:
        model_key: Key identifying the model (e.g., 'qwenvl', 'internvl')
        
    Returns:
        Model path (HuggingFace ID or local path)
        
    Raises:
        ValueError: If model_key is not recognized
    """
    if model_key not in MODEL_PATHS:
        raise ValueError(
            f"Unknown model key: {model_key}. "
            f"Available keys: {list(MODEL_PATHS.keys())}"
        )
    return MODEL_PATHS[model_key]


def get_images_dir(override: Optional[str] = None) -> Optional[str]:
    """
    Get the images base directory.
    
    Args:
        override: Optional path that takes precedence
        
    Returns:
        Images directory path or None if not configured
    """
    if override:
        return override
    return IMAGES_BASE_DIR


def get_output_dir(model_name: str, base_dir: Optional[str] = None) -> str:
    """
    Get output directory for training artifacts.
    
    Args:
        model_name: Model name or ID
        base_dir: Optional base directory override
        
    Returns:
        Path to output directory
    """
    base = base_dir or OUTPUT_BASE_DIR
    clean_name = model_name.split("/")[-1].replace(" ", "_")
    return str(Path(base) / clean_name)


def resolve_path(
    path: Optional[str],
    base_dir: Optional[str] = None,
    must_exist: bool = False
) -> Optional[str]:
    """
    Resolve a path, optionally relative to a base directory.
    
    Args:
        path: Path to resolve (can be relative or absolute)
        base_dir: Base directory for relative paths
        must_exist: If True, raise FileNotFoundError for non-existent paths
        
    Returns:
        Resolved absolute path, or None if path is None
        
    Raises:
        FileNotFoundError: If must_exist=True and path doesn't exist
    """
    if path is None:
        return None
        
    p = Path(path)
    
    if p.is_absolute():
        resolved = p
    elif base_dir:
        resolved = Path(base_dir) / p
    else:
        resolved = p.resolve()
    
    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"Path does not exist: {resolved}")
    
    return str(resolved)
