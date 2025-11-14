import torch
import random
import numpy as np
from pathlib import Path
from PIL import Image

SYSTEM_PROMPT = "Answer the following question based solely on the image contents consisely with a single term."


def set_seed(seed: int = 42):
    """Sets random seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def extract_clean_model_name(model_path_or_name: str) -> str:
    """Extracts clean model name from HuggingFace path (removes org prefix)."""
    model_name = model_path_or_name.split('/')[-1]
    if '_' in model_name:
        parts = model_name.split('_', 1)
        if len(parts) > 1:
            return parts[1]
    return model_name


def get_system_prompt() -> str:
    """Returns system prompt for multi-image VQA models."""
    return SYSTEM_PROMPT


def format_user_input(question: str) -> str:
    """Formats user input with question."""
    return f"Question: {question.strip()}"


def parse_answer(response: str) -> str:
    """Extracts answer from model response."""
    if "Answer:" in response:
        return response.split("Answer:")[-1].strip()
    return response.strip()


def load_images(image_ids: list, base_img_dir: str = "/mnt/VLAI_data/ViInfographicVQA/images") -> list:
    """Loads images from image IDs or paths."""
    images = []
    base_path = Path(base_img_dir)
    
    for img_id in image_ids:
        img_str = str(img_id).strip()
        img_path = Path(img_str).resolve() if img_str.startswith('/') else (base_path / f"{img_str}.jpg").resolve()
        
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        images.append(Image.open(img_path).convert("RGB"))
    
    return images


def get_image_paths(image_ids: list, base_img_dir: str = "/mnt/VLAI_data/ViInfographicVQA/images") -> list:
    """Gets full image paths from image IDs or paths."""
    paths = []
    base_path = Path(base_img_dir)
    
    for img_id in image_ids:
        img_str = str(img_id).strip()
        img_path = Path(img_str).resolve() if img_str.startswith('/') else (base_path / f"{img_str}.jpg").resolve()
        
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        paths.append(str(img_path))
    
    return paths
