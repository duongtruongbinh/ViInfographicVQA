import torch


def set_seed(seed: int) -> None:
    """Sets random seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def extract_clean_model_name(model_path_or_name: str) -> str:
    """Extracts clean model name from HuggingFace path (removes org prefix)."""
    model_name = model_path_or_name.split('/')[-1]
    if '_' in model_name:
        parts = model_name.split('_', 1)
        if len(parts) > 1:
            return parts[1]
    return model_name


def extract_clean_filename(filename: str) -> str:
    """Extracts clean model name from prediction filename."""
    base_name = filename.replace('.json', '')
    if '_' in base_name:
        parts = base_name.split('_')
        return '_'.join(parts[1:]) if len(parts) > 2 else base_name
    return base_name


def get_system_prompt() -> str:
    """Returns system prompt for VQA models."""
    return "Answer the following question based solely on the image content consisely with a single term."


def parse_answer(response: str) -> str:
    """Extracts answer from model response."""
    return response.split("Answer:")[-1].strip()