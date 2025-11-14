from abc import ABC, abstractmethod
from src.inference.single.models.utils import extract_clean_model_name


class VQAModel(ABC):
    """Abstract base class for VQA models."""

    def __init__(self, **kwargs):
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.model_path = None
        self.model_name = None

    def _set_clean_model_name(self):
        """Sets clean model name from model path."""
        if self.model_path:
            self.model_name = extract_clean_model_name(self.model_path)

    @abstractmethod
    def load_model(self):
        """Loads model and processor/tokenizer."""
        raise NotImplementedError

    @abstractmethod
    def infer(self, question: str, image_path: str) -> str:
        """Performs inference on image-question pair, returns answer."""
        raise NotImplementedError 