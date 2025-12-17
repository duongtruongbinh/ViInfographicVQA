"""Base class for single-image VQA models."""

from abc import ABC, abstractmethod
from typing import Any, Optional

from src.common.utils import extract_clean_model_name


class VQAModel(ABC):
    """Abstract base class for single-image VQA models."""

    def __init__(self, **kwargs: Any) -> None:
        self.model: Any = None
        self.processor: Any = None
        self.tokenizer: Any = None
        self.model_path: Optional[str] = None
        self.model_name: Optional[str] = None

    def _set_clean_model_name(self) -> None:
        """Extract clean model name from model path."""
        if self.model_path:
            self.model_name = extract_clean_model_name(self.model_path)

    @abstractmethod
    def load_model(self) -> None:
        """Load model weights and processor."""
        raise NotImplementedError

    @abstractmethod
    def infer(self, question: str, image_path: str) -> str:
        """
        Run inference on a single image.
        
        Args:
            question: Question about the image
            image_path: Path to the image file
            
        Returns:
            Model's answer string
        """
        raise NotImplementedError
