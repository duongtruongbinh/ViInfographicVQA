"""Qwen2.5-VL model wrapper for multi-image VQA."""

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from src.config import get_model_path
from src.inference.multi.models.base_model import MultiImageVQAModel
from src.common.utils import (
    format_user_input,
    get_image_paths,
    get_system_prompt,
    parse_answer,
)


class QwenVLModel(MultiImageVQAModel):
    """Qwen2.5-VL model for multi-image VQA."""

    def __init__(self, model_path: str = None):
        super().__init__()
        self.model_path = model_path or get_model_path("qwenvl")
        self._set_clean_model_name()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model()

    def load_model(self) -> None:
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path, torch_dtype=torch.bfloat16
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            min_pixels=256 * 28 * 28,
            max_pixels=1280 * 28 * 28,
            use_fast=True,
        )

    def infer(self, question: str, images: list) -> str:
        image_paths = get_image_paths(images)

        content = [{"type": "image", "image": f"file://{p}"} for p in image_paths]
        content.append({"type": "text", "text": format_user_input(question)})

        messages = [
            {"role": "system", "content": [{"type": "text", "text": get_system_prompt()}]},
            {"role": "user", "content": content},
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad(), torch.autocast(
            device_type="cuda", enabled=torch.cuda.is_available(), dtype=torch.bfloat16
        ):
            generated_ids = self.model.generate(**inputs, max_new_tokens=80)

        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return parse_answer(output[0])
