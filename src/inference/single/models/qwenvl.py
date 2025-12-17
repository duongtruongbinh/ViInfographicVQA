"""Qwen2.5-VL model wrapper for single-image VQA."""

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from src.config import get_model_path
from src.inference.single.models.base_model import VQAModel
from src.common.utils import get_system_prompt, parse_answer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QwenVLModel(VQAModel):
    """Qwen2.5-VL model for single-image VQA."""

    def __init__(self, model_path: str = None, **kwargs):
        super().__init__(**kwargs)
        self.model_path = model_path or get_model_path("qwenvl")
        self._set_clean_model_name()
        self.load_model()

    def load_model(self) -> None:
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path, torch_dtype=torch.bfloat16, device_map=DEVICE
        ).eval()
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            min_pixels=256 * 28 * 28,
            max_pixels=1280 * 28 * 28,
            use_fast=True,
        )
        self.processor.tokenizer.padding_side = "left"

    def infer(self, question: str, image_path: str) -> str:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": get_system_prompt()}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{image_path}"},
                    {"type": "text", "text": f"Question: {question}\nAnswer:"},
                ],
            },
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
        ).to(DEVICE)

        with torch.no_grad(), torch.autocast(
            device_type="cuda", enabled=torch.cuda.is_available(), dtype=torch.bfloat16
        ):
            generated_ids = self.model.generate(**inputs, max_new_tokens=100)

        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return parse_answer(response)
