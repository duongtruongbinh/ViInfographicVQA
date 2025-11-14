import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from src.inference.multi.models.base_model import MultiImageVQAModel
from src.inference.multi.models.utils import get_image_paths, parse_answer, get_system_prompt, format_user_input


class QwenVLModel(MultiImageVQAModel):
    def __init__(self):
        super().__init__()
        self.model_path = "/mnt/dataset1/pretrained_fm/Qwen_Qwen2.5-VL-7B-Instruct"
        self._set_clean_model_name()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model()

    def load_model(self):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path, torch_dtype=torch.bfloat16
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(
            self.model_path, min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28, use_fast=True
        )

    def infer(self, question: str, images: list) -> str:
        image_paths = get_image_paths(images)
        
        content = [{"type": "image", "image": f"file://{img_path}"} for img_path in image_paths]
        content.append({"type": "text", "text": format_user_input(question)})
        
        messages = [
            {"role": "system", "content": [{"type": "text", "text": get_system_prompt()}]},
            {"role": "user", "content": content}
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad(), torch.autocast(device_type="cuda", enabled=torch.cuda.is_available(), dtype=torch.bfloat16):
            generated_ids = self.model.generate(**inputs, max_new_tokens=80)

        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return parse_answer(output_text[0])
