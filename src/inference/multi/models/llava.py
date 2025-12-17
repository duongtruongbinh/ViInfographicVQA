import torch
from pathlib import Path
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
from src.inference.multi.models.base_model import MultiImageVQAModel
from src.common.utils import load_images, parse_answer, get_system_prompt, format_user_input
from src.config import get_model_path, get_images_dir


class LlavaModel(MultiImageVQAModel):
    def __init__(self, model_path: str = None, images_base_dir: str = None):
        super().__init__()
        self.model_path = model_path or get_model_path("llava")
        self._set_clean_model_name()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.base_img_dir = Path(images_base_dir) if images_base_dir else Path(get_images_dir() or ".")
        self.load_model()

    def load_model(self):
        self.processor = LlavaNextProcessor.from_pretrained(self.model_path)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            self.model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
        ).to(self.device)

    def infer(self, question: str, images: list) -> str:
        image_objects = load_images(images, str(self.base_img_dir))
        
        content = [{"type": "image", "image": img_obj} for img_obj in image_objects]
        content.append({"type": "text", "text": format_user_input(question)})
        
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": get_system_prompt()}]},
            {"role": "user", "content": content}
        ]

        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(images=image_objects, text=prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=80)
        
        output_text = self.processor.decode(generated_ids[0], skip_special_tokens=True)
        return parse_answer(output_text)
