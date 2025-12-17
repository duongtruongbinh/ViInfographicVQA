import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from src.inference.multi.models.base_model import MultiImageVQAModel
from src.common.utils import get_image_paths, get_system_prompt, format_user_input
from src.config import get_model_path


class AyaVisionModel(MultiImageVQAModel):
    def __init__(self, model_path: str = None):
        super().__init__()
        self.model_path = model_path or get_model_path("aya_vision")
        self._set_clean_model_name()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model()

    def load_model(self):
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_path, torch_dtype=torch.bfloat16
        ).eval().to(self.device)

    def infer(self, question: str, images: list) -> str:
        image_paths = get_image_paths(images)
        
        content = [{"type": "image", "image": img_path} for img_path in image_paths]
        content.append({"type": "text", "text": format_user_input(question)})
        
        messages = [
            {"role": "system", "content": [{"type": "text", "text": get_system_prompt()}]},
            {"role": "user", "content": content}
        ]

        inputs = self.processor.apply_chat_template(
            messages, padding=True, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad(), torch.autocast(device_type="cuda", enabled=torch.cuda.is_available(), dtype=torch.bfloat16):
            generated_ids = self.model.generate(**inputs, max_new_tokens=100, do_sample=True)
        
        output_text = self.processor.decode(generated_ids[0], skip_special_tokens=True)
        token = "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"
        if token in output_text:
            answer = output_text.split(token)[-1].strip()
            if answer.lower().startswith("predict:"):
                answer = answer.split(" ", 1)[1]
        else:
            answer = output_text.strip()
        
        return answer
