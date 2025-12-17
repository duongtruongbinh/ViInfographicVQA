import torch
from PIL import Image
from transformers import AutoModelForCausalLM
from src.inference.multi.models.base_model import MultiImageVQAModel
from src.common.utils import parse_answer, get_system_prompt, format_user_input
from src.config import get_model_path


class OvisModel(MultiImageVQAModel):
    def __init__(self, model_path: str = None):
        super().__init__()
        self.model_path = model_path or get_model_path("ovis")
        self._set_clean_model_name()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model()

    def load_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, torch_dtype=torch.bfloat16, multimodal_max_length=32768, trust_remote_code=True, device_map="auto"
        ).eval().to(self.device)

    def infer(self, question: str, images: list) -> str:
        content = [{"type": "image", "image": Image.open(p).convert("RGB")} for p in images]
        content.append({"type": "text", "text": format_user_input(question)})
        
        messages = [
            {"role": "system", "content": get_system_prompt()},
            {"role": "user", "content": content}
        ]
        
        input_ids, pixel_values, grid_thws = self.model.preprocess_inputs(messages, add_generation_prompt=True, enable_thinking=False)
        input_ids = input_ids.cuda()
        pixel_values = pixel_values.cuda() if pixel_values is not None else None
        grid_thws = grid_thws.cuda() if grid_thws is not None else None
        
        with torch.inference_mode():
            outputs = self.model.generate(
                inputs=input_ids, pixel_values=pixel_values, grid_thws=grid_thws,
                enable_thinking=False, enable_thinking_budget=False,
                max_new_tokens=100, thinking_budget=0, eos_token_id=self.model.text_tokenizer.eos_token_id
            )
        
        output = self.model.text_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return parse_answer(output)
