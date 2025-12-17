import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from src.inference.single.models.base_model import VQAModel
from src.common.utils import get_system_prompt, parse_answer
from src.config import get_model_path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PhiModel(VQAModel):
    def __init__(self, model_path: str = None, **kwargs):
        super().__init__(**kwargs)
        self.model_path = model_path or get_model_path("phi")
        self._set_clean_model_name()
        self.load_model()

    def load_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
            _attn_implementation="flash_attention_2", device_map="auto"
        ).eval().to(DEVICE)
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        
    def infer(self, question: str, image_path: str) -> str:
        chat = [
            {"role": "system", "content": get_system_prompt()},
            {"role": "user", "content": f"<|image_1|>\nQuestion: {question}\nAnswer:"}
        ]

        prompt = self.processor.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        if prompt.endswith('<|endoftext|>'):
            prompt = prompt.rstrip('<|endoftext|>')
        
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            generate_ids = self.model.generate(**inputs, max_new_tokens=100, num_logits_to_keep=1)
        
        generate_ids = generate_ids[:, inputs["input_ids"].shape[1]:]
        response = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return parse_answer(response)