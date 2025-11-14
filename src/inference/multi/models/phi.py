import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from src.inference.multi.models.base_model import MultiImageVQAModel
from src.inference.multi.models.utils import get_image_paths, parse_answer, get_system_prompt, format_user_input


class PhiModel(MultiImageVQAModel):
    def __init__(self):
        super().__init__()
        self.model_path = "/mnt/dataset1/pretrained_fm/microsoft_Phi-4-multimodal-instruct"
        self._set_clean_model_name()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model()

    def load_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
            _attn_implementation="flash_attention_2", device_map="auto"
        ).eval().to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)

    def infer(self, question: str, images: list) -> str:
        image_paths = get_image_paths(images)
        image_placeholders = "".join([f"<|image_{i+1}|>" for i in range(len(image_paths))])
        
        chat = [
            {"role": "system", "content": get_system_prompt()},
            {"role": "user", "content": f"{image_placeholders}\n{format_user_input(question)}"}
        ]
        
        prompt = self.processor.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        if prompt.endswith('<|endoftext|>'):
            prompt = prompt.rstrip('<|endoftext|>')
        
        pil_images = [Image.open(image_path).convert("RGB") for image_path in image_paths]
        inputs = self.processor(text=prompt, images=pil_images, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            generate_ids = self.model.generate(**inputs, max_new_tokens=100, num_logits_to_keep=1)
        
        generate_ids = generate_ids[:, inputs["input_ids"].shape[1]:]
        response = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return parse_answer(response)
