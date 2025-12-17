import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from src.inference.single.models.base_model import VQAModel
from src.common.utils import get_system_prompt, parse_answer
from src.config import get_model_path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MiniCPMModel(VQAModel):
    def __init__(self, model_path: str = None, **kwargs):
        super().__init__(**kwargs)
        self.model_path = model_path or get_model_path("minicpm")
        self._set_clean_model_name()
        self.load_model()

    def load_model(self):
        self.model = AutoModel.from_pretrained(
            self.model_path, trust_remote_code=True, attn_implementation='flash_attention_2',
            torch_dtype=torch.bfloat16, init_vision=True, init_audio=False, init_tts=False,
            low_cpu_mem_usage=True, device_map=DEVICE
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.model.init_tts()

    def infer(self, question: str, image_path: str) -> str:
        image = Image.open(image_path).convert('RGB')
        messages = [
            {"role": "system", "content": get_system_prompt()},
            {"role": "user", "content": [image, f"Question: {question}\nAnswer:"]}
        ]
        
        with torch.no_grad():
            res = self.model.chat(image=image, msgs=messages, tokenizer=self.tokenizer)
        
        return parse_answer(res)