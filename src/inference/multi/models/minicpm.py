import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from src.inference.multi.models.base_model import MultiImageVQAModel
from src.inference.multi.models.utils import get_image_paths, parse_answer, get_system_prompt, format_user_input


class MiniCPMModel(MultiImageVQAModel):
    def __init__(self):
        super().__init__()
        self.model_path = "/mnt/dataset1/pretrained_fm/openbmb_MiniCPM-o-2-6"
        self._set_clean_model_name()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model()

    def load_model(self):
        self.model = AutoModel.from_pretrained(
            self.model_path, trust_remote_code=True, attn_implementation='flash_attention_2',
            torch_dtype=torch.bfloat16, init_vision=True, init_audio=False, init_tts=False,
            low_cpu_mem_usage=True, device_map=self.device
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.model.init_tts()

    def infer(self, question: str, images: list) -> str:
        image_paths = get_image_paths(images)
        pil_images = [Image.open(image_path).convert('RGB') for image_path in image_paths]
        combined_text = f"{get_system_prompt()}\n\n{format_user_input(question)}"
        
        msgs = [{'role': 'user', 'content': pil_images + [combined_text]}]
        
        with torch.no_grad():
            response = self.model.chat(msgs=msgs, tokenizer=self.tokenizer)

        return parse_answer(response)
