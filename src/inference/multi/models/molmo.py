import torch
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from src.inference.multi.models.base_model import MultiImageVQAModel
from src.inference.multi.models.utils import load_images, parse_answer, get_system_prompt, format_user_input

orig_stack = np.stack
def patched_stack(arrays, axis=0, *args, **kwargs):
    kwargs.pop("dtype", None)
    return orig_stack(arrays, axis=axis, *args, **kwargs)
np.stack = patched_stack


class MolmoModel(MultiImageVQAModel):
    def __init__(self):
        super().__init__()
        self.model_path = "/mnt/dataset1/pretrained_fm/allenai_Molmo-7B-D-0924"
        self._set_clean_model_name()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.base_img_dir = Path("/mnt/VLAI_data/ViInfographicVQA/images")
        self.load_model()

    def load_model(self):
        self.processor = AutoProcessor.from_pretrained(
            self.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto"
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map="auto"
        ).eval()
        self.tokenizer = self.processor.tokenizer

    def infer(self, question: str, images: list) -> str:
        image_objects = load_images(images, str(self.base_img_dir))
        instruction = f"{get_system_prompt()}\n\n{format_user_input(question)}"

        inputs = self.processor.process(images=image_objects, text=instruction)
        inputs = {k: v.to(self.device).unsqueeze(0) for k, v in inputs.items()}
        inputs["images"] = inputs["images"].to(torch.bfloat16)

        with torch.no_grad(), torch.autocast(device_type="cuda", enabled=torch.cuda.is_available(), dtype=torch.bfloat16):
            output = self.model.generate_from_batch(
                inputs, GenerationConfig(max_new_tokens=100, stop_strings="<|endoftext|>"), tokenizer=self.tokenizer
            )
        
        generated_tokens = output[0, inputs['input_ids'].size(1):]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return parse_answer(generated_text)
