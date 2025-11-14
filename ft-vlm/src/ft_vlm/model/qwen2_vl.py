from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import importlib.metadata
import platform
import torch
from transformers import BitsAndBytesConfig

# Try to import both generations of Qwen VL classes; fall back gracefully
try:  # Qwen2.5-VL (transformers >= 4.44)
    from transformers import (
        Qwen2_5_VLForConditionalGeneration as Qwen25VLForConditionalGeneration,
    )
    from transformers import (
        Qwen2_5_VLProcessor as Qwen25VLProcessor,
    )
except Exception:  # pragma: no cover - not available
    Qwen25VLForConditionalGeneration = None  # type: ignore
    Qwen25VLProcessor = None  # type: ignore

try:  # Qwen2-VL
    from transformers import (
        Qwen2VLForConditionalGeneration as Qwen2VLForConditionalGeneration,
        Qwen2VLProcessor as Qwen2VLProcessor,
    )
except Exception:  # pragma: no cover - should exist with our pinned transformers
    Qwen2VLForConditionalGeneration = None  # type: ignore
    Qwen2VLProcessor = None  # type: ignore


@dataclass
class QLoRAConfig:
    use_qlora: bool = True
    bnb_4bit_compute_dtype: torch.dtype = torch.bfloat16
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True


def build_model_and_processor(
    model_id: str,
    qlora: Optional[QLoRAConfig] = None,
    device_map: Optional[str | Dict[str, int]] = "auto",
) -> Tuple[Any, Any, Optional[Any]]:
    qlora = qlora or QLoRAConfig()

    # Detect bitsandbytes availability (not present on macOS usually)
    has_bnb = False
    try:
        importlib.metadata.version("bitsandbytes")
        has_bnb = True
    except importlib.metadata.PackageNotFoundError:
        has_bnb = False

    use_4bit = bool(qlora.use_qlora and has_bnb)

    # Choose dtype suitable for the available device
    if torch.cuda.is_available():
        preferred_dtype = torch.bfloat16
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        preferred_dtype = torch.float16
    else:
        preferred_dtype = torch.float32

    quantization_config = None
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=qlora.bnb_4bit_use_double_quant,
            bnb_4bit_quant_type=qlora.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=qlora.bnb_4bit_compute_dtype,
        )

    # Select correct model/processor classes for Qwen2.5-VL vs Qwen2-VL
    use_qwen25 = "Qwen2.5-VL" in model_id or "Qwen2_5-VL" in model_id

    if (
        use_qwen25
        and Qwen25VLForConditionalGeneration is not None
        and Qwen25VLProcessor is not None
    ):
        model = Qwen25VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=preferred_dtype,
            device_map=(
                {"": "cpu"}
                if (
                    device_map == "auto"
                    and not (
                        torch.cuda.is_available()
                        or (
                            hasattr(torch.backends, "mps")
                            and torch.backends.mps.is_available()
                        )
                    )
                )
                else device_map
            ),
            low_cpu_mem_usage=False,
            quantization_config=quantization_config,
        )
        processor = Qwen25VLProcessor.from_pretrained(model_id)
    else:
        if Qwen2VLForConditionalGeneration is None or Qwen2VLProcessor is None:
            raise RuntimeError(
                "Required Qwen VL classes are not available in the installed transformers."
            )
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=preferred_dtype,
            device_map=(
                {"": "cpu"}
                if (
                    device_map == "auto"
                    and not (
                        torch.cuda.is_available()
                        or (
                            hasattr(torch.backends, "mps")
                            and torch.backends.mps.is_available()
                        )
                    )
                )
                else device_map
            ),
            low_cpu_mem_usage=False,
            quantization_config=quantization_config,
        )
        processor = Qwen2VLProcessor.from_pretrained(model_id)

    return model, processor, quantization_config
