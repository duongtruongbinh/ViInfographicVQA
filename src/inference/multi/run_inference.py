"""Multi-image VQA inference runner."""

import argparse
import json
import os

from tqdm import tqdm

from src.common.utils import set_seed

MODELS = {
    "aya_vision": "src.inference.multi.models.aya_vision.AyaVisionModel",
    "llava": "src.inference.multi.models.llava.LlavaModel",
    "ovis": "src.inference.multi.models.ovis.OvisModel",
    "qwenvl": "src.inference.multi.models.qwenvl.QwenVLModel",
    "internvl": "src.inference.multi.models.internvl.InternVLModel",
    "molmo": "src.inference.multi.models.molmo.MolmoModel",
    "phi": "src.inference.multi.models.phi.PhiModel",
    "minicpm": "src.inference.multi.models.minicpm.MiniCPMModel",
}


def import_model_class(model_key: str):
    """
    Dynamically import model class to avoid dependency conflicts.
    
    Args:
        model_key: Key from MODELS dict
        
    Returns:
        Model class
    """
    if model_key not in MODELS:
        raise ValueError(f"Unknown model: {model_key}. Available: {list(MODELS.keys())}")

    module_path, class_name = MODELS[model_key].rsplit(".", 1)
    module = __import__(module_path, fromlist=[class_name])
    return getattr(module, class_name)


def main():
    parser = argparse.ArgumentParser(description="Run multi-image VQA inference")
    parser.add_argument(
        "model",
        type=str,
        choices=MODELS.keys(),
        help=f"Model to use: {list(MODELS.keys())}",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/multi_image_test.json",
        help="Input JSON file path",
    )
    parser.add_argument(
        "--output_dir", type=str, default="results/multi", help="Output directory"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--save_interval", type=int, default=100, help="Save progress interval"
    )
    args = parser.parse_args()

    set_seed(args.seed)

    model_class = import_model_class(args.model)
    model = model_class()
    model_name = model.model_name

    with open(args.data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"{model_name}.json")

    results = []
    error_count = 0

    for i, item in enumerate(tqdm(data, desc=f"Inference ({model_name})")):
        images = item.get("image_paths", []) or item.get("image_ids", [])
        question = item.get("question", "")

        result = item.copy()

        if not images:
            result["predict"] = "ERROR: No images"
            error_count += 1
        else:
            try:
                result["predict"] = model.infer(question, images)
            except Exception as e:
                result["predict"] = f"ERROR: {e}"
                error_count += 1

        if "answer" in result:
            result["gt"] = result["answer"]
        results.append(result)

        if (i + 1) % args.save_interval == 0:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(results)} results to {output_path} (errors: {error_count})")
    return 0


if __name__ == "__main__":
    exit(main())
