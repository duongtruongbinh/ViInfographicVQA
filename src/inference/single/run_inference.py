"""Single-image VQA inference runner."""

import argparse
import json
import os

from tqdm import tqdm

from src.common.utils import set_seed

MODELS = {
    "internvl": "src.inference.single.models.internvl.InternVLModel",
    "molmo": "src.inference.single.models.molmo.MolmoModel",
    "qwenvl": "src.inference.single.models.qwenvl.QwenVLModel",
    "videollama": "src.inference.single.models.videollama.VideoLLAMAModel",
    "phi": "src.inference.single.models.phi.PhiModel",
    "ovis": "src.inference.single.models.ovis.OvisModel",
    "minicpm": "src.inference.single.models.minicpm.MiniCPMModel",
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
    from src.config import get_images_dir

    parser = argparse.ArgumentParser(description="Run single-image VQA inference")
    parser.add_argument(
        "model",
        type=str,
        choices=MODELS.keys(),
        help=f"Model to use: {list(MODELS.keys())}",
    )
    parser.add_argument(
        "--image_folder",
        type=str,
        default=None,
        help="Image directory (or set VQA_IMAGES_DIR)",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/test.json",
        help="Input JSON file path",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output_dir", type=str, default="results/single", help="Output directory"
    )
    args = parser.parse_args()

    if args.image_folder is None:
        args.image_folder = get_images_dir()
        if args.image_folder is None:
            print("Error: --image_folder not set and VQA_IMAGES_DIR not configured")
            return 1

    set_seed(args.seed)

    model_class = import_model_class(args.model)
    model = model_class()
    model_name = model.model_name

    with open(args.data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"{model_name}.json")

    error_count = 0
    for item in tqdm(data, desc=f"Inference ({model_name})"):
        img_path = os.path.join(args.image_folder, f"{item['image_id']}.jpg")

        if not os.path.exists(img_path):
            item["predict"] = "ERROR: Image not found"
            error_count += 1
            continue

        try:
            item["predict"] = model.infer(item["question"], img_path)
        except Exception as e:
            item["predict"] = f"ERROR: {e}"
            error_count += 1

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(data)} results to {output_path} (errors: {error_count})")
    return 0


if __name__ == "__main__":
    exit(main())
