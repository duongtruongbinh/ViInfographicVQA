import os
import json
import argparse
from tqdm import tqdm
from src.inference.single.models.utils import set_seed

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
    Dynamically import model class only when needed to avoid environment conflicts.
    
    Args:
        model_key: Key from MODELS dict
        
    Returns:
        Model class
    """
    if model_key not in MODELS:
        raise ValueError(f"Unknown model: {model_key}. Available: {list(MODELS.keys())}")
    
    module_path, class_name = MODELS[model_key].rsplit('.', 1)
    
    try:
        print(f"📦 Importing {class_name} from {module_path}...")
        module = __import__(module_path, fromlist=[class_name])
        model_class = getattr(module, class_name)
        return model_class
    except ImportError as e:
        print(f"❌ Failed to import {class_name}: {e}")
        print(f"💡 Make sure the required dependencies for {model_key} are installed")
        raise
    except AttributeError as e:
        print(f"❌ Class {class_name} not found in {module_path}: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with a selected model")
    parser.add_argument("model", type=str, choices=MODELS.keys(),
                        help=f"Name of the model to run. Choices: {list(MODELS.keys())}")
    parser.add_argument("--image_folder", type=str,
                        default="/mnt/VLAI_data/ViInfographicVQA/images", help="Directory of input images.")
    parser.add_argument("--data_path", type=str, default="data/test.json",
                        help="Path to the JSON file of sample questions.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save the results.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for processing (currently supports 1).")
    args = parser.parse_args()

    set_seed(args.seed)
    
    print(f"🚀 Initializing {args.model} model...")
    try:
        ModelClass = import_model_class(args.model)
        model = ModelClass()
        clean_model_name = model.model_name
        print(f"✅ Successfully loaded {clean_model_name}")
    except Exception as e:
        print(f"❌ Failed to initialize {args.model} model: {e}")
        return 1
    
    print(f"📂 Loading data from {args.data_path}...")
    with open(args.data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    output_filename = os.path.join(args.output_dir, f"{clean_model_name}.json")
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"📝 Processing {len(data)} samples with {clean_model_name}...")
    print(f"💾 Results will be saved to: {output_filename}")
    
    error_count = 0
    
    for i, item in enumerate(tqdm(data, desc=f"Processing with {clean_model_name}")):
        img_path = os.path.join(args.image_folder, f"{item['image_id']}.jpg")
        
        if not os.path.exists(img_path):
            print(f"⚠️  Image not found: {img_path}")
            item["predict"] = "ERROR: Image file not found"
            error_count += 1
            continue

        try:
            answer = model.infer(item['question'], img_path)
            item["predict"] = answer

            print(f"Q: {item['question']}")
            print(f"Predicted: {answer} | GT: {item['answer']}")
                
        except Exception as e:
            print(f"❌ Error processing item {item['image_id']}: {e}")
            item["predict"] = f"ERROR: {str(e)}"
            error_count += 1
    
    print(f"💾 Saving final results to {output_filename}")
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print("🎉 All done!")
    return 0

if __name__ == "__main__":
    exit(main()) 