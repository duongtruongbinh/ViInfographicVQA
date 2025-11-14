import os
import json
import argparse
from tqdm import tqdm
from src.inference.multi.models.utils import set_seed

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
    """Dynamically imports model class to avoid environment conflicts."""
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
    parser = argparse.ArgumentParser(description="Run multi-image VQA inference with selected model")
    parser.add_argument("model", type=str, choices=MODELS.keys(),
                        help=f"Name of the model to run. Choices: {list(MODELS.keys())}")
    parser.add_argument("--data_path", type=str, default='data/multi_image_test.json',
                        help="Path to JSON file containing questions and image_ids")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory to save results")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--save_interval", type=int, default=100,
                        help="Save progress every N items")
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

    os.makedirs(args.output_dir, exist_ok=True)
    output_filename = os.path.join(args.output_dir, f"{clean_model_name}.json")
    
    print(f"📝 Processing {len(data)} samples with {clean_model_name}...")
    print(f"💾 Results will be saved to: {output_filename}")
    
    results = []
    error_count = 0
    
    for i, item in enumerate(tqdm(data, desc=f"Processing with {clean_model_name}")):
        image_paths = item.get("image_paths", [])
        image_ids = item.get("image_ids", [])
        question = item.get("question", "")
        gt_answer = item.get("answer", "")
        question_id = item.get("question_id", str(i))
        
        images_to_process = image_paths if image_paths else image_ids
        
        if not images_to_process:
            print(f"⚠️  No images found for item {i} (question_id: {question_id})")
            result_item = item.copy()
            result_item["predict"] = "ERROR: No images provided"
            if "answer" in result_item:
                result_item["gt"] = result_item["answer"]
            results.append(result_item)
            error_count += 1
            continue

        try:
            pred_answer = model.infer(question, images_to_process)
            
            result_item = item.copy()
            result_item["predict"] = pred_answer
            
            if "answer" in result_item:
                result_item["gt"] = result_item["answer"]
            
            results.append(result_item)

            print(f"Q: {question[:60]}...")
            print(f"Predicted: {pred_answer} | GT: {gt_answer}")
                
        except Exception as e:
            print(f"❌ Error processing item {i}: {e}")
            result_item = item.copy()
            result_item["predict"] = f"ERROR: {str(e)}"
            if "answer" in result_item:
                result_item["gt"] = result_item["answer"]
            results.append(result_item)
            error_count += 1

        if (i + 1) % args.save_interval == 0:
            print(f"💾 Saving progress at {i+1} items...")
            with open(output_filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"✅ Inference complete!")
    print(f"📊 Total errors: {error_count}/{len(data)}")
    
    print(f"💾 Saving final results to {output_filename}")
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("🎉 All done!")
    return 0


if __name__ == "__main__":
    exit(main())
