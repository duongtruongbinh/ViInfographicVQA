import os
import json
import Levenshtein
from tqdm import tqdm
from collections import defaultdict


def compute_anls(gt: str, predict: str, threshold: float = 0.5) -> float:
    """Computes ANLS score between ground truth and prediction."""
    p = predict.replace('"', '').rstrip('.').lower()
    g = gt.lower()
    score = Levenshtein.ratio(p, g)
    return score if score >= threshold else 0.0


def analyze_by_categories(preds: list) -> dict:
    """Analyzes predictions by categories."""
    category_scores = {
        "Overall": {"total_anls": 0, "count": 0},
        "Answer type": defaultdict(lambda: {"total_anls": 0, "count": 0}),
        "Image type": defaultdict(lambda: {"total_anls": 0, "count": 0}),
    }
    
    for item in preds:
        if 'answer' not in item or 'predict' not in item:
            continue

        gt_clean = item['answer'].lower().strip()
        pr_clean = item['predict'].lower().strip().rstrip('.').replace('"', '')
        anls_score = compute_anls(gt_clean, pr_clean)
        
        item['anls'] = anls_score

        category_scores["Overall"]["total_anls"] += anls_score
        category_scores["Overall"]["count"] += 1

        answer_source = item.get('answer_source', 'N/A')
        if isinstance(answer_source, list):
            answer_source = answer_source[0] if answer_source else 'N/A'
        answer_source = str(answer_source).lower()
        
        category_scores["Answer type"][answer_source]["total_anls"] += anls_score
        category_scores["Answer type"][answer_source]["count"] += 1

        image_type = item.get('image_type', 'N/A')
        if isinstance(image_type, list):
            image_type = image_type[0] if image_type else 'N/A'
        image_type = str(image_type).lower()
        
        category_scores["Image type"][image_type]["total_anls"] += anls_score
        category_scores["Image type"][image_type]["count"] += 1

    return category_scores


def calculate_averages(stats: dict) -> dict:
    """Calculates average ANLS score."""
    count = stats['count']
    return {'anls': round((stats['total_anls'] / count) * 100, 2), 'count': count} if count > 0 else {'anls': 0, 'count': 0}


def create_detailed_report(category_scores: dict) -> dict:
    """Creates detailed analysis report."""
    report = {"Overall": calculate_averages(category_scores["Overall"])}
    
    for category_key, sub_categories in category_scores.items():
        if category_key == "Overall":
            continue
        report[category_key] = {
            sub_category: calculate_averages(stats)
            for sub_category, stats in sorted(sub_categories.items())
        }
    return report


def save_anls_report(detailed_analysis: dict, filename: str):
    """Saves ANLS analysis to tab-separated file."""
    excluded_keys = {'none', '[]', 'n/a'}
    
    all_answer_types = set()
    all_image_types = set()
    
    for analysis_data in detailed_analysis.values():
        if "Answer type" in analysis_data:
            all_answer_types.update(analysis_data["Answer type"].keys())
        if "Image type" in analysis_data:
            all_image_types.update(analysis_data["Image type"].keys())
    
    valid_answer_types = sorted([k for k in all_answer_types if str(k) not in excluded_keys])
    valid_image_types = sorted([k for k in all_image_types if str(k) not in excluded_keys])
    
    header = ["Method", "Overall"] + [t.replace("-", " ").title() for t in valid_answer_types] + [t.replace("-", " ").title() for t in valid_image_types]
    
    rows = []
    for model_name, analysis_data in sorted(detailed_analysis.items()):
        row = [model_name, analysis_data.get("Overall", {}).get('anls', 0)]
        
        for answer_type in valid_answer_types:
            score = analysis_data.get("Answer type", {}).get(answer_type, {}).get('anls', 0)
            row.append(score)
            
        for image_type in valid_image_types:
            score = analysis_data.get("Image type", {}).get(image_type, {}).get('anls', 0)
            row.append(score)
            
        rows.append(row)

    with open(filename, 'w', encoding='utf-8') as f:
        f.write("\t".join(header) + "\n")
        for row in rows:
            f.write("\t".join(map(str, row)) + "\n")


def main():
    results_dir = "results/multi"
    if not os.path.exists(results_dir):
        print(f"❌ Directory '{results_dir}' not found")
        return

    predict_files = [f for f in os.listdir(results_dir) if f.endswith(".json")]
    if not predict_files:
        print("❌ No prediction files found")
        return

    print(f"📁 Found {len(predict_files)} prediction files")
    
    detailed_analysis = {}
    results = {}

    for fname in tqdm(predict_files, desc="Processing files"):
        file_path = os.path.join(results_dir, fname)
        
        with open(file_path, "r", encoding="utf-8") as f:
            preds = json.load(f)

        category_scores = analyze_by_categories(preds)
        model_name = fname.replace('.json', '')
        detailed_analysis[model_name] = create_detailed_report(category_scores)
        
        overall_anls = detailed_analysis[model_name]['Overall']['anls']
        results[model_name] = {"anls": overall_anls}

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(preds, f, indent=2, ensure_ascii=False)

    print("💾 Saving results to results/multi folder...")
    os.makedirs(results_dir, exist_ok=True)
    
    with open(os.path.join(results_dir, "final_scores.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    with open(os.path.join(results_dir, "detailed_analysis.json"), "w", encoding="utf-8") as f:
        json.dump(detailed_analysis, f, indent=2, ensure_ascii=False)

    save_anls_report(detailed_analysis, os.path.join(results_dir, "detailed_analysis.txt"))

    print("✅ ANLS Results:")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"📊 Files saved in {results_dir}/: final_scores.json, detailed_analysis.json, detailed_analysis.txt")


if __name__ == "__main__":
    main()
