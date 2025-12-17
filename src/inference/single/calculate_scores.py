"""Score calculation for VQA predictions using ANLS and Accuracy metrics."""

import argparse
import json
import os
from collections import defaultdict

from src.common.metrics import compute_anls, calculate_averages as _calc_avg, clean_prediction
from src.common.utils import extract_clean_filename


def analyze_by_categories(preds: list[dict]) -> dict:
    """
    Analyze predictions by answer categories.
    
    Args:
        preds: List of prediction dictionaries with 'answer' and 'predict' keys
        
    Returns:
        Category scores dictionary with ANLS and accuracy totals
    """
    category_scores = {
        "Overall": {"total_anls": 0.0, "total_accuracy": 0, "count": 0},
        "Answer type": defaultdict(lambda: {"total_anls": 0.0, "total_accuracy": 0, "count": 0}),
        "Element": defaultdict(lambda: {"total_anls": 0.0, "total_accuracy": 0, "count": 0}),
        "Operation": defaultdict(lambda: {"total_anls": 0.0, "total_accuracy": 0, "count": 0}),
    }
    key_mapping = {
        "answer_source": "Answer type",
        "element": "Element",
        "operation": "Operation",
    }

    for item in preds:
        if "answer" not in item or "predict" not in item:
            continue

        gt_clean = item["answer"].lower().strip()
        pr_clean = clean_prediction(item["predict"])

        anls_score = compute_anls(gt_clean, pr_clean)
        accuracy_score = int(gt_clean == pr_clean)

        item["anls"] = anls_score
        item["accuracy"] = accuracy_score

        overall = category_scores["Overall"]
        overall["total_anls"] += anls_score
        overall["total_accuracy"] += accuracy_score
        overall["count"] += 1

        for orig_key, mapped_key in key_mapping.items():
            value = item.get(orig_key)
            values = (
                value
                if isinstance(value, list) and value
                else [value] if value is not None else ["N/A"]
            )
            for sub_cat in values:
                stats = category_scores[mapped_key][str(sub_cat).lower()]
                stats["total_anls"] += anls_score
                stats["total_accuracy"] += accuracy_score
                stats["count"] += 1

    return category_scores


def calculate_averages(stats: dict[str, float | int]) -> dict[str, float | int]:
    """Calculate average percentages from totals."""
    return _calc_avg(stats)


def create_detailed_report(category_scores: dict) -> dict:
    """Create detailed report from category scores."""
    report = {"Overall": calculate_averages(category_scores["Overall"])}
    for category_key, sub_categories in category_scores.items():
        if category_key == "Overall":
            continue
        report[category_key] = {
            sub_cat: calculate_averages(stats)
            for sub_cat, stats in sorted(sub_categories.items())
        }
    return report


def save_analysis_to_txt(detailed_analysis: dict, filename: str, metric: str) -> None:
    """
    Save detailed analysis to tab-separated text file.
    
    Args:
        detailed_analysis: Analysis data by model
        filename: Output file path
        metric: Metric name ('accuracy' or 'anls')
    """
    main_categories = ["Answer type", "Element", "Operation"]
    excluded_keys = {"none", "[]", "n/a"}

    all_sub_categories: dict[str, set] = defaultdict(set)
    for analysis_data in detailed_analysis.values():
        for category in main_categories:
            if category in analysis_data:
                all_sub_categories[category].update(analysis_data[category].keys())

    column_structure: dict[str, list] = {}
    header = ["Method", "Overall"]
    for category in main_categories:
        valid_subs = sorted(
            k for k in all_sub_categories[category] if str(k) not in excluded_keys
        )
        column_structure[category] = valid_subs
        header.extend(sub.replace("-", " ").capitalize() for sub in valid_subs)

    rows = []
    for model_name, analysis_data in sorted(detailed_analysis.items()):
        row = [model_name, analysis_data.get("Overall", {}).get(metric, 0)]
        for category in main_categories:
            for sub_cat in column_structure[category]:
                score = analysis_data.get(category, {}).get(sub_cat, {}).get(metric, 0)
                row.append(score)
        rows.append(row)

    with open(filename, "w", encoding="utf-8") as f:
        f.write("\t".join(str(h) for h in header) + "\n")
        for row in rows:
            f.write("\t".join(str(v) for v in row) + "\n")


def load_prediction_files(results_dir: str = "results") -> dict[str, list]:
    """Load all prediction JSON files from results directory."""
    if not os.path.exists(results_dir):
        return {}
    
    files = sorted(
        f for f in os.listdir(results_dir)
        if f.endswith(".json") and "scores" not in f and "analysis" not in f
    )
    
    data = {}
    for fname in files:
        path = os.path.join(results_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            data[fname] = json.load(f)
    return data


def save_predictions(fname: str, data: list, results_dir: str = "results") -> None:
    """Save updated predictions to file."""
    path = os.path.join(results_dir, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def compute_metrics(file_data: dict[str, list]) -> tuple[dict, dict]:
    """
    Compute ANLS and accuracy metrics for all prediction files.
    
    Args:
        file_data: Dictionary mapping filenames to prediction lists
        
    Returns:
        Tuple of (results summary, detailed analysis)
    """
    results = {}
    detailed_analysis = {}

    for fname, preds in file_data.items():
        key = extract_clean_filename(fname)
        category_scores = analyze_by_categories(preds)
        detailed_analysis[key] = create_detailed_report(category_scores)

        overall = detailed_analysis[key]["Overall"]
        results[key] = {"accuracy": overall["accuracy"], "anls": overall["anls"]}

        save_predictions(fname, preds)

    return results, detailed_analysis


def main(results_dir: str = "results") -> None:
    """Run score calculation pipeline."""
    file_data = load_prediction_files(results_dir)
    if not file_data:
        print(f"No prediction files found in '{results_dir}'")
        return

    print(f"Processing {len(file_data)} prediction files...")
    results, detailed_analysis = compute_metrics(file_data)

    os.makedirs(results_dir, exist_ok=True)

    scores_path = os.path.join(results_dir, "final_scores.json")
    with open(scores_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    analysis_path = os.path.join(results_dir, "detailed_analysis.json")
    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(detailed_analysis, f, indent=2, ensure_ascii=False)

    print("Results:")
    print(json.dumps(results, indent=2, ensure_ascii=False))

    save_analysis_to_txt(
        detailed_analysis,
        os.path.join(results_dir, "detailed_analysis_accuracy.txt"),
        "accuracy",
    )
    save_analysis_to_txt(
        detailed_analysis,
        os.path.join(results_dir, "detailed_analysis_anls.txt"),
        "anls",
    )

    print(f"Saved: {scores_path}, {analysis_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate ANLS and Accuracy scores")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory containing prediction JSON files",
    )
    args = parser.parse_args()
    main(results_dir=args.results_dir)
