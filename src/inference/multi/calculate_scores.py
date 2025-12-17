"""Score calculation for multi-image VQA predictions."""

import argparse
import json
import os
from collections import defaultdict

from tqdm import tqdm

from src.common.metrics import compute_anls, calculate_averages as _calc_avg


def analyze_by_categories(preds: list[dict]) -> dict:
    """Analyze predictions by answer and image categories."""
    category_scores = {
        "Overall": {"total_anls": 0.0, "count": 0},
        "Answer type": defaultdict(lambda: {"total_anls": 0.0, "count": 0}),
        "Image type": defaultdict(lambda: {"total_anls": 0.0, "count": 0}),
    }

    for item in preds:
        if "answer" not in item or "predict" not in item:
            continue

        gt_clean = item["answer"].lower().strip()
        pr_clean = item["predict"].lower().strip().rstrip(".").replace('"', "")
        anls_score = compute_anls(gt_clean, pr_clean)
        item["anls"] = anls_score

        category_scores["Overall"]["total_anls"] += anls_score
        category_scores["Overall"]["count"] += 1

        answer_type = item.get("answer_source", "N/A")
        if isinstance(answer_type, list):
            answer_type = answer_type[0] if answer_type else "N/A"
        category_scores["Answer type"][str(answer_type).lower()]["total_anls"] += anls_score
        category_scores["Answer type"][str(answer_type).lower()]["count"] += 1

        image_type = item.get("image_type", "N/A")
        if isinstance(image_type, list):
            image_type = image_type[0] if image_type else "N/A"
        category_scores["Image type"][str(image_type).lower()]["total_anls"] += anls_score
        category_scores["Image type"][str(image_type).lower()]["count"] += 1

    return category_scores


def calculate_averages(stats: dict[str, float | int]) -> dict[str, float | int]:
    """Calculate average ANLS from totals."""
    return _calc_avg(stats)


def create_detailed_report(category_scores: dict) -> dict:
    """Create detailed analysis report from category scores."""
    report = {"Overall": calculate_averages(category_scores["Overall"])}
    for key, sub_cats in category_scores.items():
        if key == "Overall":
            continue
        report[key] = {
            sub: calculate_averages(stats) for sub, stats in sorted(sub_cats.items())
        }
    return report


def save_report_txt(detailed_analysis: dict, filename: str) -> None:
    """Save ANLS analysis to tab-separated file."""
    excluded = {"none", "[]", "n/a"}

    answer_types: set[str] = set()
    image_types: set[str] = set()

    for data in detailed_analysis.values():
        if "Answer type" in data:
            answer_types.update(data["Answer type"].keys())
        if "Image type" in data:
            image_types.update(data["Image type"].keys())

    answer_types = sorted(k for k in answer_types if k not in excluded)
    image_types = sorted(k for k in image_types if k not in excluded)

    header = (
        ["Method", "Overall"]
        + [t.replace("-", " ").title() for t in answer_types]
        + [t.replace("-", " ").title() for t in image_types]
    )

    rows = []
    for model, data in sorted(detailed_analysis.items()):
        row = [model, data.get("Overall", {}).get("anls", 0)]
        for at in answer_types:
            row.append(data.get("Answer type", {}).get(at, {}).get("anls", 0))
        for it in image_types:
            row.append(data.get("Image type", {}).get(it, {}).get("anls", 0))
        rows.append(row)

    with open(filename, "w", encoding="utf-8") as f:
        f.write("\t".join(str(h) for h in header) + "\n")
        for row in rows:
            f.write("\t".join(str(v) for v in row) + "\n")


def main(results_dir: str = "results/multi") -> None:
    """Run score calculation for multi-image predictions."""
    if not os.path.exists(results_dir):
        print(f"Directory not found: {results_dir}")
        return

    files = [
        f for f in os.listdir(results_dir)
        if f.endswith(".json") and "scores" not in f and "analysis" not in f
    ]
    if not files:
        print("No prediction files found")
        return

    detailed_analysis = {}
    results = {}

    for fname in tqdm(files, desc="Computing scores"):
        path = os.path.join(results_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            preds = json.load(f)

        category_scores = analyze_by_categories(preds)
        model_name = fname.replace(".json", "")
        detailed_analysis[model_name] = create_detailed_report(category_scores)
        results[model_name] = {"anls": detailed_analysis[model_name]["Overall"]["anls"]}

        with open(path, "w", encoding="utf-8") as f:
            json.dump(preds, f, indent=2, ensure_ascii=False)

    with open(os.path.join(results_dir, "final_scores.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    with open(os.path.join(results_dir, "detailed_analysis.json"), "w", encoding="utf-8") as f:
        json.dump(detailed_analysis, f, indent=2, ensure_ascii=False)

    save_report_txt(detailed_analysis, os.path.join(results_dir, "detailed_analysis.txt"))

    print("Results:")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate ANLS scores for multi-image VQA")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/multi",
        help="Directory with prediction files",
    )
    args = parser.parse_args()
    main(results_dir=args.results_dir)
