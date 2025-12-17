"""Shared metrics computation for VQA evaluation."""

import Levenshtein


def compute_anls(gt: str, predict: str, threshold: float = 0.5) -> float:
    """
    Compute Average Normalized Levenshtein Similarity score.

    Args:
        gt: Ground truth answer
        predict: Predicted answer
        threshold: Minimum similarity threshold (default 0.5)

    Returns:
        ANLS score between 0.0 and 1.0
    """
    p = predict.replace('"', "").rstrip(".").lower()
    g = gt.lower()
    score = Levenshtein.ratio(p, g)
    return score if score >= threshold else 0.0


def calculate_averages(stats: dict[str, float | int]) -> dict[str, float | int]:
    """
    Calculate average percentages from totals.

    Args:
        stats: Dictionary with 'total_anls', 'count', and optionally 'total_accuracy'

    Returns:
        Dictionary with 'anls', 'count', and optionally 'accuracy' as percentages
    """
    count = stats.get("count", 0)
    if count == 0:
        result: dict[str, float | int] = {"anls": 0.0, "count": 0}
        if "total_accuracy" in stats:
            result["accuracy"] = 0.0
        return result

    result = {
        "anls": round((stats["total_anls"] / count) * 100, 2),
        "count": count,
    }
    if "total_accuracy" in stats:
        result["accuracy"] = round((stats["total_accuracy"] / count) * 100, 2)
    return result


def clean_prediction(text: str) -> str:
    """
    Normalize prediction text for comparison.

    Args:
        text: Raw prediction text

    Returns:
        Cleaned text for scoring
    """
    return (
        text.lower()
        .strip()
        .rstrip(".")
        .replace('"', "")
        .rstrip(">")
        .lstrip("<")
    )
