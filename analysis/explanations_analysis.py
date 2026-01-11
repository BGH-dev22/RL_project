from typing import List


def summarize_explanations(explanations: List[str]) -> str:
    positive = [e for e in explanations if "réussissent" in e]
    return f"Explications positives: {len(positive)}/{len(explanations)}"
