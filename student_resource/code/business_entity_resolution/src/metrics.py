"""Official macro F₀.₅ metric helpers."""
from __future__ import annotations

from typing import Dict, Iterable, List, Set


def f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    if precision == 0.0 and recall == 0.0:
        return 0.0
    b2 = beta * beta
    denom = b2 * precision + recall
    if denom == 0:
        return 0.0
    return (1 + b2) * precision * recall / denom


def score_entity(pred: Set[str], truth: Set[str]) -> float:
    """F₀.₅ for one Source-1 entity (singletons included)."""
    if not truth and not pred:
        return 1.0
    if not truth and pred:
        return 0.0
    if truth and not pred:
        return 0.0
    tp = len(pred & truth)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(truth) if truth else 0.0
    return f_beta(precision, recall, 0.5)


def macro_f05(
    predictions: Dict[str, Iterable[str]],
    ground_truth: Dict[str, Iterable[str]],
) -> float:
    scores: List[float] = []
    for s1, truth_ids in ground_truth.items():
        pred = set(predictions.get(s1, []))
        truth = set(x for x in truth_ids if x)
        scores.append(score_entity(pred, truth))
    return sum(scores) / len(scores) if scores else 0.0


def parse_id_list(cell: str) -> List[str]:
    if cell is None:
        return []
    text = str(cell).strip()
    if not text or text.lower() == "nan":
        return []
    # preserve order, drop dupes
    seen = set()
    out = []
    for part in text.split(","):
        pid = part.strip()
        if pid and pid not in seen:
            seen.add(pid)
            out.append(pid)
    return out
