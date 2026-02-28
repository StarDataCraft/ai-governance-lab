# risk_engine/risk_model.py

from __future__ import annotations
from typing import List, Dict


def compute_risk_score(
    risks: List[str],
    exposure_level: int = 1,
    data_sensitivity: int = 1,
    automation_level: int = 1,
) -> Dict:

    base = len(risks)

    score = (
        base * 2
        + exposure_level * 2
        + data_sensitivity * 3
        + automation_level * 2
    )

    level = "LOW"
    if score > 10:
        level = "MEDIUM"
    if score > 18:
        level = "HIGH"

    return {
        "risk_score": score,
        "risk_level": level,
    }
