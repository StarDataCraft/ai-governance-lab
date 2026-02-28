from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Tuple

import yaml


@dataclass(frozen=True)
class RiskResult:
    primary_risk: str
    risk_scores: Dict[str, float]
    severity: str
    likelihood: str
    detectability: str
    rationale: Dict[str, Any]


class RiskClassifier:
    """
    Minimal, explainable, rule-based AI governance risk classifier.

    Design goals (v0):
    - Deterministic: same input => same output
    - Explainable: return rationale & matched keywords
    - Extensible: taxonomy stored in YAML
    """

    def __init__(self, taxonomy_path: str | Path | None = None):
        if taxonomy_path is None:
            taxonomy_path = Path(__file__).with_name("taxonomy.yaml")
        self.taxonomy_path = Path(taxonomy_path)
        self.taxonomy = self._load_taxonomy(self.taxonomy_path)

    @staticmethod
    def _load_taxonomy(path: Path) -> Dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"taxonomy file not found: {path}")
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or "risk_categories" not in data:
            raise ValueError("invalid taxonomy format: missing 'risk_categories'")
        return data

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(text.lower().split())

    def _score_categories(self, text: str) -> Tuple[Dict[str, float], Dict[str, List[str]]]:
        cfg = self.taxonomy.get("scoring", {})
        base = float(cfg.get("base_score", 0.10))
        hit = float(cfg.get("keyword_hit_score", 0.18))
        max_score = float(cfg.get("max_score", 1.0))

        scores: Dict[str, float] = {}
        matches: Dict[str, List[str]] = {}

        for key, cat in self.taxonomy["risk_categories"].items():
            kws = cat.get("keywords", []) or []
            matched = [kw for kw in kws if kw in text]
            matches[key] = matched

            s = base + hit * len(matched)
            scores[key] = min(max_score, round(s, 4))

        # Normalize to sum to 1.0 for nicer output (optional but pleasant).
        total = sum(scores.values())
        if total > 0:
            for k in list(scores.keys()):
                scores[k] = round(scores[k] / total, 4)

        return scores, matches

    def _level_from_rules(self, text: str, rules_key: str, default: str) -> Tuple[str, Dict[str, Any]]:
        rules = self.taxonomy.get(rules_key, {}) or {}

        hi = rules.get("high_if_contains_any", []) or []
        mid = rules.get("medium_if_contains_any", []) or []
        low = rules.get("low_if_contains_any", []) or []

        hit_hi = [x for x in hi if x in text]
        hit_mid = [x for x in mid if x in text]
        hit_low = [x for x in low if x in text]

        if hit_hi:
            level = "High"
        elif hit_mid:
            level = "Medium"
        elif hit_low:
            level = "Low"
        else:
            level = default

        rationale = {
            "rule_set": rules_key,
            "matched_high": hit_hi,
            "matched_medium": hit_mid,
            "matched_low": hit_low,
            "default": default,
        }
        return level, rationale

    def classify(self, description: str) -> RiskResult:
        """
        Classify a use case description into a structured risk output.

        Returns:
            RiskResult with:
              - primary_risk: highest scoring category key
              - risk_scores: normalized probability-like scores
              - severity/likelihood/detectability: Low/Medium/High labels
              - rationale: matched keywords and rules fired
        """
        if not isinstance(description, str) or not description.strip():
            raise ValueError("description must be a non-empty string")

        text = self._normalize(description)

        risk_scores, keyword_matches = self._score_categories(text)
        primary_risk = max(risk_scores.items(), key=lambda x: x[1])[0]

        severity, sev_r = self._level_from_rules(text, "severity_rules", default="Low")
        likelihood, lik_r = self._level_from_rules(text, "likelihood_rules", default="Medium")
        detectability, det_r = self._level_from_rules(text, "detectability_rules", default="Medium")

        rationale = {
            "primary_risk_reason": {
                "top_category": primary_risk,
                "keyword_matches": keyword_matches.get(primary_risk, []),
            },
            "keyword_matches_by_category": keyword_matches,
            "severity": sev_r,
            "likelihood": lik_r,
            "detectability": det_r,
            "notes": [
                "v0 rule-based baseline for governance discussions; tune taxonomy.yaml as you learn.",
                "scores are normalized for readability; not statistical probabilities.",
            ],
        }

        return RiskResult(
            primary_risk=primary_risk,
            risk_scores=risk_scores,
            severity=severity,
            likelihood=likelihood,
            detectability=detectability,
            rationale=rationale,
        )


def _cli() -> int:
    """
    Usage:
      python -m risk_engine.classifier "your use case description..."
    """
    import json
    import sys

    if len(sys.argv) < 2:
        print('Usage: python -m risk_engine.classifier "description..."')
        return 2

    desc = " ".join(sys.argv[1:])
    clf = RiskClassifier()
    result = clf.classify(desc)

    payload = {
        "primary_risk": result.primary_risk,
        "risk_scores": result.risk_scores,
        "severity": result.severity,
        "likelihood": result.likelihood,
        "detectability": result.detectability,
        "rationale": result.rationale,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
