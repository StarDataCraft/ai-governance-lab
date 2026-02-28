import pytest

from risk_engine import RiskClassifier


def test_privacy_regulatory_signal():
    clf = RiskClassifier()
    desc = "We process personal data (PII) for customer onboarding and must comply with GDPR."
    r = clf.classify(desc)

    assert isinstance(r.primary_risk, str)
    assert "regulatory" in r.risk_scores
    assert r.severity in {"Low", "Medium", "High"}
    assert r.likelihood in {"Low", "Medium", "High"}
    assert r.detectability in {"Low", "Medium", "High"}
    assert isinstance(r.rationale, dict)


def test_security_signal():
    clf = RiskClassifier()
    desc = "Public LLM chatbot facing prompt injection and data leakage risks. Need monitoring and logging."
    r = clf.classify(desc)

    # Primary risk should often be security or operational depending on keyword hits
    assert r.primary_risk in {"security", "operational", "regulatory", "ethical", "strategic"}
    assert r.risk_scores["security"] > 0


def test_high_severity_keywords():
    clf = RiskClassifier()
    desc = "AI model used for credit loan approvals at scale with automated decisions."
    r = clf.classify(desc)
    assert r.severity == "High"
