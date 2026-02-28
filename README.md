# AI Governance Lab

Minimal, explainable tooling for AI governance prototyping.

## What this is (v0)
A rule-based **Risk Classification Engine**:
- reads a structured taxonomy from YAML
- outputs category scores + (severity, likelihood, detectability)
- returns rationale for auditability

## Quickstart
```bash
pip install -r requirements.txt
python -m risk_engine.classifier "Public LLM chatbot for customer support; risks include prompt injection and data leakage."
pytest -q
