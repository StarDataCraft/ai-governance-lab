from __future__ import annotations

import streamlit as st

from risk_engine.semantic_search import search_clauses
from risk_engine.governance_graph import GovernanceGraph
from risk_engine.risk_model import compute_risk_score


# -----------------------------
# UI Setup
# -----------------------------

st.set_page_config(page_title="Graph-based Governance Engine", layout="wide")
st.title("AI Governance Lab — Graph Engine v1")

jur = st.selectbox("Jurisdiction", ["JP", "AU"])

use_case = st.text_area(
    "Describe the AI use case",
    "LLM-based chatbot processing personal data with risk of prompt injection."
)

exposure = st.slider("Exposure level (1–3)", 1, 3, 2)
sensitivity = st.slider("Data sensitivity (1–3)", 1, 3, 2)
automation = st.slider("Automation level (1–3)", 1, 3, 2)

run_btn = st.button("Run Graph Governance Analysis", type="primary")


# -----------------------------
# Main Logic
# -----------------------------

if run_btn:

    st.subheader("🔎 Step 1 — Clause Matching")

    matches = search_clauses(
        query=use_case,
        jurisdiction=jur,
        top_k=5,
    )

    if not matches:
        st.warning("No clauses found. Check your clauses.json file.")
        st.stop()

    matched_clause_ids = [m.clause.clause_id for m in matches]

    for m in matches:
        st.markdown(
            f"- **{m.clause.clause_id}**: {m.clause.title} "
            f"(score: {m.score:.3f})"
        )

    # -----------------------------
    # Build Governance Graph
    # -----------------------------

    st.subheader("🧠 Step 2 — Graph Propagation")

    graph = GovernanceGraph()

    # Risk nodes
    graph.add_risk("privacy_risk", "Personal data misuse or leakage")
    graph.add_risk("security_risk", "Prompt injection or adversarial attack")
    graph.add_risk("hallucination_risk", "Incorrect or harmful AI output")

    # Control nodes
    graph.add_control("data_minimisation", "Minimise personal data in prompts")
    graph.add_control("access_control", "Implement logging and access control")
    graph.add_control("human_review", "Require human oversight for high-impact outputs")

    # Add clause nodes
    for m in matches:
        cid = m.clause.clause_id
        text_lower = m.clause.text.lower()

        graph.add_clause(cid, m.clause.title)

        # Simple semantic triggers
        if "personal" in text_lower or "data" in text_lower:
            graph.link_clause_to_risk(cid, "privacy_risk")

        if "injection" in text_lower or "adversarial" in text_lower:
            graph.link_clause_to_risk(cid, "security_risk")

        if "hallucination" in text_lower or "incorrect" in text_lower:
            graph.link_clause_to_risk(cid, "hallucination_risk")

    # Risk → Control links
    graph.link_risk_to_control("privacy_risk", "data_minimisation")
    graph.link_risk_to_control("privacy_risk", "access_control")

    graph.link_risk_to_control("security_risk", "access_control")

    graph.link_risk_to_control("hallucination_risk", "human_review")

    result = graph.propagate_from_clauses(matched_clause_ids)

    # -----------------------------
    # Risk Scoring
    # -----------------------------

    st.subheader("📊 Step 3 — Risk Scoring")

    risk_eval = compute_risk_score(
        result["risks"],
        exposure_level=exposure,
        data_sensitivity=sensitivity,
        automation_level=automation,
    )

    st.markdown("### Activated Risks")
    st.write(result["risks"] if result["risks"] else "None detected")

    st.markdown("### Recommended Controls")
    st.write(result["controls"] if result["controls"] else "None triggered")

    st.markdown("### Risk Evaluation")
    st.json(risk_eval)
