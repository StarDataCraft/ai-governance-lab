# app.py

from __future__ import annotations
import streamlit as st
from risk_engine.semantic_search import search_clauses
from risk_engine.governance_graph import GovernanceGraph
from risk_engine.risk_model import compute_risk_score


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

if run_btn:

    matches = search_clauses(
        query=use_case,
        jurisdiction=jur,
        cache_dir="data",
        top_k=5,
    )

    if not matches:
        st.warning("No embeddings found. Build clause embeddings first.")
        st.stop()

    matched_clause_ids = [m.clause.clause_id for m in matches]

    st.subheader("Matched Clauses")
    for m in matches:
        st.markdown(f"- {m.clause.clause_id}: {m.clause.title}")

    graph = GovernanceGraph()

    # Define risk nodes
    graph.add_risk("privacy_risk", "Personal data misuse or leakage")
    graph.add_risk("security_risk", "Prompt injection / adversarial attack")
    graph.add_risk("hallucination_risk", "Incorrect or harmful output")

    # Define control nodes
    graph.add_control("data_minimisation", "Minimise personal data in prompts")
    graph.add_control("access_control", "Access control and logging")
    graph.add_control("human_review", "Human oversight before high-impact decisions")

    # Add clause nodes + link
    for m in matches:
        cid = m.clause.clause_id
        graph.add_clause(cid, m.clause.title)

        # Simple mapping rules
        if "personal" in m.clause.text.lower():
            graph.link_clause_to_risk(cid, "privacy_risk")

        if "injection" in m.clause.text.lower():
            graph.link_clause_to_risk(cid, "security_risk")

        if "hallucination" in m.clause.text.lower():
            graph.link_clause_to_risk(cid, "hallucination_risk")

    # Link risks to controls
    graph.link_risk_to_control("privacy_risk", "data_minimisation")
    graph.link_risk_to_control("privacy_risk", "access_control")
    graph.link_risk_to_control("security_risk", "access_control")
    graph.link_risk_to_control("hallucination_risk", "human_review")

    result = graph.propagate_from_clauses(matched_clause_ids)

    risk_eval = compute_risk_score(
        result["risks"],
        exposure_level=exposure,
        data_sensitivity=sensitivity,
        automation_level=automation,
    )

    st.subheader("Activated Risks")
    st.write(result["risks"])

    st.subheader("Recommended Controls")
    st.write(result["controls"])

    st.subheader("Risk Score")
    st.json(risk_eval)
