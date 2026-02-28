from __future__ import annotations

import streamlit as st
import numpy as np
import plotly.graph_objects as go

from risk_engine.semantic_search import search_clauses, embed_texts
from risk_engine.governance_graph import GovernanceGraph


# -----------------------------
# Page Setup
# -----------------------------

st.set_page_config(page_title="Graph Governance Engine v4", layout="wide")
st.title("AI Governance Lab — Graph Engine v4")

jur = st.selectbox("Jurisdiction", ["JP", "AU"])

use_case = st.text_area(
    "Describe the AI use case",
    "LLM-based chatbot processing personal data with risk of prompt injection."
)

run_btn = st.button("Run Governance Analysis", type="primary")

# -----------------------------
# Main Execution
# -----------------------------

if run_btn:

    # -----------------------------
    # Step 1: Clause Retrieval
    # -----------------------------

    matches = search_clauses(
        query=use_case,
        jurisdiction=jur,
        top_k=5,
    )

    if not matches:
        st.warning("No clauses retrieved. Check clauses.json.")
        st.stop()

    graph = GovernanceGraph()

    # -----------------------------
    # Define Risks
    # -----------------------------

    risk_descriptions = {
        "privacy_risk": "Risk of personal data misuse or unlawful processing",
        "security_risk": "Risk of prompt injection or adversarial attack",
        "hallucination_risk": "Risk of hallucinated or misleading output"
    }

    risk_embeddings = embed_texts(list(risk_descriptions.values()))

    for i, (rid, desc) in enumerate(risk_descriptions.items()):
        graph.add_risk(
            rid,
            desc,
            np.array(risk_embeddings[i], dtype=np.float32)
        )

    # -----------------------------
    # Controls
    # -----------------------------

    graph.add_control("data_minimisation", "Minimise personal data")
    graph.add_control("access_control", "Access control and logging")
    graph.add_control("human_review", "Human oversight")

    graph.link_risk_to_control("privacy_risk", "data_minimisation")
    graph.link_risk_to_control("privacy_risk", "access_control")
    graph.link_risk_to_control("security_risk", "access_control")
    graph.link_risk_to_control("hallucination_risk", "human_review")

    # -----------------------------
    # Clause Embeddings
    # -----------------------------

    clause_ids = []
    clause_embeddings = []

    for m in matches:
        graph.add_clause(m.clause.clause_id, m.clause.title)
        clause_ids.append(m.clause.clause_id)
        clause_embeddings.append(embed_texts([m.clause.text])[0])

    clause_embeddings = np.array(clause_embeddings)

    # -----------------------------
    # Similarity Matrix
    # -----------------------------

    risk_ids, sim_matrix = graph.compute_similarity_matrix(clause_embeddings)

    threshold = graph.dynamic_threshold(sim_matrix)

    activated, explanations = graph.activate_risks(
        clause_ids,
        clause_embeddings,
        threshold
    )

    activated_risks, activated_controls = graph.propagate()

    score, level = graph.weighted_risk_score(activated)

    # -----------------------------
    # Coverage Analysis
    # -----------------------------

    expected_controls = {"data_minimisation", "access_control", "human_review"}
    missing = expected_controls - activated_controls

    coverage_ratio = (
        len(activated_controls) /
        (len(activated_controls) + len(missing))
        if (len(activated_controls) + len(missing)) > 0 else 0
    )

    # -----------------------------
    # Layout
    # -----------------------------

    left_col, right_col = st.columns([1, 1])

    # -----------------------------
    # LEFT PANEL
    # -----------------------------

    with left_col:

        st.subheader("Clause Retrieval")

        for m in matches:
            st.markdown(
                f"- **{m.clause.clause_id}**: {m.clause.title} "
                f"(score: {m.score:.3f})"
            )

        st.subheader("Similarity Matrix")

        heatmap = go.Figure(
            data=go.Heatmap(
                z=sim_matrix,
                x=risk_ids,
                y=clause_ids,
                colorscale="Viridis"
            )
        )

        st.plotly_chart(heatmap, use_container_width=True)

        st.subheader("Activated Risks")
        st.write(list(activated_risks))

        st.subheader("Recommended Controls")
        st.write(list(activated_controls))

        st.subheader("Control Coverage")
        st.write(f"Coverage ratio: {coverage_ratio:.2f}")
        st.write(f"Missing controls: {list(missing)}")

        st.subheader("Weighted Risk Score")
        st.write(f"Score: {score:.2f}")
        st.write(f"Risk Level: {level}")

    # -----------------------------
    # RIGHT PANEL
    # -----------------------------

    with right_col:

        st.subheader("Executive Summary")

        summary_text = graph.generate_executive_summary(
            activated=activated,
            activated_controls=activated_controls,
            missing_controls=missing,
            score=score,
            level=level,
            threshold=threshold,
        )

        st.markdown(summary_text)

        st.subheader("Activation Details")

        st.write(f"Dynamic similarity threshold applied: {threshold:.3f}")

        for line in explanations:
            st.write(line)
