from __future__ import annotations

import streamlit as st
import numpy as np
import plotly.graph_objects as go

from risk_engine.semantic_search import search_clauses, embed_texts
from risk_engine.governance_graph import GovernanceGraph


st.set_page_config(page_title="Graph Governance Engine v4", layout="wide")
st.title("AI Governance Lab — Graph Engine v4")

jur = st.selectbox("Jurisdiction", ["JP", "AU"])

use_case = st.text_area(
    "Describe the AI use case",
    "LLM-based chatbot processing personal data with risk of prompt injection."
)

run_btn = st.button("Run Governance Analysis", type="primary")

if run_btn:

    matches = search_clauses(
        query=use_case,
        jurisdiction=jur,
        top_k=5,
    )

    graph = GovernanceGraph()

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

    graph.add_control("data_minimisation", "Minimise personal data")
    graph.add_control("access_control", "Access control and logging")
    graph.add_control("human_review", "Human oversight")

    graph.link_risk_to_control("privacy_risk", "data_minimisation")
    graph.link_risk_to_control("privacy_risk", "access_control")
    graph.link_risk_to_control("security_risk", "access_control")
    graph.link_risk_to_control("hallucination_risk", "human_review")

    clause_ids = []
    clause_embeddings = []

    for m in matches:
        graph.add_clause(m.clause.clause_id, m.clause.title)
        clause_ids.append(m.clause.clause_id)
        clause_embeddings.append(embed_texts([m.clause.text])[0])

    clause_embeddings = np.array(clause_embeddings)

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
    # Similarity Heatmap
    # -----------------------------

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

    # -----------------------------
    # Explanation
    # -----------------------------

    st.subheader("Activation Explanation")

    st.write(f"Dynamic threshold used: {threshold:.3f}")

    for line in explanations:
        st.write(line)

    st.subheader("Activated Risks")
    st.write(list(activated_risks))

    st.subheader("Recommended Controls")
    st.write(list(activated_controls))

    # -----------------------------
    # Coverage Gap
    # -----------------------------

    expected_controls = {"data_minimisation", "access_control", "human_review"}
    missing = expected_controls - activated_controls

    coverage_ratio = len(activated_controls) / len(expected_controls)

    st.subheader("Control Coverage Analysis")

    st.write(f"Coverage ratio: {coverage_ratio:.2f}")
    st.write(f"Missing controls: {list(missing)}")

    # -----------------------------
    # Weighted Risk Score
    # -----------------------------

    st.subheader("Weighted Risk Score")

    st.write(f"Score: {score:.2f}")
    st.write(f"Risk Level: {level}")
