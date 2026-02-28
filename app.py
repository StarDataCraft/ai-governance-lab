from __future__ import annotations

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import networkx as nx

from risk_engine.semantic_search import search_clauses, embed_texts
from risk_engine.governance_graph import GovernanceGraph
from risk_engine.risk_model import compute_risk_score


st.set_page_config(page_title="Graph Governance Engine v3", layout="wide")
st.title("AI Governance Lab — Graph Engine v3")

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
        top_k=5,
    )

    graph = GovernanceGraph()

    # Risks
    risk_descriptions = {
        "privacy_risk": "Risk of personal data misuse or data leakage",
        "security_risk": "Risk of prompt injection or adversarial manipulation",
        "hallucination_risk": "Risk of hallucinated or incorrect AI output"
    }

    risk_embeddings = embed_texts(list(risk_descriptions.values()))

    for i, (rid, desc) in enumerate(risk_descriptions.items()):
        graph.add_risk(
            rid,
            desc,
            np.array(risk_embeddings[i], dtype=np.float32)
        )

    # Controls
    graph.add_control("data_minimisation", "Minimise personal data")
    graph.add_control("access_control", "Access control and logging")
    graph.add_control("human_review", "Human oversight")

    graph.link_risk_to_control("privacy_risk", "data_minimisation")
    graph.link_risk_to_control("privacy_risk", "access_control")
    graph.link_risk_to_control("security_risk", "access_control")
    graph.link_risk_to_control("hallucination_risk", "human_review")

    explanation_log = []

    for m in matches:
        cid = m.clause.clause_id
        graph.add_clause(cid, m.clause.title)

        clause_emb = np.array(embed_texts([m.clause.text])[0], dtype=np.float32)
        activated = graph.map_clause_to_risks(cid, clause_emb)

        for risk_id, sim in activated:
            explanation_log.append(
                f"{cid} → {risk_id} (sim={sim:.3f})"
            )

    result = graph.propagate()

    # -----------------------------
    # Graph Visualization
    # -----------------------------

    st.subheader("Graph Visualization")

    pos = nx.spring_layout(graph.G, seed=42)

    edge_x = []
    edge_y = []

    for edge in graph.G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1),
        hoverinfo='none',
        mode='lines'
    )

    node_x = []
    node_y = []
    text = []

    for node in graph.G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        text.append(node)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=text,
        textposition="top center",
        marker=dict(size=15)
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # Centrality
    # -----------------------------

    st.subheader("Centrality Analysis")

    centrality = graph.compute_centrality()
    st.json(centrality)

    # -----------------------------
    # Risk Clustering
    # -----------------------------

    st.subheader("Risk Clustering")

    clusters = graph.cluster_risks(n_clusters=2)
    st.json(clusters)

    # -----------------------------
    # Risk Scoring
    # -----------------------------

    st.subheader("Risk Evaluation")

    risk_eval = compute_risk_score(
        result["risks"],
        exposure_level=exposure,
        data_sensitivity=sensitivity,
        automation_level=automation,
    )

    st.json(risk_eval)
