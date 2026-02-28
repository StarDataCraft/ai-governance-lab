from __future__ import annotations

import streamlit as st
import numpy as np

from risk_engine.semantic_search import search_clauses, embed_texts
from risk_engine.governance_graph import GovernanceGraph
from risk_engine.risk_model import compute_risk_score


st.set_page_config(page_title="Graph-based Governance Engine", layout="wide")
st.title("AI Governance Lab — Graph Engine v2 (Embedding Risk Mapping)")

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

    st.subheader("🔎 Step 1 — Clause Matching")

    matches = search_clauses(
        query=use_case,
        jurisdiction=jur,
        top_k=5,
    )

    if not matches:
        st.warning("No clauses found.")
        st.stop()

    for m in matches:
        st.markdown(
            f"- **{m.clause.clause_id}**: {m.clause.title} "
            f"(score: {m.score:.3f})"
        )

    st.subheader("🧠 Step 2 — Embedding-based Risk Mapping")

    graph = GovernanceGraph()

    # Define risks
    risk_descriptions = {
        "privacy_risk": "Risk of personal data misuse or data leakage",
        "security_risk": "Risk of prompt injection or adversarial manipulation",
        "hallucination_risk": "Risk of hallucinated or incorrect AI output"
    }

    # Embed risk descriptions
    risk_texts = list(risk_descriptions.values())
    risk_embeddings = embed_texts(risk_texts)

    for i, (risk_id, desc) in enumerate(risk_descriptions.items()):
        graph.add_risk(
            risk_id,
            desc,
            np.array(risk_embeddings[i], dtype=np.float32)
        )

    # Controls
    graph.add_control("data_minimisation", "Minimise personal data in prompts")
    graph.add_control("access_control", "Implement logging and access control")
    graph.add_control("human_review", "Require human oversight for high-impact outputs")

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
                f"Clause {cid} activated {risk_id} (similarity={sim:.3f})"
            )

    result = graph.propagate()

    st.markdown("### Activation Explanation")
    for line in explanation_log:
        st.write(line)

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
