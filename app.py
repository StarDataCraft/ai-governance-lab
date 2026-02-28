import json
import streamlit as st

from risk_engine import RiskClassifier

st.set_page_config(page_title="AI Governance Lab", layout="wide")

st.title("AI Governance Lab — Risk Classifier (v0)")
st.caption("Input a use case description. Get risk category scores + severity/likelihood/detectability + rationale.")

clf = RiskClassifier()

default_text = "Public LLM chatbot for customer support, facing prompt injection and data leakage risks, processing personal data under GDPR."
desc = st.text_area("Use case description", value=default_text, height=160)

col1, col2 = st.columns([1, 1])

if st.button("Assess Risk", type="primary"):
    try:
        r = clf.classify(desc)
    except Exception as e:
        st.error(str(e))
        st.stop()

    with col1:
        st.subheader("Summary")
        st.metric("Primary risk", r.primary_risk)
        st.write(
            {
                "severity": r.severity,
                "likelihood": r.likelihood,
                "detectability": r.detectability,
            }
        )

        st.subheader("Risk scores")
        # Streamlit can chart dict directly if converted
        st.bar_chart(r.risk_scores)

    with col2:
        st.subheader("Rationale (audit-friendly)")
        payload = {
            "primary_risk": r.primary_risk,
            "risk_scores": r.risk_scores,
            "severity": r.severity,
            "likelihood": r.likelihood,
            "detectability": r.detectability,
            "rationale": r.rationale,
        }
        st.code(json.dumps(payload, ensure_ascii=False, indent=2), language="json")

st.divider()
with st.expander("What to build next"):
    st.write(
        """
        - EU AI Act tier mapping (v0 table)
        - Control mapping (prevent/detect/correct) per risk
        - Upload a CSV of use cases and batch assess
        - Add a governance report generator (Markdown/PDF)
        """
    )
