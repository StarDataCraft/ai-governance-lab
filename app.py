# app.py
from __future__ import annotations

import html
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests
import streamlit as st
import yaml
from urllib.parse import quote_plus

from risk_engine.semantic_search import search_clauses


# -------------------------
# Config
# -------------------------
SOURCES_YAML_PATH = "governance_sources.yaml"
MAX_SOURCES = 10
DEFAULT_NEWS_DAYS = 30
NEWS_MAX_ITEMS = 10

# Google News RSS locale parameters (hl/gl/ceid)
GOOGLE_NEWS_LOCALE = {
    "EU": ("en-GB", "GB", "GB:en"),  # practical default
    "US": ("en-US", "US", "US:en"),
    "UK": ("en-GB", "GB", "GB:en"),
    "JP": ("ja", "JP", "JP:ja"),
    "CA": ("en-CA", "CA", "CA:en"),
    "SG": ("en-SG", "SG", "SG:en"),
    "AU": ("en-AU", "AU", "AU:en"),
    "INTL": ("en-US", "US", "US:en"),
}

JURISDICTION_SEARCH_HINT = {
    "EU": "European Union OR EU",
    "US": "United States OR US",
    "UK": "United Kingdom OR UK",
    "JP": "Japan OR 日本",
    "CA": "Canada",
    "SG": "Singapore",
    "AU": "Australia",
    "INTL": "",
}


# -------------------------
# Helpers
# -------------------------
@dataclass
class SourceItem:
    id: str
    title: str
    authority: str
    type: str
    url: str
    priority: int


@dataclass
class NewsItem:
    title: str
    link: str
    pub_date: Optional[str]
    source: Optional[str]


def load_sources_db(path: str) -> Dict[str, List[SourceItem]]:
    raw = yaml.safe_load(open(path, "r", encoding="utf-8"))
    if not isinstance(raw, dict) or "jurisdictions" not in raw:
        raise ValueError("Invalid governance_sources.yaml: missing 'jurisdictions'")

    db: Dict[str, List[SourceItem]] = {}
    for jur, items in raw["jurisdictions"].items():
        out: List[SourceItem] = []
        for it in items:
            out.append(
                SourceItem(
                    id=str(it["id"]),
                    title=str(it["title"]),
                    authority=str(it["authority"]),
                    type=str(it.get("type", "unknown")),
                    url=str(it["url"]),
                    priority=int(it.get("priority", 0)),
                )
            )
        db[str(jur)] = out
    return db


def select_top_sources(items: List[SourceItem], k: int = MAX_SOURCES) -> List[SourceItem]:
    return sorted(items, key=lambda x: (-x.priority, x.title))[:k]


def build_citation_map(selected: List[SourceItem]) -> Dict[str, int]:
    """Map source_id -> citation number starting at 1, in display order."""
    return {it.id: i + 1 for i, it in enumerate(selected)}


def render_sources(selected: List[SourceItem], cite: Dict[str, int]) -> None:
    st.subheader(f"Top authoritative sources (≤ {len(selected)})")
    for it in selected:
        num = cite[it.id]
        st.markdown(
            f"**[{num}] {it.title}**  \n"
            f"- Authority: {it.authority}  \n"
            f"- Type: `{it.type}`  \n"
            f"- Link: {it.url}"
        )


def google_news_rss_url(query: str, jur: str) -> str:
    hl, gl, ceid = GOOGLE_NEWS_LOCALE.get(jur, ("en-US", "US", "US:en"))
    q = quote_plus(query)
    return f"https://news.google.com/rss/search?q={q}&hl={hl}&gl={gl}&ceid={ceid}"


def fetch_rss_items(url: str, timeout: int = 10) -> List[NewsItem]:
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()

    root = ET.fromstring(r.text)
    channel = root.find("channel")
    if channel is None:
        return []

    items: List[NewsItem] = []
    for item in channel.findall("item"):
        title = item.findtext("title") or ""
        link = item.findtext("link") or ""
        pub = item.findtext("pubDate")
        src_el = item.find("source")
        src = src_el.text if src_el is not None else None

        title = html.unescape(title).strip()
        link = html.unescape(link).strip()

        items.append(NewsItem(title=title, link=link, pub_date=pub, source=src))
    return items


def make_latest_query(jur: str, days: int, extra_terms: str) -> str:
    """
    Google News supports `when:30d` etc.
    Keep query tight: governance/legal/guideline terms + jurisdiction hint.
    """
    hint = JURISDICTION_SEARCH_HINT.get(jur, "")
    base_terms = "AI governance OR AI regulation OR AI law OR guideline OR framework"
    time_term = f"when:{days}d"
    parts = [base_terms, hint, (extra_terms or "").strip(), time_term]
    return " ".join([p for p in parts if p])


def strip_md(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def build_answer_template(
    jur: str,
    use_case: str,
    selected: List[SourceItem],
    cite: Dict[str, int],
) -> str:
    """
    Deterministic governance memo that references selected sources by citation number.
    Enhanced with jurisdiction-specific bullets (AU focus) without using an LLM.
    """
    selected_ids = {s.id for s in selected}

    law_sources = [s for s in selected if s.type in {"law", "bill", "regulation"}]
    framework_sources = [
        s for s in selected if s.type in {"framework", "guideline", "policy", "principles", "guidance", "regulator_guidance"}
    ]

    bullets: List[str] = []

    # Baseline bullets (general)
    if law_sources:
        refs = ", ".join(f"[{cite[s.id]}]" for s in law_sources[:2])
        bullets.append(
            f"**Legal baseline**: Start from the binding instrument(s) in this jurisdiction {refs} "
            f"and confirm whether the use case triggers regulated categories (e.g., high-risk / prohibited / disclosure duties)."
        )

    if framework_sources:
        refs = ", ".join(f"[{cite[s.id]}]" for s in framework_sources[:2])
        bullets.append(
            f"**Operational governance baseline**: Use the leading national framework/guideline(s) to structure "
            f"risk management, documentation, and oversight {refs}."
        )

    # -------------------------
    # AU-specific augmentation
    # -------------------------
    if jur == "AU":
        # Detect OAIC guidance presence (by id prefix or known ids)
        oaic_ids = {
            "oaic_ai_products_guidance",
            "oaic_genai_training_guidance",
        }
        has_oaic = any(x in selected_ids for x in oaic_ids) or any(s.id.startswith("oaic_") for s in selected)

        # Detect government policy / assurance framework
        gov_policy_ids = {
            "au_ai_in_gov_policy",
            "au_ai_assurance_framework_pdf",
            "au_ai_assurance_framework_page",
            "au_ai_impact_assessment_privacy_security",
        }
        has_gov_policy = any(x in selected_ids for x in gov_policy_ids) or any(
            s.id.startswith("au_ai_in_gov_policy") or s.id.startswith("au_ai_assurance_framework") for s in selected
        )

        if has_oaic:
            refs = []
            for sid in ["oaic_ai_products_guidance", "oaic_genai_training_guidance"]:
                if sid in cite:
                    refs.append(f"[{cite[sid]}]")
            ref_str = ", ".join(refs) if refs else ", ".join(f"[{cite[s.id]}]" for s in selected[: min(3, len(selected))])

            bullets.append(
                f"**AU privacy boundary for LLM chatbots**: Treat prompts, logs, and outputs as potentially containing personal information. "
                f"Apply data minimisation, purpose limitation, and retention controls; avoid feeding sensitive or unnecessary identifiers into prompts {ref_str}."
            )
            bullets.append(
                f"**Vendor / cross-border & product due diligence**: Assess whether the AI service retains prompts, uses them for training, where data is stored/processed, "
                f"and what contractual/operational safeguards exist (incl. cross-border data flows and access controls) {ref_str}."
            )
            bullets.append(
                f"**Logging, prompt governance & leakage prevention**: Implement redaction, access control, and logging policies that balance auditability with privacy. "
                f"Define prompt templates/guardrails and incident playbooks for suspected leakage or misuse {ref_str}."
            )

        if has_gov_policy:
            # Prefer exact refs if present
            refs = []
            for sid in [
                "au_ai_in_gov_policy",
                "au_ai_assurance_framework_pdf",
                "au_ai_assurance_framework_page",
                "au_ai_impact_assessment_privacy_security",
            ]:
                if sid in cite:
                    refs.append(f"[{cite[sid]}]")
            ref_str = ", ".join(refs) if refs else ", ".join(f"[{cite[s.id]}]" for s in selected[: min(3, len(selected))])

            bullets.append(
                f"**Impact assessment & approval gates**: Before deployment, run an impact assessment (privacy/security, user impact, failure modes) and define approval thresholds "
                f"for high-impact changes (model updates, new data sources, new user groups) {ref_str}."
            )
            bullets.append(
                f"**Assurance practices (evidence-based)**: Maintain evidence for governance decisions—risk register, testing results, monitoring metrics, "
                f"change logs, and accountability assignments—to support internal/external assurance expectations {ref_str}."
            )
            bullets.append(
                f"**Incident pathway & escalation**: Define triage, rollback/fallback, user communications, and reporting steps for LLM incidents (harmful output, leakage, "
                f"security exploitation), and test the pathway periodically {ref_str}."
            )

    # -------------------------
    # General operational bullets (still included)
    # -------------------------
    refs_all = ", ".join(f"[{cite[s.id]}]" for s in selected[: min(3, len(selected))])
    bullets.extend(
        [
            f"**Risk management**: Define risks, assess impact/likelihood, and track mitigations as a maintained process {refs_all}.",
            f"**Documentation & accountability**: Maintain technical + governance documentation (purpose, data, eval, change mgmt, approvals) {refs_all}.",
            f"**Monitoring & incident response**: Monitor for drift/abuse and have an incident path (triage → rollback/fallback → reporting) {refs_all}.",
            f"**Human oversight**: Specify when humans must review/override, and ensure operators understand limitations {refs_all}.",
        ]
    )

    use_case_clean = strip_md(use_case)[:800]
    header = f"## Governance guidance (jurisdiction: {jur})\n\n**Use case**: {use_case_clean}\n"
    body = "\n".join([f"- {b}" for b in bullets])
    closing = "\n\n**References used**: " + ", ".join(f"[{cite[s.id]}]" for s in selected)
    return header + "\n\n" + body + closing


# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="AI Governance Lab", layout="wide")
st.title("AI Governance Lab — Jurisdiction-aware Governance Finder (v0)")
st.caption(
    "Pick a jurisdiction → get top authoritative AI governance sources (≤10) + a cite-backed memo + semantic evidence retrieval + latest developments (RSS)."
)

# Load sources DB
try:
    sources_db = load_sources_db(SOURCES_YAML_PATH)
except Exception as e:
    st.error(f"Failed to load {SOURCES_YAML_PATH}: {e}")
    st.stop()

jurisdictions = sorted(list(sources_db.keys()))
if not jurisdictions:
    st.error("No jurisdictions found in governance_sources.yaml")
    st.stop()

left, right = st.columns([0.45, 0.55])

with left:
    jur = st.selectbox("Jurisdiction", jurisdictions, index=0)
    include_intl = st.checkbox("Include INTL (OECD etc.) as supplements", value=True)

    st.markdown("### Use case")
    use_case = st.text_area(
        "Describe the AI system/use case briefly",
        value="LLM-based customer support chatbot processing personal data; risks include prompt injection, data leakage, and hallucinations.",
        height=140,
    )

    st.markdown("### Semantic evidence (zero-shot embeddings)")
    enable_semantic = st.checkbox("Show semantic evidence matches from your local clause corpus", value=True)
    top_k = st.slider("Top-K clauses", min_value=3, max_value=12, value=5, step=1)
    embed_model = st.selectbox(
        "Local embedding model",
        [
            "intfloat/multilingual-e5-small",
            "sentence-transformers/all-MiniLM-L6-v2",
            "BAAI/bge-m3",
        ],
        index=0,
    )

    st.markdown("### Latest developments (optional)")
    enable_latest = st.checkbox("Fetch latest developments (news RSS)", value=True)
    days = st.slider("Recency window (days)", min_value=7, max_value=60, value=DEFAULT_NEWS_DAYS, step=1)
    extra_terms = st.text_input("Extra search terms (optional)", value="AI Act OR guideline OR framework OR compliance")

    run_btn = st.button("Generate (sources + memo + evidence + latest)", type="primary")

with right:
    st.markdown("### Output")
    if not run_btn:
        st.info("Configure options on the left, then click **Generate**.")
        st.stop()

    # Select top sources
    selected = select_top_sources(sources_db.get(jur, []), k=MAX_SOURCES)

    if include_intl and jur != "INTL" and "INTL" in sources_db:
        intl = select_top_sources(sources_db["INTL"], k=MAX_SOURCES)
        merged = selected + [s for s in intl if s.id not in {x.id for x in selected}]
        selected = select_top_sources(merged, k=MAX_SOURCES)

    if not selected:
        st.warning("No sources configured for this jurisdiction in governance_sources.yaml.")
        st.stop()

    cite = build_citation_map(selected)

    # Show sources
    render_sources(selected, cite)

    st.divider()

    # Memo / answer with citations (enhanced with AU-specific bullets)
    st.subheader("Cite-backed guidance memo (deterministic v0)")
    memo = build_answer_template(jur=jur, use_case=use_case, selected=selected, cite=cite)
    st.markdown(memo)

    st.divider()

    # Semantic evidence retrieval (local embeddings)
    if enable_semantic:
        st.subheader("Semantic Evidence (law/guideline clause matches) — zero-shot local embeddings")
        st.caption(
            "Requires local clause corpus + cached embeddings. "
            "Build cache via: `python scripts/build_embeddings.py --jur <JUR>`"
        )

        try:
            matches = search_clauses(
                query=use_case,
                jurisdiction=jur,
                cache_dir="data",
                top_k=top_k,
                model=embed_model,
            )
        except FileNotFoundError as e:
            st.warning(str(e))
            st.info(
                "Fix:\n"
                "1) Add clauses at `law_corpus/<JUR>/clauses.jsonl`\n"
                "2) Run: `python scripts/build_embeddings.py --jur <JUR>`"
            )
            matches = []
        except Exception as e:
            st.error(f"Semantic search error: {e}")
            matches = []

        if matches:
            for rank, m in enumerate(matches, start=1):
                cite_num = cite.get(m.clause.source_id)
                cite_tag = f"[{cite_num}]" if cite_num else "[?]"

                st.markdown(f"### {rank}. {m.clause.title} — score: `{m.score:.3f}` {cite_tag}")
                st.markdown(f"- Source: `{m.clause.source_id}` {cite_tag}")
                if m.clause.url:
                    st.markdown(f"- Link: {m.clause.url}")
                st.write(m.clause.text)
            st.caption("Tip: keep clause chunks short (~200–800 chars). Add more clauses to improve retrieval quality.")
        else:
            st.write("No matches found (or cache missing).")

        st.divider()

    # Latest developments (RSS)
    if enable_latest:
        st.subheader(f"Latest developments (Google News RSS, last {days} days, ≤{NEWS_MAX_ITEMS})")

        query = make_latest_query(jur=jur, days=days, extra_terms=extra_terms)
        rss_url = google_news_rss_url(query=query, jur=jur)

        st.caption(f"RSS query: `{query}`")
        st.caption(f"RSS URL: {rss_url}")

        try:
            news_items = fetch_rss_items(rss_url)
        except Exception as e:
            st.warning(f"Failed to fetch RSS: {e}")
            news_items = []

        if not news_items:
            st.write("No items returned (or fetch blocked). Try changing query/locale/recency window.")
        else:
            for i, it in enumerate(news_items[:NEWS_MAX_ITEMS], start=1):
                meta = []
                if it.pub_date:
                    meta.append(it.pub_date)
                if it.source:
                    meta.append(it.source)
                meta_str = " · ".join(meta) if meta else ""
                st.markdown(f"**{i}. [{it.title}]({it.link})**  \n{meta_str}")

        st.divider()

    # References section (explicit, numbered)
    st.subheader("References")
    for it in selected:
        num = cite[it.id]
        st.markdown(f"[{num}] **{it.title}** — {it.authority} — {it.url}")
