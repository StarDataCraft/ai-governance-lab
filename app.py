# app.py
from __future__ import annotations

import datetime as dt
import html
import re
import textwrap
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote_plus
import xml.etree.ElementTree as ET

import requests
import streamlit as st
import yaml


# -------------------------
# Config
# -------------------------
SOURCES_YAML_PATH = "governance_sources.yaml"
MAX_SOURCES = 10
DEFAULT_NEWS_DAYS = 30
NEWS_MAX_ITEMS = 10

# Google News RSS locale parameters (hl/gl/ceid)
GOOGLE_NEWS_LOCALE = {
    "EU": ("en-GB", "GB", "GB:en"),  # closest practical default
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

    # Parse RSS XML
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
    Google News advanced query supports 'when:30d' etc.
    We'll bias for governance/legal/guideline terms.
    """
    hint = JURISDICTION_SEARCH_HINT.get(jur, "")
    # Keep query tight: we want governance + law + guideline + AI Act style terms
    base_terms = "AI governance OR AI regulation OR AI law OR guideline OR framework"
    time_term = f"when:{days}d"
    parts = [base_terms, hint, extra_terms.strip(), time_term]
    return " ".join([p for p in parts if p])


def strip_md(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def build_answer_template(
    jur: str,
    use_case: str,
    selected: List[SourceItem],
    cite: Dict[str, int],
) -> str:
    """
    A deterministic 'governance memo' style answer that:
    - references selected sources by citation number
    - stays concise and actionable
    You will evolve this later (e.g., map to controls, EU AI Act tiers, etc.).
    """
    # Identify key anchors by type/id heuristics
    # (Keep this simple and transparent; no hidden AI.)
    law_sources = [s for s in selected if s.type in {"law", "bill", "regulation"}]
    framework_sources = [s for s in selected if s.type in {"framework", "guideline", "policy", "principles", "guidance"}]

    bullets: List[str] = []

    if law_sources:
        # Mention legal anchor(s)
        refs = ", ".join(f"[{cite[s.id]}]" for s in law_sources[:2])
        bullets.append(f"**Legal baseline**: Start from the binding legal instrument(s) in this jurisdiction {refs} and confirm whether the described use case falls into regulated categories (e.g., high-risk / prohibited / disclosure duties).")

    if framework_sources:
        refs = ", ".join(f"[{cite[s.id]}]" for s in framework_sources[:2])
        bullets.append(f"**Operational governance baseline**: Use the leading national framework/guideline(s) to structure risk management, documentation, and oversight {refs}.")

    # Always include: risk mgmt, documentation, monitoring, human oversight
    # (Generic, but correct across major frameworks; user can refine per jurisdiction.)
    refs_all = ", ".join(f"[{cite[s.id]}]" for s in selected[: min(3, len(selected))])
    bullets.extend(
        [
            f"**Risk management**: Define risks, assess impact/likelihood, and track mitigations as a maintained process (not a one-off checklist) {refs_all}.",
            f"**Documentation & accountability**: Maintain clear technical and governance documentation (purpose, data sources, evaluation, change management, approvals) {refs_all}.",
            f"**Monitoring & incident response**: Implement monitoring for drift/abuse and have an incident response path (triage → rollback/fallback → reporting) {refs_all}.",
            f"**Human oversight**: Specify when humans must review/override, and ensure users/operators understand limitations {refs_all}.",
        ]
    )

    use_case_clean = strip_md(use_case)[:600]
    header = f"## Governance guidance (jurisdiction: {jur})\n\n**Use case**: {use_case_clean}\n"
    body = "\n".join([f"- {b}" for b in bullets])

    closing = "\n\n**References used**: " + ", ".join(f"[{cite[s.id]}]" for s in selected)

    return header + "\n\n" + body + closing


# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="AI Governance Lab", layout="wide")
st.title("AI Governance Lab — Jurisdiction-aware Governance Finder (v0)")
st.caption("Pick a jurisdiction → get top authoritative AI governance sources (≤10) + a cite-backed guidance memo + latest developments (RSS).")

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

    st.markdown("### Use case (optional, but recommended)")
    use_case = st.text_area(
        "Describe the AI system/use case briefly",
        value="LLM-based customer support chatbot processing personal data; risks include prompt injection, data leakage, and hallucinations.",
        height=140,
    )

    st.markdown("### Latest developments (optional)")
    enable_latest = st.checkbox("Fetch latest developments (news RSS)", value=True)
    days = st.slider("Recency window (days)", min_value=7, max_value=60, value=DEFAULT_NEWS_DAYS, step=1)
    extra_terms = st.text_input("Extra search terms (optional)", value="AI Act OR guideline OR framework OR compliance")

    run_btn = st.button("Generate (sources + memo + latest)", type="primary")

with right:
    st.markdown("### Output")
    if not run_btn:
        st.info("Configure options on the left, then click **Generate**.")
        st.stop()

    # Select top sources
    selected = select_top_sources(sources_db.get(jur, []), k=MAX_SOURCES)

    if include_intl and jur != "INTL" and "INTL" in sources_db:
        intl = select_top_sources(sources_db["INTL"], k=MAX_SOURCES)
        # Merge and re-trim
        merged = selected + [s for s in intl if s.id not in {x.id for x in selected}]
        selected = select_top_sources(merged, k=MAX_SOURCES)

    if not selected:
        st.warning("No sources configured for this jurisdiction in governance_sources.yaml.")
        st.stop()

    cite = build_citation_map(selected)

    # Show sources
    render_sources(selected, cite)

    st.divider()

    # Memo / answer with citations
    st.subheader("Cite-backed guidance memo (deterministic v0)")
    memo = build_answer_template(jur=jur, use_case=use_case, selected=selected, cite=cite)
    st.markdown(memo)

    st.divider()

    # Latest developments
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
            st.write("No items returned (or fetch blocked). Try changing the query/locale/recency window.")
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
