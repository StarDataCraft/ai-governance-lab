# app.py
# AI Governance Lab — Jurisdiction-aware Explainability Finder (Graph-based, no-key, OSS)
# - Clause semantic retrieval (local embeddings, cached)
# - Embedding-based risk mapping (clause -> risk)
# - Graph propagation (risk -> controls)
# - Similarity matrix visualization + dynamic threshold
# - Weighted risk score + coverage gap analysis
# - Graph visualization + centrality + risk clustering
# - Executive Explainability Narrative / Decision Trace + Explainability Score
#
# Data layout (recommended):
#   law_corpus/
#     JP/
#       clauses.json   # JSON array of clause objects
#     AU/
#       clauses.json
#     ...
#
# Clause object schema (minimum):
#   {
#     "clause_id": "JP-001",
#     "jurisdiction": "JP",
#     "framework": "...",
#     "domain": "...",
#     "risk_tags": ["privacy_risk", ...],   # optional
#     "title": "...",
#     "text": "...",
#     "url": "https://..."
#   }
#
# This app is designed to "sell organizational explainability", not legal compliance.

from __future__ import annotations

import os
import re
import json
import math
import time
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
import streamlit as st

# Lightweight deps (ensure in requirements.txt on Streamlit Cloud):
# sentence-transformers, torch (transitive), networkx, matplotlib, feedparser
from sentence_transformers import SentenceTransformer
import networkx as nx
import matplotlib.pyplot as plt
import feedparser
from risk_engine.i18n import t
from risk_engine.graph_viz import draw_governance_graph

# =========================
# Benchmark loader
# =========================

def load_benchmark_cases(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        return []
    return [x for x in data if isinstance(x, dict)]

def get_benchmark_options(
    benchmark_cases: List[Dict[str, Any]],
    jurisdiction: str,
) -> List[Dict[str, Any]]:
    filtered = []
    for case in benchmark_cases:
        case_jur = str(case.get("jurisdiction", "")).strip().upper()
        if case_jur == jurisdiction.upper():
            filtered.append(case)
    return filtered
# =========================
# Explainability Layer (v0)
# =========================

@dataclass
class ExplainabilityScore:
    total: float
    components: Dict[str, float]
    label: str
    narrative: str

def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))

def _label_from_score(s: float) -> str:
    if s >= 0.80:
        return "EXCELLENT"
    if s >= 0.65:
        return "GOOD"
    if s >= 0.45:
        return "OK"
    return "WEAK"

def compute_explainability_score(
    clause_matches: List[Dict[str, Any]],
    activation_details: List[Dict[str, Any]],
    activated_risks: List[str],
    recommended_controls: List[str],
    coverage_ratio: float,
    dynamic_threshold: Optional[float],
) -> ExplainabilityScore:
    """
    Score what you can explain to an executive:
    - evidence_trace: matched clauses have URLs + stable IDs
    - reasoning_transparency: show thresholds + activation explanations
    - control_coverage: how well controls cover activated risks
    - monitoring_readiness: presence of monitoring/incident/human oversight controls (heuristic)
    """
    matched = clause_matches or []

    # 1) Evidence trace completeness
    url_ok = sum(1 for m in matched if m.get("url"))
    id_ok = sum(1 for m in matched if m.get("clause_id"))
    denom = max(1, len(matched))
    evidence_trace = _clamp01(0.5 * (url_ok / denom) + 0.5 * (id_ok / denom))
    evidence_trace = _clamp01(evidence_trace * (1.0 if len(matched) >= 2 else 0.6))

    # 2) Reasoning transparency
    transparency = 0.0
    if dynamic_threshold is not None and isinstance(dynamic_threshold, (int, float)):
        transparency += 0.4
    # at least as many activation explanations as matched clauses (roughly)
    if activation_details and len(activation_details) >= max(1, len(matched)):
        transparency += 0.6
    reasoning_transparency = _clamp01(transparency)

    # 3) Control coverage
    control_coverage = _clamp01(coverage_ratio if coverage_ratio is not None else 0.0)

    # 4) Monitoring readiness (cheap heuristic)
    ctrl = set(recommended_controls or [])
    keywords = ["monitor", "logging", "incident", "human", "review", "fallback", "eval"]
    hits = sum(1 for k in keywords if any(k in c for c in ctrl))
    monitoring_readiness = _clamp01(hits / 4.0)  # saturate quickly

    components = {
        "evidence_trace": evidence_trace,
        "reasoning_transparency": reasoning_transparency,
        "control_coverage": control_coverage,
        "monitoring_readiness": monitoring_readiness,
    }

    total = (
        0.35 * evidence_trace +
        0.30 * reasoning_transparency +
        0.25 * control_coverage +
        0.10 * monitoring_readiness
    )

    label = _label_from_score(total)

    thr_s = f"{dynamic_threshold:.3f}" if isinstance(dynamic_threshold, (int, float)) else "n/a"
    narrative = (
        f"This output is designed to be *organizationally explainable* rather than merely 'compliant'. "
        f"Evidence trace is {evidence_trace:.2f} (matched clauses with URLs/IDs). "
        f"Reasoning transparency is {reasoning_transparency:.2f} (dynamic threshold={thr_s} + activation explanations). "
        f"Control coverage is {control_coverage:.2f} (how many activated risks have mapped controls). "
        f"Monitoring readiness is {monitoring_readiness:.2f} (presence of monitoring/incident/human oversight controls). "
        f"Overall Explainability Score = {total:.2f} ({label})."
    )

    return ExplainabilityScore(total=total, components=components, label=label, narrative=narrative)


def build_explainability_brief(
    lang: str,
    jurisdiction: str,
    use_case: str,
    clause_matches: List[Dict[str, Any]],
    activation_details: List[Dict[str, Any]],
    activated_risks: List[str],
    recommended_controls: List[str],
    weighted_risk_score: float,
    risk_level: str,
    dynamic_threshold: Optional[float],
    coverage_ratio: float,
    missing_controls: List[str],
) -> str:
    """
    Executive-friendly narrative that stitches evidence -> reasoning -> actions.
    Deterministic, no LLM.
    """
    if lang == "ja":
        heading = f"### 説明ナラティブ / 意思決定トレース（対象: {jurisdiction}）"
        use_case_label = "**ユースケース**"
        evidence_label = "**1) 証拠チェーン（何を根拠にしたか）**"
        reasoning_label = "**2) 推論の透明性（なぜそのリスクが発火したか）**"
        posture_label = "**3) リスクの意味（運用上どう解釈するか）**"
        action_label = "**4) コントロールとギャップ（次に何をするか）**"
        interp_label = "**解釈（このツールが本当に売っているもの）**"
        legal_note = "*(これは再現可能な意思決定トレースであり、法的意見ではありません。)*"
        no_evidence = "- まだ証拠が一致していません（条項を追加するか、検索条件を調整してください）。"
        no_activation = "- 発火した詳細はありません。"
        interpretation = (
            "このツールは単なるコンプライアンス・チェックリストではありません。"
            "組織内の承認、監査対応、インシデント振り返り、経営向け説明に再利用できる"
            "『再現可能な意思決定トレース』を提供します。"
        )
    else:
        heading = f"### Explainability Narrative / Decision Trace (Jurisdiction: {jurisdiction})"
        use_case_label = "**Use case**"
        evidence_label = "**1) Evidence chain (what we relied on)**"
        reasoning_label = "**2) Transparent reasoning (why risks activated)**"
        posture_label = "**3) Risk posture (what this means operationally)**"
        action_label = "**4) Controls + coverage gaps (what to do next)**"
        interp_label = "**Interpretation (what this tool really sells)**"
        legal_note = "*(This is a reproducible decision trace, not a legal opinion.)*"
        no_evidence = "- No evidence matched yet (add more clauses or adjust retrieval)."
        no_activation = "- No activation details available."
        interpretation = (
            "This tool is not merely a compliance checklist. It produces an "
            "*organizationally explainable decision trace* that can be reused for internal approvals, "
            "audit readiness, incident retrospectives, and governance communication."
        )

    top_evidence = clause_matches[:5] if clause_matches else []

    evidence_lines = []
    for i, m in enumerate(top_evidence, start=1):
        title = m.get("title", "Evidence")
        cid = m.get("clause_id", "?")
        sim = m.get("score", m.get("similarity", None))
        sim_s = f"{sim:.3f}" if isinstance(sim, (int, float)) else "n/a"
        url = m.get("url", "")
        fw = m.get("framework", "")
        fw_s = f" · {fw}" if fw else ""
        evidence_lines.append(f"[E{i}] {cid} — {title}{fw_s} (sim={sim_s}) — {url}".strip())

    why = sorted(
        (activation_details or []),
        key=lambda x: float(x.get("similarity", 0.0) or 0.0),
        reverse=True
    )[:10]
    why_lines = []
    for w in why:
        why_lines.append(
            f"- {w.get('clause_id')} → {w.get('risk_id')} (sim={float(w.get('similarity', 0.0) or 0.0):.3f})"
        )

    thr_s = f"{dynamic_threshold:.3f}" if isinstance(dynamic_threshold, (int, float)) else "n/a"

    brief = f"""
{heading}

{use_case_label}  
{use_case}

{evidence_label}  
{chr(10).join(evidence_lines) if evidence_lines else no_evidence}

{reasoning_label}  
Dynamic threshold used: **{thr_s}**  
{chr(10).join(why_lines) if why_lines else no_activation}

{posture_label}  
Activated risks: **{", ".join(activated_risks) if activated_risks else "None"}**  
Weighted risk score: **{weighted_risk_score:.2f}** → **{risk_level}**  
{legal_note}

{action_label}  
Recommended controls: **{", ".join(recommended_controls) if recommended_controls else "None"}**  
Coverage ratio: **{coverage_ratio:.2f}**  
Missing controls: **{", ".join(missing_controls) if missing_controls else "None"}**

{interp_label}  
{interpretation}
"""
    return brief.strip()


# =========================
# Embedding + Similarity
# =========================

def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]

def _normalize(vecs: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    return vecs / denom

def cosine_sim_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    # A: [n,d], B:[m,d] assumed normalized
    return A @ B.T

@st.cache_resource(show_spinner=False)
def load_embedding_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)

def embed_texts(model: SentenceTransformer, model_name: str, texts: List[str], is_query: bool) -> np.ndarray:
    """
    Handles E5-style prefixing for better quality.
    """
    if "e5" in model_name.lower():
        prefix = "query: " if is_query else "passage: "
        texts = [prefix + t.strip() for t in texts]
    emb = model.encode(texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
    return emb


# =========================
# Clause corpus loading (robust)
# =========================

def _try_parse_json_array(text: str) -> Optional[List[Dict[str, Any]]]:
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
        return None
    except Exception:
        return None

def load_clauses_any(path_json: str, path_jsonl: str) -> List[Dict[str, Any]]:
    """
    Load clauses from either:
      - clauses.jsonl: preferred if exists
      - clauses.json: fallback JSON array
    Also tolerant to the common mistake: dumping a JSON array into .jsonl file.
    """

    if os.path.exists(path_jsonl):
        with open(path_jsonl, "r", encoding="utf-8") as f:
            raw = f.read().strip()

        # If user accidentally put JSON array into jsonl file
        arr = _try_parse_json_array(raw)
        if isinstance(arr, list):
            return [c for c in arr if isinstance(c, dict)]

        clauses: List[Dict[str, Any]] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            line = line.rstrip(",")

            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    clauses.append(obj)
            except json.JSONDecodeError:
                if line in ("[", "]"):
                    continue
                repaired = re.sub(r",\s*}$", "}", line)
                try:
                    obj = json.loads(repaired)
                    if isinstance(obj, dict):
                        clauses.append(obj)
                except Exception:
                    raise ValueError(
                        f"Invalid JSONL line in {path_jsonl}:\n{line}\n\n"
                        f"Tip: Prefer one JSON object per line in clauses.jsonl."
                    )
        return clauses

    if os.path.exists(path_json):
        with open(path_json, "r", encoding="utf-8") as f:
            arr = json.load(f)
        if not isinstance(arr, list):
            raise ValueError(f"{path_json} must be a JSON array.")
        return [c for c in arr if isinstance(c, dict)]

    return []
    
def validate_and_sanitize_clauses(clauses: List[Dict[str, Any]], jurisdiction: str) -> List[Dict[str, Any]]:
    out = []
    for c in clauses:
        if not isinstance(c, dict):
            continue
        cid = c.get("clause_id") or c.get("id") or c.get("cid")
        title = c.get("title") or ""
        text = c.get("text") or c.get("content") or ""
        url = c.get("url") or ""
        if not cid:
            # create stable-ish id
            cid = f"{jurisdiction}-AUTO-{_sha1(title + '|' + text + '|' + url)}"
        c["clause_id"] = str(cid)
        c["title"] = str(title).strip()
        c["text"] = str(text).strip()
        c["url"] = str(url).strip()
        c["jurisdiction"] = c.get("jurisdiction") or jurisdiction
        out.append(c)

    # remove empty texts
    out = [c for c in out if c.get("text")]
    return out


# =========================
# Embedding cache
# =========================

def cache_paths(cache_dir: str, jurisdiction: str, model_name: str) -> Tuple[str, str]:
    os.makedirs(cache_dir, exist_ok=True)
    key = f"{jurisdiction}__{model_name}"
    h = _sha1(key)
    npz_path = os.path.join(cache_dir, f"clause_emb__{h}.npz")
    meta_path = os.path.join(cache_dir, f"clause_meta__{h}.json")
    return npz_path, meta_path

def build_or_load_clause_embeddings(
    clauses: List[Dict[str, Any]],
    jurisdiction: str,
    model_name: str,
    cache_dir: str = ".cache",
    force_rebuild: bool = False,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Returns:
      mat: [n,d] normalized embeddings
      clauses: aligned list
    """
    npz_path, meta_path = cache_paths(cache_dir, jurisdiction, model_name)

    # fingerprint clause ids + texts (quick)
    fp = _sha1("||".join([c["clause_id"] + ":" + _sha1(c["text"]) for c in clauses]))

    if (not force_rebuild) and os.path.exists(npz_path) and os.path.exists(meta_path):
        try:
            meta = json.load(open(meta_path, "r", encoding="utf-8"))
            if meta.get("fingerprint") == fp:
                data = np.load(npz_path)
                mat = data["emb"]
                return mat, clauses
        except Exception:
            pass  # fallback rebuild

    model = load_embedding_model(model_name)
    texts = [c["title"] + "\n" + c["text"] for c in clauses]
    emb = embed_texts(model, model_name, texts, is_query=False)

    np.savez_compressed(npz_path, emb=emb)
    json.dump(
        {"jurisdiction": jurisdiction, "model_name": model_name, "fingerprint": fp, "n": len(clauses)},
        open(meta_path, "w", encoding="utf-8"),
        ensure_ascii=False,
        indent=2,
    )
    return emb, clauses


# =========================
# Step 1 — Clause Matching
# =========================

def search_clauses(
    query: str,
    clauses: List[Dict[str, Any]],
    clause_emb: np.ndarray,
    model_name: str,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    if not clauses:
        return []

    model = load_embedding_model(model_name)
    q_emb = embed_texts(model, model_name, [query], is_query=True)  # [1,d]
    sims = (clause_emb @ q_emb.T).reshape(-1)  # [n]
    idx = np.argsort(-sims)[: max(1, top_k)]

    out = []
    for i in idx:
        c = dict(clauses[int(i)])
        c["score"] = float(sims[int(i)])
        out.append(c)
    return out


# =========================
# Risk + Control ontology (cheap but effective)
# =========================

RISK_CATALOG: Dict[str, Dict[str, Any]] = {
    "privacy_risk": {
        "title": "Personal data privacy risk",
        "desc": "Risk of personal information leakage, unlawful processing, over-collection, excessive retention, or weak access control in AI operations.",
        "weight": 1.2,
    },
    "security_risk": {
        "title": "Security / adversarial risk",
        "desc": "Risk of prompt injection, jailbreaks, data exfiltration, model extraction, or abuse via malicious inputs and weak operational security.",
        "weight": 1.3,
    },
    "hallucination_risk": {
        "title": "Hallucination / reliability risk",
        "desc": "Risk of confident but incorrect outputs, misinformation, inconsistent behavior, and unsafe advice; includes evaluation and monitoring gaps.",
        "weight": 1.1,
    },
    "transparency_risk": {
        "title": "Transparency / explainability risk",
        "desc": "Risk that stakeholders cannot understand AI use, limitations, or decision logic; missing disclosures, unclear reasoning, weak documentation.",
        "weight": 1.0,
    },
    "accountability_risk": {
        "title": "Accountability / governance risk",
        "desc": "Risk that roles, approvals, change control, and escalation pathways are unclear; decisions are not traceable or auditable.",
        "weight": 1.2,
    },
    "bias_risk": {
        "title": "Bias / fairness risk",
        "desc": "Risk of unfair outcomes or discriminatory impact due to biased data, evaluation gaps, or unaddressed group-level performance differences.",
        "weight": 1.1,
    },
    "incident_risk": {
        "title": "Incident response / harm management risk",
        "desc": "Risk that incidents are not detected, triaged, rolled back, communicated, or learned from; weak incident pathway and post-incident review.",
        "weight": 1.0,
    },
}

CONTROL_CATALOG: Dict[str, Dict[str, Any]] = {
    "data_minimisation": {
        "title": "Data minimisation + retention controls",
        "desc": "Minimise personal data processed; define retention; de-identify where possible; secure deletion.",
        "covers": ["privacy_risk"],
    },
    "access_control": {
        "title": "Access control + logging hygiene",
        "desc": "RBAC/least privilege for logs, prompts, data; audit trails; protect sensitive artifacts.",
        "covers": ["privacy_risk", "security_risk", "accountability_risk"],
    },
    "human_review": {
        "title": "Human review / fallback",
        "desc": "Require human approval for high-impact outputs; escalation paths; user recourse; override.",
        "covers": ["hallucination_risk", "transparency_risk", "incident_risk"],
    },
    "eval_monitoring": {
        "title": "Evaluation + continuous monitoring",
        "desc": "Pre-deploy test; red teaming; monitor drift/hallucinations/abuse; feedback loop.",
        "covers": ["hallucination_risk", "security_risk", "incident_risk"],
    },
    "prompt_change_control": {
        "title": "Prompt/system instruction change control",
        "desc": "Version prompts; review changes; test before release; document risk impact.",
        "covers": ["security_risk", "hallucination_risk", "accountability_risk"],
    },
    "transparency_notice": {
        "title": "Transparency notice + documentation",
        "desc": "Disclose AI use; limitations; uncertainty; maintain decision logs and evidence trail.",
        "covers": ["transparency_risk", "accountability_risk"],
    },
    "incident_playbook": {
        "title": "AI incident playbook",
        "desc": "Triage → rollback/fallback → comms → postmortem; define thresholds and owners.",
        "covers": ["incident_risk", "security_risk", "hallucination_risk"],
    },
    "fairness_eval": {
        "title": "Fairness evaluation + mitigation",
        "desc": "Assess disparate impact; dataset governance; mitigation and monitoring.",
        "covers": ["bias_risk"],
    },
}


# =========================
# Step 2 — Embedding-based Risk Mapping (+ dynamic threshold)
# =========================

def compute_dynamic_threshold(sim_matrix: np.ndarray, floor: float = 0.78, ceil: float = 0.90) -> float:
    """
    A cheap but robust dynamic threshold:
    - use a high percentile of the clause->risk similarity distribution
    - clamp to [floor, ceil]
    """
    if sim_matrix.size == 0:
        return 0.85
    vals = sim_matrix.flatten()
    # pick something like 85th percentile to avoid over-triggering
    p = float(np.percentile(vals, 85))
    # slight bias upward if distribution is flat
    spread = float(np.std(vals))
    thr = p + 0.15 * spread
    thr = max(floor, min(ceil, thr))
    return float(thr)

def risk_mapping_embedding(
    matched_clauses: List[Dict[str, Any]],
    model_name: str,
    threshold: Optional[float] = None,
) -> Tuple[List[str], List[Dict[str, Any]], np.ndarray, List[str]]:
    """
    Returns:
      activated_risks: list of risk_ids
      activation_details: edges clause->risk with similarity
      sim_matrix: [num_clauses, num_risks]
      risk_ids: ordered risk ids
    """
    if not matched_clauses:
        return [], [], np.zeros((0, 0), dtype=float), list(RISK_CATALOG.keys())

    model = load_embedding_model(model_name)

    clause_texts = [(c.get("title", "") + "\n" + c.get("text", "")).strip() for c in matched_clauses]
    risk_ids = list(RISK_CATALOG.keys())
    risk_texts = [(RISK_CATALOG[r]["title"] + "\n" + RISK_CATALOG[r]["desc"]).strip() for r in risk_ids]

    c_emb = embed_texts(model, model_name, clause_texts, is_query=False)
    r_emb = embed_texts(model, model_name, risk_texts, is_query=False)

    sim_matrix = cosine_sim_matrix(c_emb, r_emb)  # [C,R]

    dyn_thr = threshold if isinstance(threshold, (int, float)) else compute_dynamic_threshold(sim_matrix)

    activation_details: List[Dict[str, Any]] = []
    activated: set[str] = set()

    for i, clause in enumerate(matched_clauses):
        for j, rid in enumerate(risk_ids):
            sim = float(sim_matrix[i, j])
            if sim >= dyn_thr:
                activated.add(rid)
                activation_details.append({
                    "clause_id": clause.get("clause_id"),
                    "risk_id": rid,
                    "similarity": sim,
                })

    activated_risks = sorted(list(activated))
    return activated_risks, activation_details, sim_matrix, risk_ids


# =========================
# Step 2b — Graph Propagation (risk -> controls)
# =========================

def recommend_controls_for_risks(
    activated_risks: List[str],
) -> List[str]:
    ctrl_scores: Dict[str, float] = {cid: 0.0 for cid in CONTROL_CATALOG.keys()}

    for r in activated_risks:
        for cid, cinfo in CONTROL_CATALOG.items():
            if r in cinfo.get("covers", []):
                # simple additive score
                ctrl_scores[cid] += float(RISK_CATALOG.get(r, {}).get("weight", 1.0))

    ranked = sorted(ctrl_scores.items(), key=lambda x: x[1], reverse=True)
    # keep controls with positive score, max 6
    out = [cid for cid, s in ranked if s > 0][:6]
    return out


# =========================
# Step 3 — Weighted Risk Score + Coverage Gap
# =========================

def weighted_risk_score(
    activation_details: List[Dict[str, Any]],
    activated_risks: List[str],
) -> float:
    """
    Score = sum_r ( weight_r * max_similarity_over_edges_for_r )
    """
    if not activated_risks:
        return 0.0
    max_sim: Dict[str, float] = {r: 0.0 for r in activated_risks}
    for e in activation_details or []:
        r = e.get("risk_id")
        if r in max_sim:
            max_sim[r] = max(max_sim[r], float(e.get("similarity", 0.0) or 0.0))

    score = 0.0
    for r in activated_risks:
        w = float(RISK_CATALOG.get(r, {}).get("weight", 1.0))
        score += w * max_sim.get(r, 0.0)
    return float(score)

def risk_level_from_weighted_score(s: float) -> str:
    # Calibrate cheaply: typical sims ~0.80-0.90, sum weights ~2-4
    if s >= 3.2:
        return "HIGH"
    if s >= 2.4:
        return "MEDIUM"
    return "LOW"

def control_coverage_gap(
    activated_risks: List[str],
    recommended_controls: List[str],
) -> Tuple[float, List[str]]:
    if not activated_risks:
        return 1.0, []

    covered: set[str] = set()
    for cid in recommended_controls or []:
        covers = CONTROL_CATALOG.get(cid, {}).get("covers", [])
        for r in covers:
            covered.add(r)

    missing = [r for r in activated_risks if r not in covered]
    ratio = 1.0 - (len(missing) / max(1, len(activated_risks)))
    return float(ratio), missing


# =========================
# Graph build + centrality + clustering + visualization
# =========================

def build_governance_graph(
    clause_matches: List[Dict[str, Any]],
    activation_details: List[Dict[str, Any]],
    activated_risks: List[str],
    recommended_controls: List[str],
) -> nx.Graph:
    """
    Build a heterogeneous graph:
      clause nodes: prefix "clause:"
      risk nodes: "risk:"
      control nodes: "control:"
    edges:
      clause->risk weighted by similarity
      risk->control weighted by 1.0 (or by risk weight)
    """
    G = nx.Graph()

    # nodes
    for c in clause_matches or []:
        nid = f"clause:{c.get('clause_id')}"
        G.add_node(nid, kind="clause", title=c.get("title", ""), url=c.get("url", ""))

    for r in activated_risks or []:
        nid = f"risk:{r}"
        G.add_node(nid, kind="risk", title=RISK_CATALOG.get(r, {}).get("title", r))

    for ctrl in recommended_controls or []:
        nid = f"control:{ctrl}"
        G.add_node(nid, kind="control", title=CONTROL_CATALOG.get(ctrl, {}).get("title", ctrl))

    # edges clause->risk
    for e in activation_details or []:
        c = e.get("clause_id")
        r = e.get("risk_id")
        sim = float(e.get("similarity", 0.0) or 0.0)
        if not c or not r:
            continue
        cn = f"clause:{c}"
        rn = f"risk:{r}"
        if cn in G and rn in G:
            G.add_edge(cn, rn, weight=sim, kind="clause_risk")

    # edges risk->control
    for r in activated_risks or []:
        rn = f"risk:{r}"
        rw = float(RISK_CATALOG.get(r, {}).get("weight", 1.0))
        for ctrl in recommended_controls or []:
            if r in CONTROL_CATALOG.get(ctrl, {}).get("covers", []):
                cn = f"control:{ctrl}"
                if rn in G and cn in G:
                    G.add_edge(rn, cn, weight=1.0 * rw, kind="risk_control")

    return G

def compute_centrality(G: nx.Graph) -> Dict[str, Dict[str, float]]:
    if G.number_of_nodes() == 0:
        return {"degree": {}, "betweenness": {}}
    deg = nx.degree_centrality(G)
    btw = nx.betweenness_centrality(G, normalized=True)
    return {
        "degree": {k: float(v) for k, v in deg.items()},
        "betweenness": {k: float(v) for k, v in btw.items()},
    }

def cluster_risks(sim_matrix: np.ndarray, risk_ids: List[str], min_edges: float = 0.80) -> Dict[str, List[str]]:
    """
    Risk clustering using a risk-risk similarity graph derived from clause->risk similarities.
    We build risk vectors by aggregating similarities across clauses, then compute risk-risk cosine.
    """
    if sim_matrix.size == 0 or len(risk_ids) == 0:
        return {}

    # risk vectors: each risk is a column vector across clauses
    R = sim_matrix.T  # [R,C]
    # normalize
    Rn = R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-12)
    rr = Rn @ Rn.T  # [R,R]

    H = nx.Graph()
    for i, rid in enumerate(risk_ids):
        H.add_node(rid)

    for i in range(len(risk_ids)):
        for j in range(i + 1, len(risk_ids)):
            w = float(rr[i, j])
            if w >= min_edges:
                H.add_edge(risk_ids[i], risk_ids[j], weight=w)

    if H.number_of_edges() == 0:
        # fallback: each risk alone
        return {str(i): [rid] for i, rid in enumerate(risk_ids)}

    comms = list(nx.algorithms.community.greedy_modularity_communities(H, weight="weight"))
    out: Dict[str, List[str]] = {}
    for idx, c in enumerate(comms):
        out[str(idx)] = sorted(list(c))
    return out

def plot_similarity_matrix(sim_matrix: np.ndarray, clause_ids: List[str], risk_ids: List[str], title: str):
    if sim_matrix.size == 0:
        st.info("No similarity matrix to plot.")
        return

    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111)
    im = ax.imshow(sim_matrix, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("Risks")
    ax.set_ylabel("Matched Clauses")
    ax.set_xticks(range(len(risk_ids)))
    ax.set_xticklabels(risk_ids, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(clause_ids)))
    ax.set_yticklabels(clause_ids, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    st.pyplot(fig)

def plot_graph(G: nx.Graph):
    if G.number_of_nodes() == 0:
        st.info("Graph is empty.")
        return

    # layout
    pos = nx.spring_layout(G, seed=42, k=0.9)

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111)
    ax.axis("off")

    # color by kind without specifying colors explicitly (we'll use defaults via alpha/markers)
    kinds = nx.get_node_attributes(G, "kind")

    clause_nodes = [n for n, k in kinds.items() if k == "clause"]
    risk_nodes = [n for n, k in kinds.items() if k == "risk"]
    control_nodes = [n for n, k in kinds.items() if k == "control"]

    nx.draw_networkx_nodes(G, pos, nodelist=clause_nodes, node_shape="s", node_size=900, alpha=0.9, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=risk_nodes, node_shape="o", node_size=900, alpha=0.9, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=control_nodes, node_shape="^", node_size=900, alpha=0.9, ax=ax)

    # edges with width by weight
    weights = []
    for u, v, d in G.edges(data=True):
        weights.append(float(d.get("weight", 1.0)))
    if weights:
        w_min, w_max = min(weights), max(weights)
        widths = []
        for w in weights:
            # scale to [1, 5]
            if w_max - w_min < 1e-9:
                widths.append(2.0)
            else:
                widths.append(1.0 + 4.0 * (w - w_min) / (w_max - w_min))
    else:
        widths = 2.0

    nx.draw_networkx_edges(G, pos, width=widths, alpha=0.6, ax=ax)

    # labels: shorten
    def short(n: str) -> str:
        if n.startswith("clause:"):
            return n.replace("clause:", "")
        if n.startswith("risk:"):
            return n.replace("risk:", "")
        if n.startswith("control:"):
            return n.replace("control:", "")
        return n

    labels = {n: short(n) for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, ax=ax)

    st.pyplot(fig)


# =========================
# Latest developments (RSS)
# =========================

def build_google_news_rss(jurisdiction: str, extra_terms: str, recency_days: int) -> str:
    # basic mapping for locale hints
    locale = {
        "JP": ("hl=ja&gl=JP&ceid=JP:ja", "Japan"),
        "AU": ("hl=en-AU&gl=AU&ceid=AU:en", "Australia"),
        "EU": ("hl=en&gl=EU&ceid=EU:en", "Europe"),
        "US": ("hl=en-US&gl=US&ceid=US:en", "United States"),
        "UK": ("hl=en-GB&gl=GB&ceid=GB:en", "United Kingdom"),
        "CA": ("hl=en-CA&gl=CA&ceid=CA:en", "Canada"),
    }
    loc_qs, country_hint = locale.get(jurisdiction, ("hl=en&gl=US&ceid=US:en", jurisdiction))

    base_terms = f"(AI governance OR AI regulation OR AI law OR guideline OR framework) {country_hint}"
    if extra_terms.strip():
        base_terms += f" {extra_terms.strip()}"

    # Google News supports when:Xd in query
    q = base_terms + f" when:{int(recency_days)}d"
    q = q.replace(" ", "+")
    return f"https://news.google.com/rss/search?q={q}&{loc_qs}"

def fetch_rss_items(url: str, limit: int = 10) -> List[Dict[str, Any]]:
    feed = feedparser.parse(url)
    items = []
    for e in (feed.entries or [])[:limit]:
        items.append({
            "title": e.get("title", ""),
            "link": e.get("link", ""),
            "published": e.get("published", e.get("updated", "")),
            "source": (e.get("source", {}) or {}).get("title", ""),
        })
    return items

def load_benchmark_cases(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        return []
    return [x for x in data if isinstance(x, dict)]

def get_benchmark_options(
    benchmark_cases: List[Dict[str, Any]],
    jurisdiction: str,
) -> List[Dict[str, Any]]:
    filtered = []
    for case in benchmark_cases:
        case_jur = str(case.get("jurisdiction", "")).strip().upper()
        if case_jur == jurisdiction.upper():
            filtered.append(case)
    return filtered
    
def build_next_steps(
    lang: str,
    activated_risks: List[str],
    missing_controls: List[str],
    coverage_ratio: float,
    evidence_count: int,
    top_central: List[Tuple[str, float]],
    risk_level: str,
    recommended_controls: List[str],
) -> str:
    """
    Build risk-driven next-step suggestions.
    This is intentionally deterministic but should vary meaningfully across use cases.
    """

    if lang == "ja":
        header = "**次にやるべきこと（実行可能）**"

        risk_action_map = {
            "privacy_risk": [
                "個人情報の取得・保存・ログ出力の範囲を見直し、データ最小化を明文化する。",
                "会話ログや埋め込みに対するアクセス権限を整理し、保持期間を設定する。",
            ],
            "security_risk": [
                "プロンプトインジェクションや情報漏洩のシナリオを明示し、入力防御とアクセス制御を強化する。",
                "運用上の攻撃面（外部接続、検索連携、内部文書参照）ごとに監視ポイントを決める。",
            ],
            "hallucination_risk": [
                "高リスク出力については human review / fallback を導入し、誤回答時のエスカレーション手順を作る。",
                "評価データセットを用意し、事前評価と継続監視の仕組みを作る。",
            ],
            "transparency_risk": [
                "利用者向けに『AIが回答していること』『限界』『人間への引き継ぎ条件』を明確に表示する。",
                "内部向けには、どの証拠からどの判断に至ったかを記録できる説明ログを整備する。",
            ],
            "accountability_risk": [
                "誰が承認し、誰が運用し、誰が事故対応を持つのかを明確にし、責任分界を定義する。",
                "変更管理・例外承認・定期レビューのルールを作り、意思決定を再利用可能な形で残す。",
            ],
            "bias_risk": [
                "評価データの偏りや対象群ごとの差異を点検し、公平性評価を定例化する。",
                "採用・与信・人事評価のような高影響用途では、人間レビューを外さない。",
            ],
            "incident_risk": [
                "インシデント発生時の triage → rollback/fallback → 報告 → 振り返りの手順を整備する。",
                "異常検知の閾値と、誰が止めるかを事前に決めておく。",
            ],
        }

        control_action_map = {
            "data_minimisation": "データ最小化と保持期間ルールを優先して整備する。",
            "access_control": "ログ・モデル・社内データへのアクセス制御を優先して整備する。",
            "human_review": "高影響出力に対する人手確認フローを優先して整備する。",
            "eval_monitoring": "事前評価と継続監視の仕組みを優先して整備する。",
            "prompt_change_control": "プロンプト／システム指示の変更管理を優先して整備する。",
            "transparency_notice": "利用者向け説明と内部記録のテンプレートを優先して整備する。",
            "incident_playbook": "AIインシデント対応手順を優先して整備する。",
            "fairness_eval": "公平性評価と差分検証を優先して整備する。",
        }

        intro = []
        if risk_level == "HIGH":
            intro.append("現在のリスク水準は高いため、まずは高影響リスクへの即応策を優先する。")
        elif risk_level == "MEDIUM":
            intro.append("現在のリスク水準は中程度であり、運用ルールと監視体制の整備を優先する。")
        else:
            intro.append("現在のリスク水準は比較的低いが、将来の拡張を見据えて最小限の統制を先に整える。")

        if evidence_count < 5:
            intro.append("根拠条項が少ないため、まず証拠データセットの拡充が必要である。")

        lines: List[str] = [header, ""]

        for s in intro:
            lines.append(f"- {s}")

        if activated_risks:
            lines.append("")
            lines.append("**リスク別の優先アクション**")
            for rid in activated_risks:
                actions = risk_action_map.get(rid, [])
                if actions:
                    lines.append(f"- `{rid}`")
                    for a in actions[:2]:
                        lines.append(f"  - {a}")

        if missing_controls:
            lines.append("")
            lines.append("**不足コントロールへの対応**")
            for ctrl in missing_controls:
                msg = control_action_map.get(ctrl, f"`{ctrl}` を優先的に整備する。")
                lines.append(f"- {msg}")

        if recommended_controls:
            lines.append("")
            lines.append("**優先して運用に載せる統制**")
            for ctrl in recommended_controls[:4]:
                msg = control_action_map.get(ctrl, f"`{ctrl}` を実装・運用対象に含める。")
                lines.append(f"- {msg}")

        if coverage_ratio < 0.50:
            lines.append("")
            lines.append("- 現在の統制カバー率が低いため、まずは不足コントロールの補完を最優先にする。")
        elif coverage_ratio > 0.95:
            lines.append("")
            lines.append("- 統制カバー率は高いが、過剰検出の可能性もあるため、しきい値やリスク定義の妥当性を見直す。")

        # centrality: only show useful nodes
        useful_central = [(n, v) for n, v in top_central if v > 0.0]
        if useful_central:
            lines.append("")
            lines.append("**構造上の重要ノード**")
            tops = ", ".join([f"{n}({v:.2f})" for n, v in useful_central[:5]])
            lines.append(f"- {tops}")

        return "\n".join(lines)

    else:
        header = "**What to do next (actionable)**"

        risk_action_map = {
            "privacy_risk": [
                "Tighten data minimisation rules for collection, storage, and logging of personal data.",
                "Define retention windows and access boundaries for logs, embeddings, and customer records.",
            ],
            "security_risk": [
                "Document prompt injection / data exfiltration scenarios and strengthen input defenses plus access control.",
                "Define monitoring points for the main attack surfaces: external inputs, retrieval connections, and internal data access.",
            ],
            "hallucination_risk": [
                "Introduce human review / fallback for high-impact outputs and define escalation for incorrect answers.",
                "Set up an evaluation dataset and continuous monitoring process for output reliability.",
            ],
            "transparency_risk": [
                "Clearly disclose AI use, limitations, and handoff conditions to users.",
                "Create internal explanation logs showing which evidence led to which governance conclusion.",
            ],
            "accountability_risk": [
                "Clarify who approves, who operates, and who owns incident response for the AI system.",
                "Create change control, exception approval, and recurring review rules so decisions become reusable artifacts.",
            ],
            "bias_risk": [
                "Audit dataset and performance differences across groups, and operationalize fairness evaluation.",
                "Keep human review in the loop for high-impact use cases such as hiring, lending, and promotion.",
            ],
            "incident_risk": [
                "Create a triage → rollback/fallback → reporting → postmortem workflow for AI incidents.",
                "Define escalation thresholds and assign explicit stop/go decision owners in advance.",
            ],
        }

        control_action_map = {
            "data_minimisation": "Prioritize data minimisation and retention controls.",
            "access_control": "Prioritize access control for logs, models, and internal data.",
            "human_review": "Prioritize a human review workflow for high-impact outputs.",
            "eval_monitoring": "Prioritize pre-deployment evaluation and continuous monitoring.",
            "prompt_change_control": "Prioritize change control for prompts and system instructions.",
            "transparency_notice": "Prioritize user-facing disclosures and internal decision logging.",
            "incident_playbook": "Prioritize an AI incident response playbook.",
            "fairness_eval": "Prioritize fairness evaluation and disparity testing.",
        }

        intro = []
        if risk_level == "HIGH":
            intro.append("The current risk level is high, so immediate mitigation for high-impact risks should come first.")
        elif risk_level == "MEDIUM":
            intro.append("The current risk level is medium, so operational governance and monitoring should be the priority.")
        else:
            intro.append("The current risk level is relatively low, but baseline controls should still be established early.")

        if evidence_count < 5:
            intro.append("Evidence coverage is still thin, so expanding the clause dataset should be an early priority.")

        lines: List[str] = [header, ""]

        for s in intro:
            lines.append(f"- {s}")

        if activated_risks:
            lines.append("")
            lines.append("**Risk-specific priority actions**")
            for rid in activated_risks:
                actions = risk_action_map.get(rid, [])
                if actions:
                    lines.append(f"- `{rid}`")
                    for a in actions[:2]:
                        lines.append(f"  - {a}")

        if missing_controls:
            lines.append("")
            lines.append("**Missing controls to add first**")
            for ctrl in missing_controls:
                msg = control_action_map.get(ctrl, f"Prioritize `{ctrl}`.")
                lines.append(f"- {msg}")

        if recommended_controls:
            lines.append("")
            lines.append("**Controls to operationalize next**")
            for ctrl in recommended_controls[:4]:
                msg = control_action_map.get(ctrl, f"Operationalize `{ctrl}`.")
                lines.append(f"- {msg}")

        if coverage_ratio < 0.50:
            lines.append("")
            lines.append("- Control coverage is still low, so closing the missing-control gap should be the first priority.")
        elif coverage_ratio > 0.95:
            lines.append("")
            lines.append("- Coverage is high, but you should also review whether the threshold or mappings are over-triggering.")

        useful_central = [(n, v) for n, v in top_central if v > 0.0]
        if useful_central:
            lines.append("")
            lines.append("**Structurally important nodes**")
            tops = ", ".join([f"{n}({v:.2f})" for n, v in useful_central[:5]])
            lines.append(f"- {tops}")

        return "\n".join(lines)

def load_benchmark_cases(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
        return []
    except Exception:
        return []


def get_benchmark_options(
    benchmark_cases: List[Dict[str, Any]],
    jurisdiction: str,
) -> List[Dict[str, Any]]:
    filtered = []
    for case in benchmark_cases:
        case_jur = str(case.get("jurisdiction", "")).strip().upper()
        if case_jur == jurisdiction.upper():
            filtered.append(case)
    return filtered
    
# =========================
# UI
# =========================
st.set_page_config(page_title="AI Governance Lab — Explainability Finder", layout="wide")

# benchmark dataset
BENCHMARK_PATH = "benchmark_ai_governance.json"
benchmark_cases_all = load_benchmark_cases(BENCHMARK_PATH)

# language selector
lang_label_map = {"English": "en", "日本語": "ja"}
lang_ui = st.sidebar.selectbox(
    "Language / 言語",
    options=["English", "日本語"],
    index=0,
    key="sidebar_language_select",
)
lang = lang_label_map[lang_ui]

BENCHMARK_PATH = "benchmark_ai_governance.json"
benchmark_cases = load_benchmark_cases(BENCHMARK_PATH)

# language selector
lang_label_map = {"English": "en", "日本語": "ja"}
lang_ui = st.sidebar.selectbox(
    "Language / 言語",
    options=["English", "日本語"],
    index=0,
)
lang = lang_label_map[lang_ui]

st.title(t(lang, "app_title"))
st.caption("This tool sells **organizational explainability**: evidence → thresholds → mappings → graph propagation → coverage gaps → decision trace.")

with st.sidebar:
    st.header(t(lang, "settings"))

    jurisdiction = st.selectbox(
    t(lang, "jurisdiction"),
    ["JP", "AU", "EU", "US", "UK", "CA"],
    index=0,
    help="Choose where governance evidence should come from. Add your own corpus under law_corpus/<JUR>/clauses.json.",
    key="sidebar_jurisdiction_select",
)

    # =========================
    # Benchmark selector
    # =========================
    st.header(t(lang, "settings"))

    jurisdiction = st.selectbox(
        t(lang, "jurisdiction"),
        ["JP", "AU", "EU", "US", "UK", "CA"],
        index=0,
        help="Choose where governance evidence should come from. Add your own corpus under law_corpus/<JUR>/clauses.json.",
    )
    
    # =========================
    # Benchmark selector
    # =========================
    benchmark_cases = get_benchmark_options(benchmark_cases_all, jurisdiction)
    
    st.subheader("Benchmark")
    
    benchmark_mode = st.checkbox(
        "Use benchmark case",
        value=False
    )
    
    selected_benchmark_case = None
    default_use_case = "LLM-based customer support chatbot processing personal data; risks include prompt injection, data leakage, and hallucinations."
    
    if benchmark_mode and benchmark_cases:
        benchmark_labels = [
            f"{c.get('id', 'unknown')} | {c.get('use_case', '')[:80]}"
            for c in benchmark_cases
        ]
    
        selected_label = st.selectbox(
        "Benchmark case",
        options=benchmark_labels,
        index=0,
        key="sidebar_benchmark_case_select",
        )
    
        selected_idx = benchmark_labels.index(selected_label)
        selected_benchmark_case = benchmark_cases[selected_idx]
        default_use_case = selected_benchmark_case.get("use_case", default_use_case)
    
    elif benchmark_mode and not benchmark_cases:
        st.info(f"No benchmark cases found for jurisdiction: {jurisdiction}")
    
    default_use_case = "LLM-based customer support chatbot processing personal data."

    if selected_case:
        default_use_case = selected_case.get("use_case", default_use_case)
    
    use_case = st.text_area(
        t(lang, "use_case"),
        value=default_use_case,
        height=120,
    )
    
    if selected_benchmark_case is not None:
        with st.expander("Benchmark expected outputs", expanded=False):
            st.json({
                "id": selected_benchmark_case.get("id"),
                "expected_risks": selected_benchmark_case.get("expected_risks", []),
                "expected_controls": selected_benchmark_case.get("expected_controls", []),
                "expected_risk_level": selected_benchmark_case.get("expected_risk_level", ""),
                "jurisdiction": selected_benchmark_case.get("jurisdiction", ""),
            })
    
    st.subheader(t(lang, "semantic_evidence"))
    top_k = st.slider(t(lang, "topk_clauses"), 3, 12, 5)
    
    model_name = st.selectbox(
        t(lang, "embedding_model"),
        [
            "intfloat/multilingual-e5-small",
            "sentence-transformers/all-MiniLM-L6-v2",
            "BAAI/bge-m3",
        ],
        index=0,
        key="sidebar_embedding_model_select",
    )
   
    st.subheader(t(lang, "dyn_threshold_scoring"))
    use_auto_threshold = st.checkbox(t(lang, "use_dynamic_threshold"), value=True)
    manual_threshold = st.slider(t(lang, "manual_threshold"), 0.70, 0.95, 0.83, 0.005)
    
    # graph style controls
    st.subheader("Visualization")
    viz_style = st.selectbox(
        t(lang, "viz_style"),
        options=[t(lang, "viz_soft"), t(lang, "viz_contrast")],
        index=0,
        key="sidebar_viz_style_select",
    )
    palette_name = "soft" if viz_style == t(lang, "viz_soft") else "contrast"
    node_alpha = st.slider(t(lang, "node_alpha"), 0.50, 1.00, 0.92)
    edge_alpha = st.slider(t(lang, "edge_alpha"), 0.10, 1.00, 0.55)
    edge_width_scale = st.slider(t(lang, "edge_width"), 0.5, 6.0, 3.0)
    
    st.subheader(t(lang, "latest_dev"))
    enable_rss = st.checkbox(t(lang, "fetch_rss"), value=True)
    recency_days = st.slider(t(lang, "recency_days"), 7, 60, 30)
    
    extra_terms = st.text_input(
        "Extra search terms (optional)",
        value="AI Act OR guideline OR framework OR compliance",
        help="Optional keywords added to the governance news search query.",
    )
    
    st.subheader("Corpus")
    force_rebuild = st.checkbox(
        "Force rebuild embeddings cache",
        value=False,
        help="Use if you updated clauses.json."
    )


# Load corpus
corpus_dir = os.path.join("law_corpus", jurisdiction)
clauses_json = os.path.join(corpus_dir, "clauses.json")
clauses_jsonl = os.path.join(corpus_dir, "clauses.jsonl")

raw_clauses = load_clauses_any(clauses_json, clauses_jsonl)
clauses = validate_and_sanitize_clauses(raw_clauses, jurisdiction)

left, right = st.columns([1.1, 1.2])

with left:
    st.subheader("📚 Evidence Sources (authoritative, cite-backed)")
    if not clauses:
        st.error(
            f"No clauses found for {jurisdiction}.\n\n"
            f"Add one of these files:\n"
            f"- {clauses_json} (recommended: JSON array)\n"
            f"- {clauses_jsonl} (JSONL)\n"
        )
        st.stop()

    st.write(f"Loaded **{len(clauses)}** clauses from `{corpus_dir}`.")
    with st.expander("Show clause schema expectations / tips", expanded=False):
        st.markdown(
            """
- Prefer `clauses.json` (a JSON array).  
- Each clause should include: `clause_id`, `title`, `text`, `url` (recommended).  
- Avoid pasting a JSON array into `.jsonl` — but if you did, this app tries to handle it anyway.
"""
        )

    # Build embeddings
    with st.spinner("Building / loading clause embeddings (cached)…"):
        clause_emb, clauses_aligned = build_or_load_clause_embeddings(
            clauses=clauses,
            jurisdiction=jurisdiction,
            model_name=model_name,
            cache_dir=".cache",
            force_rebuild=force_rebuild,
        )

    st.success("Embeddings ready.")

    # Step 1: retrieval
    st.subheader("🔎 Step 1 — Clause Matching")
    matches = search_clauses(
        query=use_case,
        clauses=clauses_aligned,
        clause_emb=clause_emb,
        model_name=model_name,
        top_k=top_k,
    )

    for m in matches:
        st.write(f"**{m.get('clause_id')}**: {m.get('title','(no title)')}  \nscore: `{m.get('score', 0.0):.3f}`  \n{m.get('url','')}")

    # Step 2: risk mapping
    st.subheader("🧠 Step 2 — Embedding-based Risk Mapping")
    thr = None
    if not use_auto_threshold:
        thr = float(manual_threshold)

    activated_risks, activation_details, sim_matrix, risk_ids = risk_mapping_embedding(
        matched_clauses=matches,
        model_name=model_name,
        threshold=thr,
    )

    # dynamic threshold used (even when manual is None we compute)
    dynamic_threshold = thr if isinstance(thr, (int, float)) else compute_dynamic_threshold(sim_matrix)

    st.markdown("**Activation Explanation**")
    st.write(f"Dynamic threshold used: `{dynamic_threshold:.3f}`")

    # show activation details (sorted)
    activation_details_sorted = sorted(
        activation_details,
        key=lambda x: float(x.get("similarity", 0.0) or 0.0),
        reverse=True
    )

    if activation_details_sorted:
        for e in activation_details_sorted[:25]:
            st.write(
                f"{e.get('clause_id')} activated **{e.get('risk_id')}** "
                f"(similarity={float(e.get('similarity', 0.0) or 0.0):.3f})"
            )
    else:
        st.info("No risks activated at current threshold. Lower threshold or enrich clauses.")

    # similarity matrix visualization
    with st.expander("Similarity matrix (clause → risk)", expanded=True):
        clause_ids = [m.get("clause_id", "?") for m in matches]
        plot_similarity_matrix(sim_matrix, clause_ids, risk_ids, "Clause → Risk similarity matrix")

    # Recommend controls
    st.subheader("🧩 Step 2b — Graph Propagation (Risk → Controls)")
    recommended_controls = recommend_controls_for_risks(activated_risks)

    st.markdown("**Activated Risks**")
    st.json(activated_risks)

    st.markdown("**Recommended Controls**")
    st.json(recommended_controls)

    # Step 3 scoring
    st.subheader("📊 Step 3 — Weighted Risk Score + Coverage Gap")
    wscore = weighted_risk_score(activation_details, activated_risks)
    rlevel = risk_level_from_weighted_score(wscore)

    coverage_ratio, missing_controls = control_coverage_gap(activated_risks, recommended_controls)

    st.markdown("**Control Coverage Analysis**")
    st.write(f"Coverage ratio: `{coverage_ratio:.2f}`")
    st.write(f"Missing controls: `{missing_controls}`")

    st.markdown("**Weighted Risk Score**")
    st.write(f"Score: `{wscore:.2f}`")
    st.write(f"Risk Level: **{rlevel}**")

    # Graph + centrality + clustering
    st.subheader("🕸️ " + t(lang, "graph_section_title"))
    G = build_governance_graph(matches, activation_details, activated_risks, recommended_controls)

    with st.expander(t(lang, "graph_viz"), expanded=True):
        fig_graph = draw_governance_graph(
            G,
            palette_name=palette_name,
            node_alpha=node_alpha,
            edge_alpha=edge_alpha,
            edge_width_scale=edge_width_scale,
            seed=7,
        )
        st.pyplot(fig_graph, clear_figure=True)
    
        st.caption(
            f"{t(lang, 'legend')}: "
            f"■ {t(lang, 'legend_clause')}  "
            f"● {t(lang, 'legend_risk')}  "
            f"▲ {t(lang, 'legend_control')}"
        )

    centrality = compute_centrality(G)
    with st.expander("Centrality analysis (degree / betweenness)", expanded=False):
        # make keys shorter for readability
        def pretty_key(k: str) -> str:
            return k.replace("clause:", "").replace("risk:", "").replace("control:", "")
        centrality_pretty = {
            "degree": {pretty_key(k): v for k, v in centrality["degree"].items()},
            "betweenness": {pretty_key(k): v for k, v in centrality["betweenness"].items()},
        }
        st.json(centrality_pretty)

    with st.expander("Risk clustering", expanded=False):
        clusters = cluster_risks(sim_matrix, risk_ids, min_edges=max(0.80, dynamic_threshold - 0.03))
        st.json(clusters)

    # Latest developments RSS
    if enable_rss:
        st.subheader("📰 " + t(lang, "latest_dev_title"))
        rss_url = build_google_news_rss(jurisdiction, extra_terms, recency_days)
        st.caption(f"RSS URL: {rss_url}")
        with st.spinner("Fetching RSS…"):
            items = fetch_rss_items(rss_url, limit=10)
        if items:
            for i, it in enumerate(items, start=1):
                st.write(f"{i}. {it['title']}  \n{it['published']} · {it.get('source','')}  \n{it['link']}")
        else:
            st.info("No RSS items found (try different terms).")

with right:
    st.subheader("🧭 " + t(lang, "explainability"))
    score_obj = compute_explainability_score(
        clause_matches=matches,
        activation_details=activation_details,
        activated_risks=activated_risks,
        recommended_controls=recommended_controls,
        coverage_ratio=coverage_ratio,
        dynamic_threshold=dynamic_threshold,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(t(lang, "explainability_score"), f"{score_obj.total:.2f}", score_obj.label)
    c2.metric(t(lang, "evidence_trace"), f"{score_obj.components['evidence_trace']:.2f}")
    c3.metric(t(lang, "reasoning_transparency"), f"{score_obj.components['reasoning_transparency']:.2f}")
    c4.metric(t(lang, "control_coverage"), f"{score_obj.components['control_coverage']:.2f}")

    with st.expander("Why this score (deterministic)", expanded=True):
        st.write(score_obj.narrative)

    st.subheader("📝 " + t(lang, "decision_trace"))
    brief = build_explainability_brief(
        lang=lang,
        jurisdiction=jurisdiction,
        use_case=use_case,
        clause_matches=matches,
        activation_details=activation_details,
        activated_risks=activated_risks,
        recommended_controls=recommended_controls,
        weighted_risk_score=wscore,
        risk_level=rlevel,
        dynamic_threshold=dynamic_threshold,
        coverage_ratio=coverage_ratio,
        missing_controls=missing_controls,
    )
    st.markdown(brief)

    st.subheader("✅ " + t(lang, "next_steps"))

    raw_central = sorted(
        compute_centrality(G).get("betweenness", {}).items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    # remove zero-score centrality nodes and shorten names a bit
    top_central = []
    for n, v in raw_central:
        if v <= 0.0:
            continue
        pretty_n = n.replace("clause:", "").replace("risk:", "").replace("control:", "")
        top_central.append((pretty_n, v))
    
    ns = build_next_steps(
        lang=lang,
        activated_risks=activated_risks,
        missing_controls=missing_controls,
        coverage_ratio=coverage_ratio,
        evidence_count=len(matches),
        top_central=top_central,
        risk_level=rlevel,
        recommended_controls=recommended_controls,
    )
    st.markdown(ns)
    
    if selected_benchmark_case is not None:
        st.subheader("🧪 Benchmark comparison")
    
        expected_risks = set(selected_benchmark_case.get("expected_risks", []))
        expected_controls = set(selected_benchmark_case.get("expected_controls", []))
        expected_level = selected_benchmark_case.get("expected_risk_level", "")
    
        actual_risks = set(activated_risks)
        actual_controls = set(recommended_controls)
        actual_level = rlevel
    
        risk_overlap = sorted(list(expected_risks & actual_risks))
        control_overlap = sorted(list(expected_controls & actual_controls))
    
        st.json({
            "benchmark_id": selected_benchmark_case.get("id"),
            "expected_risks": sorted(list(expected_risks)),
            "actual_risks": sorted(list(actual_risks)),
            "matched_risks": risk_overlap,
            "expected_controls": sorted(list(expected_controls)),
            "actual_controls": sorted(list(actual_controls)),
            "matched_controls": control_overlap,
            "expected_risk_level": expected_level,
            "actual_risk_level": actual_level,
        })    
        
