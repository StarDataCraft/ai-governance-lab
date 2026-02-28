from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


# -----------------------------
# Data structure
# -----------------------------

@dataclass
class Clause:
    clause_id: str
    source_id: str
    title: str
    url: str
    text: str


@dataclass
class Match:
    clause: Clause
    score: float


# -----------------------------
# Embedding model (local)
# -----------------------------

_model = SentenceTransformer("intfloat/multilingual-e5-small")


def embed_texts(texts: List[str]):
    return _model.encode(texts, normalize_embeddings=True)


# -----------------------------
# JSON Loader (NOT JSONL)
# -----------------------------

def load_clauses_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Clause file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    clauses = []
    for obj in data:
        clauses.append(
            Clause(
                clause_id=str(obj["clause_id"]),
                source_id=str(obj["source_id"]),
                title=str(obj.get("title", "")),
                url=str(obj.get("url", "")),
                text=str(obj.get("text", "")),
            )
        )

    return clauses


# -----------------------------
# Retrieval
# -----------------------------

def cosine_top_k(query_vec: np.ndarray, mat: np.ndarray, k: int):
    sims = mat @ query_vec
    idx = np.argsort(-sims)[:k]
    return [(int(i), float(sims[i])) for i in idx]


def search_clauses(
    query: str,
    jurisdiction: str,
    corpus_dir: str | Path = "law_corpus",
    top_k: int = 5,
):

    corpus_dir = Path(corpus_dir)

    clauses_path = corpus_dir / jurisdiction / "clauses.json"

    clauses = load_clauses_json(clauses_path)

    texts = [c.text for c in clauses]
    mat = np.array(embed_texts(texts), dtype=np.float32)

    q_vec = np.array(embed_texts([query])[0], dtype=np.float32)

    top = cosine_top_k(q_vec, mat, k=top_k)

    return [Match(clause=clauses[i], score=score) for i, score in top]
