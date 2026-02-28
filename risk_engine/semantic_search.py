# risk_engine/semantic_search.py

from __future__ import annotations

from pathlib import Path
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from .embeddings import embed_texts_local

def load_clauses_jsonl(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Clause file not found: {path}")

    clauses = []
    with path.open("r", encoding="utf-8") as f:
        for i, raw in enumerate(f):
            line = raw.strip()

            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print("========== JSON ERROR ==========")
                print("File:", path)
                print("Line number:", i + 1)
                print("Raw content:")
                print(repr(line))
                print("================================")
                raise e

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


def load_clauses_jsonl(path: Path) -> List[Clause]:
    if not path.exists():
        raise FileNotFoundError(f"Clause file not found: {path}")
    clauses: List[Clause] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
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


def cosine_top_k(query_vec: np.ndarray, mat: np.ndarray, k: int):
    sims = mat @ query_vec
    idx = np.argsort(-sims)[:k]
    return [(int(i), float(sims[i])) for i in idx]


def build_embeddings_on_the_fly(
    jurisdiction: str,
    corpus_dir: Path,
):
    clauses_path = corpus_dir / jurisdiction / "clauses.jsonl"
    clauses = load_clauses_jsonl(clauses_path)

    texts = [c.text for c in clauses]
    embs = embed_texts_local(texts)
    mat = np.array(embs, dtype=np.float32)

    return mat, clauses


def search_clauses(
    query: str,
    jurisdiction: str,
    cache_dir: str | Path = "data",
    corpus_dir: str | Path = "law_corpus",
    top_k: int = 5,
):

    corpus_dir = Path(corpus_dir)

    # 🔥 No cache. Build dynamically.
    mat, clauses = build_embeddings_on_the_fly(jurisdiction, corpus_dir)

    q_emb = embed_texts_local([query])[0]
    q = np.array(q_emb, dtype=np.float32)

    top = cosine_top_k(q, mat, k=top_k)
    return [Match(clause=clauses[i], score=score) for i, score in top]
