# risk_engine/semantic_search.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from .embeddings import embed_texts_local


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
            line = line.strip()
            if not line:
                continue
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
    if not clauses:
        raise ValueError(f"No clauses loaded from: {path}")
    return clauses


def cosine_top_k(query_vec: np.ndarray, mat: np.ndarray, k: int) -> List[Tuple[int, float]]:
    # If embeddings already normalized, dot product = cosine similarity
    sims = mat @ query_vec
    if k >= len(sims):
        idx = np.argsort(-sims)
    else:
        idx = np.argpartition(-sims, k)[:k]
        idx = idx[np.argsort(-sims[idx])]
    return [(int(i), float(sims[i])) for i in idx]


def build_cache_paths(jurisdiction: str, cache_dir: Path) -> Tuple[Path, Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    vec_path = cache_dir / f"embeddings_{jurisdiction}.npz"
    meta_path = cache_dir / f"embeddings_{jurisdiction}.meta.json"
    return vec_path, meta_path


def load_embedding_cache(jurisdiction: str, cache_dir: Path) -> Tuple[np.ndarray, List[Clause], Dict[str, Any]]:
    vec_path, meta_path = build_cache_paths(jurisdiction, cache_dir)
    if not vec_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"Embedding cache not found for {jurisdiction}. "
            f"Run scripts/build_embeddings.py first."
        )
    mat = np.load(vec_path)["embeddings"].astype(np.float32)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    clauses = [Clause(**c) for c in meta["clauses"]]
    info = meta.get("info", {})
    return mat, clauses, info


def search_clauses(
    query: str,
    jurisdiction: str,
    cache_dir: str | Path = "data",
    top_k: int = 5,
    model: str = "intfloat/multilingual-e5-small",
) -> List[Match]:
    """
    Zero-shot semantic search over clause corpus using local embeddings.
    Requires cache built by scripts/build_embeddings.py.
    """
    cache_dir = Path(cache_dir)
    mat, clauses, _info = load_embedding_cache(jurisdiction, cache_dir)

    # E5 query format
    q_emb = embed_texts_local([f"query: {query}"], model=model)[0]
    q = np.array(q_emb, dtype=np.float32)

    top = cosine_top_k(q, mat, k=top_k)
    return [Match(clause=clauses[i], score=score) for i, score in top]
