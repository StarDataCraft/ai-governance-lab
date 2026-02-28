# scripts/build_embeddings.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

from risk_engine.semantic_search import load_clauses_jsonl, build_cache_paths
from risk_engine.embeddings import embed_texts_local


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--jur", required=True, help="Jurisdiction code, e.g., EU, US, JP")
    p.add_argument("--corpus_dir", default="law_corpus", help="Path to law_corpus directory")
    p.add_argument("--cache_dir", default="data", help="Path to cache directory")
    p.add_argument("--model", default="intfloat/multilingual-e5-small", help="Local embedding model name")
    args = p.parse_args()

    jur = args.jur.strip()
    corpus_dir = Path(args.corpus_dir)
    cache_dir = Path(args.cache_dir)

    clauses_path = corpus_dir / jur / "clauses.jsonl"
    clauses = load_clauses_jsonl(clauses_path)

    # E5 passage format: "passage: ..."
    texts = []
    for c in clauses:
        chunk = f"{c.title}\n\n{c.text}".strip()
        texts.append(f"passage: {chunk}")

    embs = embed_texts_local(texts, model=args.model)
    mat = np.array(embs, dtype=np.float32)  # already normalized

    vec_path, meta_path = build_cache_paths(jur, cache_dir)
    np.savez_compressed(vec_path, embeddings=mat)

    meta: Dict[str, Any] = {
        "info": {
            "jurisdiction": jur,
            "embedding_model": args.model,
            "num_clauses": len(clauses),
        },
        "clauses": [c.__dict__ for c in clauses],
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"✅ Saved embeddings to: {vec_path}")
    print(f"✅ Saved metadata to:  {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
