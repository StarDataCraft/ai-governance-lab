# risk_engine/embeddings.py
from __future__ import annotations

from functools import lru_cache
from typing import List

from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=4)
def _get_model(model_name: str) -> SentenceTransformer:
    # Will auto-download from HuggingFace on first run (no API key needed)
    return SentenceTransformer(model_name)


def embed_texts_local(texts: List[str], model: str = "intfloat/multilingual-e5-small") -> List[List[float]]:
    """
    Local open-source embeddings via sentence-transformers.

    Good default for JP/EN mixed governance docs:
      intfloat/multilingual-e5-small

    Notes:
    - E5 style works best if you prefix query/document, but it's optional.
    - We'll prefix automatically for better retrieval quality.
    """
    m = _get_model(model)

    # E5 recommended formatting
    # query: "query: ...", doc: "passage: ..."
    # Here we assume caller passes docs; for queries we handle in semantic_search.
    embs = m.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
    return embs.tolist()
