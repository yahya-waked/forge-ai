"""
BM25 keyword search — Phase 5 of the project spec.

A lightweight, in-memory keyword index over the same CodeChunks the
vector store holds. Combined with vector search in hybrid_search.py:
BM25 reliably catches exact identifier/string matches that embeddings
sometimes miss; vector search catches semantic matches that pure keyword
search misses.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from app.retrieval.chunking import CodeChunk

_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_SUBWORD_RE = re.compile(r"[A-Z]?[a-z0-9]+|[A-Z]+(?![a-z])")


def _tokenize(text: str) -> list[str]:
    # Also split snake_case/camelCase identifiers into their parts, so
    # searching "expired token" finds `is_token_expired` / `isTokenExpired`.
    tokens: list[str] = []
    for word in _WORD_RE.findall(text):
        tokens.append(word.lower())
        parts = _SUBWORD_RE.findall(word)
        if len(parts) > 1:
            tokens.extend(p.lower() for p in parts)
    return tokens


@dataclass
class BM25Hit:
    chunk_id: str
    score: float


class BM25Index:
    """Rebuilt in memory each time a repository is (re-)indexed — cheap enough
    for repo-sized corpora, and avoids keeping a second on-disk index in sync
    with ChromaDB."""

    def __init__(self) -> None:
        self._chunk_ids: list[str] = []
        self._bm25: BM25Okapi | None = None

    def build(self, chunks: list[CodeChunk]) -> None:
        self._chunk_ids = [c.chunk_id for c in chunks]
        tokenized = [_tokenize(c.code) for c in chunks]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None

    def query(self, query: str, top_k: int = 10) -> list[BM25Hit]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(zip(self._chunk_ids, scores), key=lambda pair: pair[1], reverse=True)
        return [BM25Hit(chunk_id=cid, score=score) for cid, score in ranked[:top_k] if score > 0]