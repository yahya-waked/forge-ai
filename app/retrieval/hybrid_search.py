"""
Hybrid Search — Phase 5 of the project spec.

Combines BM25 (keyword) and vector (semantic) search over the same set
of CodeChunks using Reciprocal Rank Fusion (RRF): a chunk's final score
is the sum of 1/(k + rank) across both rankings it appears in. RRF needs
no score normalization between BM25 and cosine distance, which is why
it's the standard way to combine two heterogeneous retrieval signals.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from app.retrieval.bm25 import BM25Index
from app.retrieval.chunking import CodeChunk, chunk_file
from app.retrieval.embeddings import EmbeddingProvider
from app.retrieval.vector_store import VectorStore

_RRF_K = 60  # standard constant from the original RRF paper; rarely needs tuning

_IGNORED_DIRS = {
    ".git", "__pycache__", ".venv", "venv", "node_modules",
    ".pytest_cache", ".ruff_cache", ".forgeai",
}
_INDEXABLE_EXTENSIONS = {".py"}  # widen as chunk_file() learns more languages
_MAX_FILE_BYTES = 500_000


@dataclass
class SearchResult:
    file_path: str
    name: str
    kind: str
    start_line: int
    end_line: int
    code: str
    score: float


class CodeIndex:
    """One CodeIndex per repository: owns the chunk registry, the vector
    store, and the BM25 index — the only thing hybrid search needs."""

    def __init__(self, repo_root: str, embeddings: EmbeddingProvider | None = None) -> None:
        self.repo_root = os.path.abspath(repo_root)
        self.embeddings = embeddings or EmbeddingProvider()
        self.vector_store = VectorStore(self.repo_root)
        self.bm25 = BM25Index()
        self._chunks_by_id: dict[str, CodeChunk] = {}

    def build(self) -> int:
        """Walks the repository, chunks every indexable file, embeds every
        chunk, and populates both the vector store and BM25 index. Returns
        the number of chunks indexed."""
        chunks = [c for c in self._collect_chunks() if c.code.strip()]
        self._chunks_by_id = {c.chunk_id: c for c in chunks}

        self.vector_store.clear()
        if chunks:
            vectors = self.embeddings.embed_documents([c.code for c in chunks])
            self.vector_store.upsert(chunks, vectors)

        self.bm25.build(chunks)
        return len(chunks)

    def search(self, query: str, top_k: int = 8) -> list[SearchResult]:
        if not self._chunks_by_id:
            raise RuntimeError("Index is empty — call build() before searching")

        query_vector = self.embeddings.embed_query(query)
        vector_hits = self.vector_store.query(query_vector, top_k=top_k * 3)
        bm25_hits = self.bm25.query(query, top_k=top_k * 3)

        rrf_scores: dict[str, float] = {}
        for rank, hit in enumerate(vector_hits):
            rrf_scores[hit.chunk_id] = rrf_scores.get(hit.chunk_id, 0.0) + 1.0 / (_RRF_K + rank + 1)
        for rank, hit in enumerate(bm25_hits):
            rrf_scores[hit.chunk_id] = rrf_scores.get(hit.chunk_id, 0.0) + 1.0 / (_RRF_K + rank + 1)

        ranked_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)[:top_k]

        results: list[SearchResult] = []
        for chunk_id in ranked_ids:
            chunk = self._chunks_by_id.get(chunk_id)
            if chunk is None:
                continue  # stale id — never let that crash retrieval
            results.append(
                SearchResult(
                    file_path=chunk.file_path,
                    name=chunk.name,
                    kind=chunk.kind,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    code=chunk.code,
                    score=rrf_scores[chunk_id],
                )
            )
        return results

    def _collect_chunks(self) -> list[CodeChunk]:
        chunks: list[CodeChunk] = []
        for current_root, dirs, files in os.walk(self.repo_root):
            dirs[:] = [d for d in dirs if d not in _IGNORED_DIRS]
            for name in files:
                if not any(name.endswith(ext) for ext in _INDEXABLE_EXTENSIONS):
                    continue
                full_path = os.path.join(current_root, name)
                if os.path.getsize(full_path) > _MAX_FILE_BYTES:
                    continue
                rel_path = os.path.relpath(full_path, self.repo_root)
                try:
                    with open(full_path, "r", encoding="utf-8") as f:
                        source = f.read()
                except (UnicodeDecodeError, OSError):
                    continue
                chunks.extend(chunk_file(rel_path, source))
        return chunks