"""
Vector store — Phase 5 of the project spec.

Thin wrapper around a local, persistent ChromaDB collection: stores each
CodeChunk's embedding plus enough metadata to map a hit straight back to
a real file and line range.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import chromadb

from app.retrieval.chunking import CodeChunk

_DB_DIR = ".forgeai/chroma"  # relative to the repo root; add this to .gitignore
_COLLECTION_NAME = "code_chunks"


@dataclass
class VectorHit:
    chunk_id: str
    file_path: str
    name: str
    kind: str
    start_line: int
    end_line: int
    code: str
    distance: float


class VectorStore:
    def __init__(self, repo_root: str) -> None:
        db_path = os.path.join(repo_root, _DB_DIR)
        os.makedirs(db_path, exist_ok=True)
        self._client = chromadb.PersistentClient(path=db_path)
        self._collection = self._client.get_or_create_collection(name=_COLLECTION_NAME)

    def clear(self) -> None:
        """Drop everything indexed so far (used when re-indexing from scratch)."""
        self._client.delete_collection(_COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(name=_COLLECTION_NAME)

    def upsert(self, chunks: list[CodeChunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.code for c in chunks],
            metadatas=[
                {
                    "file_path": c.file_path,
                    "name": c.name,
                    "kind": c.kind,
                    "start_line": c.start_line,
                    "end_line": c.end_line,
                }
                for c in chunks
            ],
        )

    def delete_file(self, file_path: str) -> None:
        """Remove every chunk belonging to one file, before re-indexing it."""
        self._collection.delete(where={"file_path": file_path})

    def query(self, query_embedding: list[float], top_k: int = 10) -> list[VectorHit]:
        if self.count() == 0:
            return []

        result = self._collection.query(query_embeddings=[query_embedding], n_results=top_k)
        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        return [
            VectorHit(
                chunk_id=chunk_id,
                file_path=meta["file_path"],
                name=meta["name"],
                kind=meta["kind"],
                start_line=meta["start_line"],
                end_line=meta["end_line"],
                code=code,
                distance=distance,
            )
            for chunk_id, code, meta, distance in zip(ids, documents, metadatas, distances)
        ]

    def count(self) -> int:
        return self._collection.count()