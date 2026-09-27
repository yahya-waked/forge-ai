"""
Embeddings provider — Phase 5 of the project spec.

Wraps Gemini's embedding endpoint behind the same "swap the backend
later" philosophy as app/llm/factory.py: retrieval code calls
embed_documents()/embed_query() and never imports google.genai directly.
"""

from __future__ import annotations

import os

from google import genai
from google.genai import types
from tenacity import retry, retry_if_exception, stop_after_attempt

from app.llm.retry_utils import is_retryable_genai_error, wait_for_genai_retry

DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"
# Gemini's embeddings support Matryoshka truncation: a smaller dimensionality
# keeps ChromaDB storage/compute down while still being a strong retrieval
# embedding. Query and document vectors must use the same dimensionality.
DEFAULT_DIMENSIONALITY = 768
_BATCH_SIZE = 100  # embed_content accepts a batch of texts per call


class EmbeddingProvider:
    def __init__(
        self,
        model: str = DEFAULT_EMBEDDING_MODEL,
        dimensionality: int = DEFAULT_DIMENSIONALITY,
        api_key: str | None = None,
    ) -> None:
        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._dimensionality = dimensionality

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """One embedding vector per input text (same order), for text being indexed."""
        return self._embed(texts, task_type="RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> list[float]:
        """Embedding for a search query — a different task_type than documents,
        which measurably improves retrieval quality for Gemini's embeddings."""
        return self._embed([text], task_type="RETRIEVAL_QUERY")[0]

    @retry(
        retry=retry_if_exception(is_retryable_genai_error),
        wait=wait_for_genai_retry,
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def _call_api(self, batch: list[str], task_type: str) -> list[list[float]]:
        response = self._client.models.embed_content(
            model=self._model,
            contents=batch,
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=self._dimensionality,
            ),
        )
        return [embedding.values for embedding in response.embeddings]

    def _embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), _BATCH_SIZE):
            batch = texts[start : start + _BATCH_SIZE]
            vectors.extend(self._call_api(batch, task_type))
        return vectors