"""
Code RAG tools — Phase 5 of the project spec, exposed to the agent.

index_repository builds (or rebuilds) the hybrid search index for a
repository; search_code_semantic queries it. Indexes are kept in memory
per repo_root for the process's lifetime — cheap enough for a single
dev/agent run — while the underlying ChromaDB collection is persisted
to disk, so re-indexing an unchanged repo is optional, not required.
"""

from __future__ import annotations

from app.retrieval.hybrid_search import CodeIndex
from app.tools.registry import RiskLevel, ToolError, registry

_indexes: dict[str, CodeIndex] = {}


def _get_index(repo_root: str) -> CodeIndex:
    if repo_root not in _indexes:
        raise ToolError(
            f"Repository '{repo_root}' has not been indexed yet — call index_repository first"
        )
    return _indexes[repo_root]


@registry.register(
    name="index_repository",
    description=(
        "Build (or rebuild) the semantic + keyword search index for the repository. "
        "Call this once at the start of a task, before using search_code_semantic — "
        "without it, search_code_semantic has nothing to search."
    ),
    parameters={
        "type": "object",
        "properties": {
            "repo_root": {
                "type": "string",
                "description": "Absolute path to the repository root on disk.",
            },
        },
        "required": ["repo_root"],
    },
    risk=RiskLevel.LOW,
)
def index_repository(repo_root: str) -> str:
    index = CodeIndex(repo_root)
    chunk_count = index.build()
    _indexes[repo_root] = index
    return f"Indexed {chunk_count} code chunks from '{repo_root}'"


@registry.register(
    name="search_code_semantic",
    description=(
        "Search the indexed repository by meaning, not just exact text — finds relevant "
        "functions/classes even when they don't contain your exact search words (e.g. "
        "'where is authentication handled' can find a matching function with no literal "
        "text match). Combines keyword and semantic search. Requires index_repository to "
        "have been called first this run. Prefer plain search_code for an exact string/regex match."
    ),
    parameters={
        "type": "object",
        "properties": {
            "repo_root": {
                "type": "string",
                "description": "Absolute path to the repository root on disk.",
            },
            "query": {
                "type": "string",
                "description": "Natural-language or keyword description of what you're looking for.",
            },
            "top_k": {
                "type": "integer",
                "description": "How many results to return.",
                "default": 8,
            },
        },
        "required": ["repo_root", "query"],
    },
    risk=RiskLevel.LOW,
)
def search_code_semantic(repo_root: str, query: str, top_k: int = 8) -> str:
    index = _get_index(repo_root)
    results = index.search(query, top_k=top_k)
    if not results:
        return "No relevant code found."

    blocks = [
        f"{r.file_path}:{r.start_line}-{r.end_line} [{r.kind} {r.name}] (score={r.score:.4f})\n{r.code}"
        for r in results
    ]
    return "\n---\n".join(blocks)