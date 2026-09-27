"""
Code chunking — Phase 5 of the project spec.

Splits a source file into logical chunks (functions, classes, methods,
plus whatever module-level code is left over) instead of fixed-size text
windows, so each retrieved chunk is a complete, meaningful unit of code.
Python uses the `ast` module for exact boundaries; anything else (or
unparsable Python) falls back to one whole-file chunk until a real
multi-language parser is added later.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass


@dataclass
class CodeChunk:
    file_path: str
    kind: str  # "module" | "function" | "class" | "method" | "file"
    name: str
    start_line: int
    end_line: int
    code: str

    @property
    def chunk_id(self) -> str:
        return f"{self.file_path}::{self.name}:{self.start_line}-{self.end_line}"


def chunk_file(file_path: str, source: str) -> list[CodeChunk]:
    """Entry point: dispatches by extension. Non-Python files are chunked whole for now."""
    if file_path.endswith(".py"):
        return chunk_python_file(file_path, source)
    return [_whole_file_chunk(file_path, source)]


def chunk_python_file(file_path: str, source: str) -> list[CodeChunk]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return [_whole_file_chunk(file_path, source)]

    lines = source.splitlines()
    if not lines:
        return [_whole_file_chunk(file_path, source)]

    chunks: list[CodeChunk] = []
    remaining_lines = set(range(1, len(lines) + 1))

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            chunks.append(_node_chunk(file_path, "function", node.name, node, lines))
            _mark_covered(remaining_lines, node)

        elif isinstance(node, ast.ClassDef):
            chunks.append(_node_chunk(file_path, "class", node.name, node, lines))
            _mark_covered(remaining_lines, node)
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    chunks.append(
                        _node_chunk(
                            file_path, "method", f"{node.name}.{child.name}", child, lines
                        )
                    )

    # Whatever's left at module level (imports, constants, top-level statements)
    # becomes its own chunk, so nothing is silently dropped from retrieval.
    ordered_remaining = sorted(remaining_lines)
    if ordered_remaining and any(lines[i - 1].strip() for i in ordered_remaining):
        module_code = "\n".join(lines[i - 1] for i in ordered_remaining)
        chunks.append(
            CodeChunk(
                file_path=file_path,
                kind="module",
                name="<module-level>",
                start_line=ordered_remaining[0],
                end_line=ordered_remaining[-1],
                code=module_code,
            )
        )

    return chunks or [_whole_file_chunk(file_path, source)]


def _node_chunk(file_path: str, kind: str, name: str, node: ast.AST, lines: list[str]) -> CodeChunk:
    start = node.lineno
    end = getattr(node, "end_lineno", start)
    code = "\n".join(lines[start - 1 : end])
    return CodeChunk(
        file_path=file_path, kind=kind, name=name, start_line=start, end_line=end, code=code
    )


def _mark_covered(remaining: set[int], node: ast.AST) -> None:
    start = node.lineno
    end = getattr(node, "end_lineno", start)
    for i in range(start, end + 1):
        remaining.discard(i)


def _whole_file_chunk(file_path: str, source: str) -> CodeChunk:
    line_count = source.count("\n") + 1
    return CodeChunk(
        file_path=file_path,
        kind="file",
        name="<whole-file>",
        start_line=1,
        end_line=line_count,
        code=source,
    )