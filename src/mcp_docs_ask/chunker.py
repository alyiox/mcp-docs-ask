"""Markdown heading chunker for documentation."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)

# Sentinel for paths that match no named-layer glob (not exposed as a tool filter).
UNNAMED_LAYER = ""
# Joins ancestor headings into a chunk's heading path.
HEADING_SEP = " > "


@dataclass(frozen=True)
class Chunk:
    path: str  # repo-relative posix path
    heading: str  # ancestor headings joined by HEADING_SEP; "" for preamble
    body: str
    layer: str  # named layer, or UNNAMED_LAYER if no layer glob matched
    seq: int = 0  # position within the file, so split parts can be rejoined in order

    @property
    def search_blob(self) -> str:
        """Text that gets embedded and ASCII-boosted."""
        return f"{self.path}\n{self.heading}\n{self.body}"


def infer_layer(
    rel_path: str,
    layers: Mapping[str, tuple[str, ...]] | None = None,
) -> str:
    posix = PurePosixPath(rel_path.replace("\\", "/"))
    for name, patterns in (layers or {}).items():
        if any(posix.full_match(pattern) for pattern in patterns):
            return name
    return UNNAMED_LAYER


def strip_html_comments(text: str) -> str:
    return _HTML_COMMENT_RE.sub("", text)


def _split_oversized(body: str, max_chars: int) -> list[str]:
    if len(body) <= max_chars:
        return [body] if body.strip() else []
    parts: list[str] = []
    start = 0
    while start < len(body):
        end = min(start + max_chars, len(body))
        if end < len(body):
            # Prefer breaking on newline
            nl = body.rfind("\n", start, end)
            if nl > start + max_chars // 2:
                end = nl + 1
        chunk = body[start:end].strip()
        if chunk:
            parts.append(chunk)
        start = end
    return parts


def chunk_markdown(
    rel_path: str,
    text: str,
    *,
    chunk_max_chars: int = 1500,
    layers: Mapping[str, tuple[str, ...]] | None = None,
) -> list[Chunk]:
    cleaned = strip_html_comments(text)
    layer = infer_layer(rel_path, layers)
    matches = list(_HEADING_RE.finditer(cleaned))
    chunks: list[Chunk] = []

    def add(heading: str, body: str) -> None:
        for part in _split_oversized(body, chunk_max_chars):
            chunks.append(
                Chunk(path=rel_path, heading=heading, body=part, layer=layer, seq=len(chunks))
            )

    if not matches:
        add("", cleaned.strip())
        return chunks

    # Preamble before first heading
    add("", cleaned[: matches[0].start()].strip())

    # (level, title) of the current heading and its ancestors
    stack: list[tuple[int, str]] = []
    for i, match in enumerate(matches):
        level = len(match.group(1))
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, match.group(2).strip()))
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(cleaned)
        add(HEADING_SEP.join(title for _, title in stack), cleaned[start:end].strip())
    return chunks


def chunk_file(
    repo_root: Path,
    file_path: Path,
    *,
    chunk_max_chars: int = 1500,
    layers: Mapping[str, tuple[str, ...]] | None = None,
) -> list[Chunk]:
    rel = file_path.resolve().relative_to(repo_root.resolve()).as_posix()
    text = file_path.read_text(encoding="utf-8")
    return chunk_markdown(rel, text, chunk_max_chars=chunk_max_chars, layers=layers)
