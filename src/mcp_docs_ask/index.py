"""Local vector index: build, load, search with ASCII-token boost."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from .chunker import UNNAMED_LAYER, Chunk, chunk_file
from .config import DocsEntry, indexes_dir
from .embedder import Embedder
from .sync import DocsCheckout

logger = logging.getLogger(__name__)

_ASCII_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_./-]{2,}|/[A-Za-z][A-Za-z0-9_/-]*")
# Separator, digit, or a capital past the first letter: OrderList, API_KEY,
# config.json, /Preferences. Plain words ("configure", "Where") don't qualify.
_IDENTIFIER_HINT_RE = re.compile(r"[_./0-9-]|.[A-Z]")
META_NAME = "meta.json"
CHUNKS_NAME = "chunks.json"
VECTORS_NAME = "vectors.npy"
SCORE_FLOOR = 0.05
ASCII_BOOST = 0.15
ASCII_BOOST_CAP = 0.30
CANDIDATE_MULT = 4
# Below this much room left in the budget, drop the next passage rather than
# trim it to a stub.
MIN_TRIMMED_CHARS = 200
GAP_MARKER = "\n\n…\n\n"
TRIM_MARKER = "\n…"
# Bump when the on-disk chunk or meta shape changes: it feeds the fingerprint,
# so a stale index rebuilds instead of failing to parse.
INDEX_FORMAT = "3"


@dataclass
class SearchHit:
    path: str
    heading: str
    layer: str
    score: float
    text: str
    seq: int = 0


@dataclass
class IndexMeta:
    docs: str
    embedding_model: str
    fingerprint: str
    rev: str | None
    origin: str
    chunk_count: int
    files: list[str]
    layers: list[str] = field(default_factory=list)


def _fingerprint(
    files: list[Path],
    embedding_model: str,
    chunk_max_chars: int,
    layers: dict[str, tuple[str, ...]],
) -> str:
    h = hashlib.sha256()
    h.update(INDEX_FORMAT.encode())
    h.update(embedding_model.encode())
    h.update(str(chunk_max_chars).encode())
    h.update(json.dumps(layers, sort_keys=True).encode())
    for path in sorted(files, key=lambda p: str(p).lower()):
        stat = path.stat()
        h.update(str(path).encode())
        h.update(str(stat.st_mtime_ns).encode())
        h.update(str(stat.st_size).encode())
    return h.hexdigest()[:32]


def _match_globs(repo_root: Path, include: tuple[str, ...], exclude: tuple[str, ...]) -> list[Path]:
    found: set[Path] = set()
    for pattern in include:
        found.update(repo_root.glob(pattern))
    excluded: set[Path] = set()
    for pattern in exclude:
        excluded.update(repo_root.glob(pattern))
    files = [
        p.resolve()
        for p in found
        if p.is_file() and p.suffix.lower() == ".md" and p.resolve() not in excluded
    ]
    return sorted(files, key=lambda p: p.as_posix().lower())


def index_dir(docs_id: str) -> Path:
    return indexes_dir() / docs_id


def load_meta(docs_id: str) -> IndexMeta | None:
    path = index_dir(docs_id) / META_NAME
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    try:
        return IndexMeta(**data)
    except TypeError:
        # Written by a different version of this dataclass; rebuild instead.
        return None


def _write_index(
    docs_id: str,
    chunks: list[Chunk],
    vectors: np.ndarray,
    meta: IndexMeta,
) -> None:
    """Atomically publish index files via a staging dir + replace."""
    dest = index_dir(docs_id)
    dest.mkdir(parents=True, exist_ok=True)
    staging = dest / f".staging-{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)
    try:
        chunk_dicts = [asdict(c) for c in chunks]
        (staging / CHUNKS_NAME).write_text(
            json.dumps(chunk_dicts, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        np.save(staging / VECTORS_NAME, vectors)
        (staging / META_NAME).write_text(
            json.dumps(asdict(meta), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        for name in (CHUNKS_NAME, VECTORS_NAME, META_NAME):
            os.replace(staging / name, dest / name)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def _load_chunks_vectors(docs_id: str) -> tuple[list[Chunk], np.ndarray]:
    dest = index_dir(docs_id)
    raw = json.loads((dest / CHUNKS_NAME).read_text(encoding="utf-8"))
    chunks = [Chunk(**item) for item in raw]
    vectors = np.load(dest / VECTORS_NAME)
    return chunks, vectors


def build_index(
    docs_id: str,
    checkout: DocsCheckout,
    cfg: DocsEntry,
    embedder: Embedder,
    *,
    chunk_max_chars: int,
    force: bool = False,
) -> IndexMeta:
    files = _match_globs(checkout.root, cfg.include, cfg.exclude)
    if not files:
        raise FileNotFoundError(
            f"No markdown files matched include={list(cfg.include)} under {checkout.root}. "
            "Check docs source path and include globs."
        )

    layer_patterns = cfg.layer_patterns()
    fp = _fingerprint(files, embedder.model_name, chunk_max_chars, layer_patterns)
    existing = load_meta(docs_id)
    if (
        not force
        and existing is not None
        and existing.fingerprint == fp
        and existing.embedding_model == embedder.model_name
        and (index_dir(docs_id) / VECTORS_NAME).is_file()
    ):
        logger.info("Index for %s is up to date (%d files)", docs_id, len(files))
        return existing

    chunks: list[Chunk] = []
    for path in files:
        chunks.extend(
            chunk_file(
                checkout.root,
                path,
                chunk_max_chars=chunk_max_chars,
                layers=layer_patterns,
            )
        )

    if not chunks:
        raise RuntimeError(f"No chunks produced from {len(files)} files under {checkout.root}")

    logger.info("Embedding %d chunks for docs %s", len(chunks), docs_id)
    batch_size = 64
    vectors_list: list[np.ndarray] = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        vectors_list.append(embedder.embed([c.search_blob for c in batch]))
    vectors = np.vstack(vectors_list)

    rel_files = [p.relative_to(checkout.root).as_posix() for p in files]
    meta = IndexMeta(
        docs=docs_id,
        embedding_model=embedder.model_name,
        fingerprint=fp,
        rev=checkout.rev,
        origin=checkout.origin,
        chunk_count=len(chunks),
        files=rel_files,
        layers=sorted({chunk.layer for chunk in chunks if chunk.layer != UNNAMED_LAYER}),
    )
    _write_index(docs_id, chunks, vectors, meta)
    return meta


def ascii_tokens(query: str) -> list[str]:
    """Identifier-like tokens in ``query``, deduplicated in order."""
    tokens: list[str] = []
    for raw in _ASCII_TOKEN_RE.findall(query):
        # Drop sentence punctuation the token pattern swallows ("config.json.")
        token = raw.rstrip(".-")
        if len(token) >= 3 and _IDENTIFIER_HINT_RE.search(token):
            tokens.append(token)
    return list(dict.fromkeys(tokens))


def ascii_boost(tokens: list[str], blob: str, path: str) -> float:
    """Light identifier boost, capped so it cannot swamp cosine scores."""
    matches = sum(1 for tok in tokens if tok in blob or tok in path)
    return min(ASCII_BOOST_CAP, matches * ASCII_BOOST)


def search(
    docs_id: str,
    question: str,
    embedder: Embedder,
    *,
    top_k: int = 8,
    layer: str = "all",
) -> tuple[list[SearchHit], IndexMeta]:
    meta = load_meta(docs_id)
    if meta is None:
        raise FileNotFoundError(
            f"No index for docs {docs_id!r}. Call reindex first "
            f"(and ensure config docs.{docs_id}.source is set)."
        )
    if meta.embedding_model != embedder.model_name:
        raise RuntimeError(
            f"Index embedding_model={meta.embedding_model!r} != "
            f"config {embedder.model_name!r}. Call reindex."
        )

    chunks, vectors = _load_chunks_vectors(docs_id)
    if layer != "all":
        indices = [i for i, c in enumerate(chunks) if c.layer == layer]
        if not indices:
            return [], meta
        chunks_f = [chunks[i] for i in indices]
        vectors_f = vectors[indices]
    else:
        chunks_f = chunks
        vectors_f = vectors

    q = embedder.embed([question])[0]
    scores = vectors_f @ q  # cosine if rows normalized
    candidate_n = min(len(chunks_f), max(top_k * CANDIDATE_MULT, top_k))
    if candidate_n == 0:
        return [], meta

    top_idx = np.argpartition(-scores, candidate_n - 1)[:candidate_n]
    tokens = ascii_tokens(question)
    ranked: list[tuple[float, int]] = []
    for i in top_idx:
        score = float(scores[i])
        chunk = chunks_f[int(i)]
        boost = ascii_boost(tokens, chunk.search_blob, chunk.path)
        ranked.append((score + boost, int(i)))

    ranked.sort(key=lambda x: x[0], reverse=True)
    hits: list[SearchHit] = []
    for score, i in ranked:
        # Soft floor: skip weak hits only after we already have at least one
        if hits and score < SCORE_FLOOR:
            break
        chunk = chunks_f[i]
        hits.append(
            SearchHit(
                path=chunk.path,
                heading=chunk.heading,
                layer=chunk.layer,
                score=round(score, 4),
                text=chunk.body,
                seq=chunk.seq,
            )
        )
        if len(hits) >= top_k:
            break
    return hits, meta


def _merge_sections(hits: list[SearchHit]) -> list[SearchHit]:
    """Join hits from the same file section into one passage, in file order.

    A section keeps the score of its best part; sections stay best-first.
    """
    groups: dict[tuple[str, str], list[SearchHit]] = {}
    for hit in hits:
        groups.setdefault((hit.path, hit.heading), []).append(hit)
    merged: list[SearchHit] = []
    for parts in groups.values():
        best = parts[0]
        parts.sort(key=lambda h: h.seq)
        text = parts[0].text
        for prev, cur in zip(parts, parts[1:], strict=False):
            # Adjacent parts were one block before splitting; mark skipped ones.
            text += ("\n" if cur.seq == prev.seq + 1 else GAP_MARKER) + cur.text
        merged.append(replace(best, text=text, seq=parts[0].seq))
    return merged


def _fit_budget(hits: list[SearchHit], max_chars: int) -> list[SearchHit]:
    """Keep passages best-first until ``max_chars`` of text is used.

    The passage that overflows is trimmed at a line break and ends the list;
    the first passage is always kept, trimmed if need be.
    """
    kept: list[SearchHit] = []
    used = 0
    for hit in hits:
        remaining = max_chars - used
        if len(hit.text) <= remaining:
            kept.append(hit)
            used += len(hit.text)
            continue
        if kept and remaining < MIN_TRIMMED_CHARS:
            break
        cut = hit.text.rfind("\n", 0, remaining)
        if cut < remaining // 2:
            cut = remaining
        kept.append(replace(hit, text=hit.text[:cut].rstrip() + TRIM_MARKER))
        break
    return kept


def assemble_context(hits: list[SearchHit], *, max_chars: int) -> list[SearchHit]:
    """Turn ranked hits into the passages handed to the host."""
    return _fit_budget(_merge_sections(hits), max_chars)


def format_answer_context(hits: list[SearchHit]) -> str:
    if not hits:
        return (
            "No relevant passages found in the indexed docs. "
            "Try rephrase, another layer, or call reindex."
        )
    parts: list[str] = []
    for i, hit in enumerate(hits, start=1):
        heading = hit.heading or "(preamble)"
        layer_part = f" · layer={hit.layer}" if hit.layer != UNNAMED_LAYER else ""
        parts.append(f"[{i}] {hit.path} · {heading}{layer_part} · score={hit.score}\n{hit.text}")
    return "\n\n".join(parts)


def hits_to_citations(hits: list[SearchHit]) -> list[dict[str, Any]]:
    """Machine-readable pointers. ``n`` matches the ``[n]`` marker in
    ``format_answer_context``; passage text lives only there, never twice.
    """
    return [
        {
            "n": i,
            "path": h.path,
            "heading": h.heading,
            "layer": h.layer,
            "score": h.score,
        }
        for i, h in enumerate(hits, start=1)
    ]


def index_summary(meta: IndexMeta, root: str | None) -> dict[str, Any]:
    """Index provenance, one shape for ``list_docs`` and ``reindex``."""
    return {
        "origin": meta.origin,
        "root": root,
        "rev": meta.rev,
        "files": len(meta.files),
        "chunks": meta.chunk_count,
        "layers": meta.layers,
    }
