"""Rebuild docs index (and git-sync when source is a URL)."""

from __future__ import annotations

from typing import Any

from ..config import Config
from ..embedder import Embedder
from ..index import build_index
from ..sync import resolve_docs_root


def reindex_impl(
    config: Config,
    embedder: Embedder,
    *,
    docs: str | None = None,
) -> dict[str, Any]:
    docs_id = docs or config.default_docs
    cfg = config.doc(docs_id)
    checkout = resolve_docs_root(docs_id, cfg, update=True)
    meta = build_index(
        docs_id,
        checkout,
        cfg,
        embedder,
        chunk_max_chars=config.chunk_max_chars_for(docs_id),
        force=True,
    )
    return {
        "docs": docs_id,
        "docs_rev": meta.docs_rev,
        "docs_source": meta.docs_source,
        "chunk_count": meta.chunk_count,
        "indexed_files": len(meta.files),
        "layers": meta.layers,
        "embedding_model": meta.embedding_model,
        "fingerprint": meta.fingerprint,
        "status": "ok",
    }
