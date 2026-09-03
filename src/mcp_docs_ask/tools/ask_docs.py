"""Tool implementations for ask_docs."""

from __future__ import annotations

from typing import Any

from ..config import Config
from ..embedder import Embedder
from ..index import (
    build_index,
    format_answer_context,
    hits_to_citations,
    search,
)
from ..sync import resolve_docs_root


def ask_docs_impl(
    config: Config,
    embedder: Embedder,
    *,
    question: str,
    docs: str | None = None,
    top_k: int | None = None,
    layer: str = "all",
) -> dict[str, Any]:
    if not question or not question.strip():
        raise ValueError("question must be a non-empty string")

    docs_id = docs or config.default_docs
    cfg = config.doc(docs_id)
    layer_norm = layer.strip().lower() if layer else "all"
    allowed_layers = {"all", *cfg.layers}
    if layer_norm not in allowed_layers:
        known = ", ".join(sorted(allowed_layers))
        raise ValueError(f"layer must be one of: {known}")

    k = top_k if top_k is not None else config.top_k_for(docs_id)
    if k < 1:
        raise ValueError("top_k must be >= 1")

    checkout = resolve_docs_root(docs_id, cfg, update=False)

    build_index(
        docs_id,
        checkout,
        cfg,
        embedder,
        chunk_max_chars=config.chunk_max_chars_for(docs_id),
        force=False,
    )

    hits, meta = search(
        docs_id,
        question.strip(),
        embedder,
        top_k=k,
        layer=layer_norm,
    )
    return {
        "docs": docs_id,
        "layer": layer_norm,
        # Answer-scoped provenance only: root to reach the files, rev to trust
        # them. Collection state lives on list_docs / reindex.
        "index": {
            "root": str(checkout.root),
            "rev": meta.rev or checkout.rev,
        },
        "answer_context": format_answer_context(hits),
        "citations": hits_to_citations(hits),
        "note": "Answer only from answer_context; do not invent facts.",
    }
