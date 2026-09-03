"""List configured documentation collections, layer filters, and index state."""

from __future__ import annotations

from typing import Any

from ..config import Config, DocsEntry
from ..index import index_summary, load_meta
from ..sync import docs_location


def _index_state(docs_id: str, entry: DocsEntry) -> dict[str, Any] | None:
    """Built-index state for a collection, or None when never indexed.

    ``root`` is the local checkout, so a host can join it with a citation path
    without asking a question first. Resolved without cloning or fetching.
    """
    meta = load_meta(docs_id)
    if meta is None:
        return None
    try:
        root: str | None = str(docs_location(docs_id, entry)[0])
    except FileNotFoundError:
        # Index outlived its source dir; report state without a root.
        root = None
    return index_summary(meta, root)


def list_docs_impl(config: Config) -> dict[str, Any]:
    docs: list[dict[str, Any]] = []
    for docs_id in sorted(config.docs):
        entry = config.docs[docs_id]
        configured_layers = [
            {"id": name, "desc": layer.desc} for name, layer in entry.layers.items()
        ]
        docs.append(
            {
                "id": docs_id,
                "desc": entry.desc,
                "default": docs_id == config.default_docs,
                "embedding_model": config.embedding_model_for(docs_id),
                "top_k": config.top_k_for(docs_id),
                "chunk_max_chars": config.chunk_max_chars_for(docs_id),
                "layers": configured_layers,
                "index": _index_state(docs_id, entry),
            }
        )
    return {
        # Mirrors the config `default` block so agents read one shape everywhere.
        "default": {
            "docs": config.default_docs,
            "embedding_model": config.embedding_model,
            "top_k": config.top_k,
            "chunk_max_chars": config.chunk_max_chars,
        },
        "docs": docs,
    }
