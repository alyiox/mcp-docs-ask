"""List configured documentation collections and layer filters."""

from __future__ import annotations

from typing import Any

from ..config import Config


def list_docs_impl(config: Config) -> dict[str, Any]:
    docs: list[dict[str, Any]] = []
    for docs_id in sorted(config.docs):
        entry = config.docs[docs_id]
        configured_layers = [
            {"id": name, "desc": layer.desc} for name, layer in entry.layers.items()
        ]
        layer_ids = [layer["id"] for layer in configured_layers]
        docs.append(
            {
                "id": docs_id,
                "desc": entry.desc,
                "default": docs_id == config.default_docs,
                "embedding_model": config.embedding_model_for(docs_id),
                "top_k": config.top_k_for(docs_id),
                "chunk_max_chars": config.chunk_max_chars_for(docs_id),
                "layers": configured_layers,
                "layer_filters": ["all", *layer_ids, "other"],
            }
        )
    return {
        "default_docs": config.default_docs,
        "docs": docs,
    }
