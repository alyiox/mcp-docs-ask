"""Configured docs collection discovery tests."""

from __future__ import annotations

from mcp_docs_ask.config import config_from_dict
from mcp_docs_ask.tools.list_docs import list_docs_impl


def test_list_docs_returns_sanitized_layer_filters() -> None:
    config = config_from_dict(
        {
            "docs": {
                "product": {
                    "source": "https://example.invalid/private/docs.git",
                    "desc": "Product guides and API reference",
                    "layers": {
                        "guides": {
                            "desc": "Guides",
                            "include": ["docs/guides/**"],
                        },
                        "api": {
                            "desc": "API",
                            "include": ["docs/api/**"],
                        },
                    },
                    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                    "top_k": 3,
                },
                "flat": {
                    "source": "/private/local/docs",
                    "desc": "Flat local docs",
                },
            },
            "default": {
                "docs": "product",
                "embedding_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                "top_k": 8,
                "chunk_max_chars": 1500,
            },
        }
    )

    result = list_docs_impl(config)

    assert result == {
        "default": {
            "docs": "product",
            "embedding_model": ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
            "top_k": 8,
            "chunk_max_chars": 1500,
        },
        "docs": [
            {
                "id": "flat",
                "desc": "Flat local docs",
                "default": False,
                "embedding_model": ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
                "top_k": 8,
                "chunk_max_chars": 1500,
                "layers": [],
                "layer_filters": ["all"],
            },
            {
                "id": "product",
                "desc": "Product guides and API reference",
                "default": True,
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "top_k": 3,
                "chunk_max_chars": 1500,
                "layers": [
                    {"id": "guides", "desc": "Guides"},
                    {"id": "api", "desc": "API"},
                ],
                "layer_filters": ["all", "guides", "api"],
            },
        ],
    }
    # Must not leak private source paths/URLs.
    dumped = str(result)
    assert "example.invalid" not in dumped
    assert "/private/local" not in dumped
