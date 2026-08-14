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
                },
                "flat": {
                    "source": "/private/local/docs",
                    "desc": "Flat local docs",
                },
            },
            "default_docs": "product",
        }
    )

    result = list_docs_impl(config)

    assert result == {
        "default_docs": "product",
        "docs": [
            {
                "id": "flat",
                "desc": "Flat local docs",
                "default": False,
                "layers": [],
                "layer_filters": ["all", "other"],
            },
            {
                "id": "product",
                "desc": "Product guides and API reference",
                "default": True,
                "layers": [
                    {"id": "guides", "desc": "Guides"},
                    {"id": "api", "desc": "API"},
                ],
                "layer_filters": ["all", "guides", "api", "other"],
            },
        ],
    }
    # Must not leak private source paths/URLs.
    dumped = str(result)
    assert "example.invalid" not in dumped
    assert "/private/local" not in dumped
