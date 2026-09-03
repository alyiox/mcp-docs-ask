"""Configured docs collection discovery tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from mcp_docs_ask.config import config_from_dict
from mcp_docs_ask.embedder import HashEmbedder
from mcp_docs_ask.tools.list_docs import list_docs_impl
from mcp_docs_ask.tools.reindex import reindex_impl


def test_list_docs_returns_sanitized_layers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MCP_DOCS_ASK_CACHE", str(tmp_path / "cache"))
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
                "index": None,
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
                "index": None,
            },
        ],
    }
    # Never echoes the configured source string itself (it can carry credentials).
    dumped = str(result)
    assert "example.invalid" not in dumped
    assert "/private/local" not in dumped


def test_list_docs_reports_built_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MCP_DOCS_ASK_CACHE", str(tmp_path / "cache"))
    docs = tmp_path / "docs"
    (docs / "guides").mkdir(parents=True)
    (docs / "guides" / "a.md").write_text("# A\n\nAlpha body text.\n", encoding="utf-8")
    config = config_from_dict(
        {
            "docs": {
                "product": {
                    "source": str(docs),
                    "layers": {"guides": {"include": ["guides/**"]}},
                }
            },
            "default": {"docs": "product", "embedding_model": "hash-embedder/v1"},
        }
    )
    reindex_impl(config, HashEmbedder(model_name="hash-embedder/v1"))

    entry = list_docs_impl(config)["docs"][0]
    assert entry["index"]["origin"] == "path"
    assert entry["index"]["file_count"] == 1
    assert entry["index"]["chunk_count"] > 0
    assert entry["index"]["layers"] == ["guides"]
    assert entry["index"]["embedding_model"] == "hash-embedder/v1"
    # Root is disclosed so a host can build full paths from citation paths.
    assert entry["index"]["root"] == str(docs.resolve())
