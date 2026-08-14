"""ask_docs / reindex tool impl tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from mcp_docs_ask.config import config_from_dict
from mcp_docs_ask.embedder import HashEmbedder
from mcp_docs_ask.tools.ask_docs import ask_docs_impl
from mcp_docs_ask.tools.reindex import reindex_impl


def _fixture_docs(root: Path) -> None:
    guide = root / "docs" / "guides"
    guide.mkdir(parents=True)
    (guide / "preferences.md").write_text(
        """# Preferences

## Overview

The Preferences page summarizes account settings. It is read-only.

## List

No checkboxes and no Action column.
""",
        encoding="utf-8",
    )
    api = root / "docs" / "api"
    api.mkdir(parents=True)
    (api / "preferences.md").write_text(
        """# Preferences API

## Preferences list

Route PreferencesList → views/Preferences/index.vue
""",
        encoding="utf-8",
    )


def test_ask_docs_returns_citations(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MCP_DOCS_ASK_CACHE", str(tmp_path / "cache"))
    docs = tmp_path / "docs-repo"
    _fixture_docs(docs)
    config = config_from_dict(
        {
            "docs": {
                "default": {
                    "source": str(docs),
                    "include": ["docs/**/*.md"],
                    "layers": {
                        "guides": {"include": ["docs/guides/**"]},
                        "api": {"include": ["docs/api/**"]},
                    },
                }
            },
            "default_docs": "default",
            "embedding_model": "hash-embedder/v1",
            "top_k": 5,
        }
    )
    embedder = HashEmbedder(model_name="hash-embedder/v1")
    result = ask_docs_impl(
        config,
        embedder,
        question="Is the Preferences page read-only? Does it have an Action column?",
        layer="guides",
    )
    assert "citations" in result
    assert result["citations"]
    assert "answer_context" in result
    assert result["docs"] == "default"
    assert result["layer"] == "guides"
    assert result["available_layers"] == ["api", "guides"]
    assert "synthesize" in result["note"].lower() or "Retrieval" in result["note"]


def test_ask_docs_configured_layer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MCP_DOCS_ASK_CACHE", str(tmp_path / "cache"))
    docs = tmp_path / "docs-repo"
    _fixture_docs(docs)
    config = config_from_dict(
        {
            "docs": {
                "default": {
                    "source": str(docs),
                    "layers": {"api": {"include": ["docs/api/**"]}},
                }
            },
            "embedding_model": "hash-embedder/v1",
        }
    )
    embedder = HashEmbedder(model_name="hash-embedder/v1")
    result = ask_docs_impl(
        config,
        embedder,
        question="PreferencesList route entry",
        layer="api",
    )
    assert result["citations"]
    assert all(c["layer"] == "api" for c in result["citations"])


def test_ask_docs_rejects_unknown_layer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MCP_DOCS_ASK_CACHE", str(tmp_path / "cache"))
    docs = tmp_path / "docs-repo"
    _fixture_docs(docs)
    config = config_from_dict(
        {
            "docs": {
                "default": {
                    "source": str(docs),
                    "layers": {"guides": {"include": ["docs/guides/**"]}},
                }
            },
            "embedding_model": "hash-embedder/v1",
        }
    )
    with pytest.raises(ValueError, match="all, guides, other"):
        ask_docs_impl(
            config,
            HashEmbedder(model_name="hash-embedder/v1"),
            question="preferences",
            layer="missing",
        )


def test_ask_docs_empty_question(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MCP_DOCS_ASK_CACHE", str(tmp_path / "cache"))
    docs = tmp_path / "docs"
    _fixture_docs(docs)
    config = config_from_dict(
        {
            "docs": {"default": {"source": str(docs)}},
            "embedding_model": "hash-embedder/v1",
        }
    )
    with pytest.raises(ValueError, match="question"):
        ask_docs_impl(config, HashEmbedder(), question="  ")


def test_reindex_local(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MCP_DOCS_ASK_CACHE", str(tmp_path / "cache"))
    docs = tmp_path / "docs"
    _fixture_docs(docs)
    config = config_from_dict(
        {
            "docs": {"default": {"source": str(docs)}},
            "embedding_model": "hash-embedder/v1",
        }
    )
    out = reindex_impl(config, HashEmbedder(model_name="hash-embedder/v1"))
    assert out["status"] == "ok"
    assert out["docs"] == "default"
    assert out["chunk_count"] > 0


def test_config_missing_docs_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MCP_DOCS_ASK_CACHE", str(tmp_path / "cache"))
    missing = tmp_path / "missing-docs"
    config = config_from_dict(
        {
            "docs": {"default": {"source": str(missing)}},
            "embedding_model": "hash-embedder/v1",
        }
    )
    with pytest.raises(FileNotFoundError):
        ask_docs_impl(
            config,
            HashEmbedder(model_name="hash-embedder/v1"),
            question="anything",
        )
