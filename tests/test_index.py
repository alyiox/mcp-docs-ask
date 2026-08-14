"""Index build / search with HashEmbedder."""

from __future__ import annotations

from pathlib import Path

import pytest

from mcp_docs_ask.config import DocsEntry, LayerEntry
from mcp_docs_ask.embedder import HashEmbedder
from mcp_docs_ask.index import (
    ASCII_BOOST_CAP,
    CHUNKS_NAME,
    META_NAME,
    VECTORS_NAME,
    ascii_boost,
    ascii_tokens,
    build_index,
    index_dir,
    search,
)
from mcp_docs_ask.sync import DocsCheckout

LAYERS = {
    "guides": LayerEntry(include=("docs/guides/**",)),
    "api": LayerEntry(include=("docs/api/**",)),
}


def _write_fixture_repo(root: Path) -> None:
    guides = root / "docs" / "guides"
    guides.mkdir(parents=True)
    (guides / "home-overview.md").write_text(
        """# Home Overview

## Overview

The home page helps users see business health at a glance. It is a read-only board.

## Panels

Summary / Efficiency / Awareness KPI groups.
""",
        encoding="utf-8",
    )
    api = root / "docs" / "api"
    api.mkdir(parents=True)
    (api / "overview.md").write_text(
        """# API — Overview

## Entry points

Route `/Preferences` opens `views/Preferences/index.vue`.

## API mapping

OrderList loads from api/report/orders.
getSummary → OrderController.
""",
        encoding="utf-8",
    )


def test_ascii_tokens() -> None:
    toks = ascii_tokens("Where is OrderList and OrderController at /Preferences?")
    assert "OrderList" in toks
    assert "OrderController" in toks
    assert "/Preferences" in toks


def test_ascii_boost_is_capped() -> None:
    tokens = ["OrderList", "OrderController", "UserController", "AuthService", "FooBar"]
    blob = " ".join(tokens)
    boost = ascii_boost(tokens, blob, "path/OrderList.md")
    assert boost == ASCII_BOOST_CAP
    assert boost <= 0.30


def test_atomic_write_publishes_complete_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MCP_DOCS_ASK_CACHE", str(tmp_path / "cache"))
    root = tmp_path / "docs"
    _write_fixture_repo(root)
    embedder = HashEmbedder()
    cfg = DocsEntry(source=str(root), layers=LAYERS)
    checkout = DocsCheckout(root=root, source="path", docs_rev=None)
    meta = build_index(
        "default",
        checkout,
        cfg,
        embedder,
        chunk_max_chars=1500,
        force=True,
    )
    dest = index_dir("default")
    assert (dest / META_NAME).is_file()
    assert (dest / CHUNKS_NAME).is_file()
    assert (dest / VECTORS_NAME).is_file()
    assert not any(dest.glob(".staging-*"))
    assert meta.chunk_count > 0
    assert meta.docs == "default"


def test_build_and_search_guides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MCP_DOCS_ASK_CACHE", str(tmp_path / "cache"))
    root = tmp_path / "docs"
    _write_fixture_repo(root)
    embedder = HashEmbedder()
    cfg = DocsEntry(source=str(root), layers=LAYERS)
    checkout = DocsCheckout(root=root, source="path", docs_rev=None)
    meta = build_index(
        "default",
        checkout,
        cfg,
        embedder,
        chunk_max_chars=1500,
        force=True,
    )
    assert meta.chunk_count > 0
    assert any("guides" in f for f in meta.files)
    assert meta.layers == ["api", "guides"]

    hits, _ = search(
        "default",
        "home page business health at a glance read-only board",
        embedder,
        top_k=5,
        layer="guides",
    )
    assert hits
    assert all(h.layer == "guides" for h in hits)
    assert any("home-overview" in h.path for h in hits)


def test_ascii_boost_api_layer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MCP_DOCS_ASK_CACHE", str(tmp_path / "cache"))
    root = tmp_path / "docs"
    _write_fixture_repo(root)
    embedder = HashEmbedder()
    cfg = DocsEntry(source=str(root), layers=LAYERS)
    checkout = DocsCheckout(root=root, source="path", docs_rev=None)
    build_index("default", checkout, cfg, embedder, chunk_max_chars=1500, force=True)

    hits, _ = search(
        "default",
        "OrderList OrderController",
        embedder,
        top_k=5,
        layer="api",
    )
    assert hits
    assert all(h.layer == "api" for h in hits)
    assert any("OrderList" in h.snippet or "OrderController" in h.body for h in hits)


def test_layer_all_may_mix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MCP_DOCS_ASK_CACHE", str(tmp_path / "cache"))
    root = tmp_path / "docs"
    _write_fixture_repo(root)
    embedder = HashEmbedder()
    cfg = DocsEntry(source=str(root), layers=LAYERS)
    checkout = DocsCheckout(root=root, source="path", docs_rev=None)
    build_index("default", checkout, cfg, embedder, chunk_max_chars=1500, force=True)
    hits, _ = search("default", "Home Overview Panels", embedder, top_k=8, layer="all")
    assert hits
