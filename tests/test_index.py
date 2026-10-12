"""Index build / search with HashEmbedder."""

from __future__ import annotations

from pathlib import Path

import pytest

import mcp_docs_ask.index as index_module
from mcp_docs_ask.config import DocsEntry, LayerEntry
from mcp_docs_ask.embedder import HashEmbedder
from mcp_docs_ask.index import (
    ASCII_BOOST_CAP,
    CHUNKS_NAME,
    GAP_MARKER,
    META_NAME,
    TRIM_MARKER,
    VECTORS_NAME,
    SearchHit,
    ascii_boost,
    ascii_tokens,
    assemble_context,
    build_index,
    index_dir,
    load_meta,
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
    checkout = DocsCheckout(root=root, origin="file", rev=None)
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
    checkout = DocsCheckout(root=root, origin="file", rev=None)
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
    checkout = DocsCheckout(root=root, origin="file", rev=None)
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
    assert any("OrderList" in h.text or "OrderController" in h.text for h in hits)


def test_layer_all_may_mix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MCP_DOCS_ASK_CACHE", str(tmp_path / "cache"))
    root = tmp_path / "docs"
    _write_fixture_repo(root)
    embedder = HashEmbedder()
    cfg = DocsEntry(source=str(root), layers=LAYERS)
    checkout = DocsCheckout(root=root, origin="file", rev=None)
    build_index("default", checkout, cfg, embedder, chunk_max_chars=1500, force=True)
    hits, _ = search("default", "Home Overview Panels", embedder, top_k=8, layer="all")
    assert hits


def test_index_format_change_triggers_rebuild(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MCP_DOCS_ASK_CACHE", str(tmp_path / "cache"))
    root = tmp_path / "docs"
    _write_fixture_repo(root)
    embedder = HashEmbedder()
    cfg = DocsEntry(source=str(root), layers=LAYERS)
    checkout = DocsCheckout(root=root, origin="file", rev=None)
    monkeypatch.setattr(index_module, "INDEX_FORMAT", "old")
    old = build_index("default", checkout, cfg, embedder, chunk_max_chars=1500)
    monkeypatch.undo()
    monkeypatch.setenv("MCP_DOCS_ASK_CACHE", str(tmp_path / "cache"))
    new = build_index("default", checkout, cfg, embedder, chunk_max_chars=1500)
    assert new.fingerprint != old.fingerprint
    assert load_meta("default") == new


def _hit(path: str, heading: str, score: float, text: str, seq: int) -> SearchHit:
    return SearchHit(path=path, heading=heading, layer="", score=score, text=text, seq=seq)


def test_search_returns_full_body(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MCP_DOCS_ASK_CACHE", str(tmp_path / "cache"))
    root = tmp_path / "docs"
    root.mkdir()
    body = "\n".join(f"- step {i}: run `cmd-{i}` and check the output" for i in range(25))
    (root / "setup.md").write_text(f"# Setup\n\n{body}\n", encoding="utf-8")
    embedder = HashEmbedder()
    checkout = DocsCheckout(root=root, origin="file", rev=None)
    build_index("default", checkout, DocsEntry(source=str(root)), embedder, chunk_max_chars=1500)
    hits, _ = search("default", "setup steps", embedder, top_k=1)
    assert len(body) > 400
    assert hits[0].text == body


def test_assemble_merges_parts_of_one_section_in_order() -> None:
    hits = [
        _hit("a.md", "A > Big", 0.9, "part two", 2),
        _hit("b.md", "B", 0.8, "other", 0),
        _hit("a.md", "A > Big", 0.7, "part one", 1),
        _hit("a.md", "A > Big", 0.6, "part four", 4),
    ]
    out = assemble_context(hits, max_chars=8000)
    assert [(h.path, h.score) for h in out] == [("a.md", 0.9), ("b.md", 0.8)]
    assert out[0].text == f"part one\npart two{GAP_MARKER}part four"


def test_assemble_keeps_same_heading_in_different_files_apart() -> None:
    hits = [_hit("a.md", "Usage", 0.9, "a", 0), _hit("b.md", "Usage", 0.8, "b", 0)]
    assert [h.text for h in assemble_context(hits, max_chars=8000)] == ["a", "b"]


def test_assemble_trims_last_passage_at_line_break() -> None:
    first = "x" * 600
    second = "\n".join(["line " + "y" * 45] * 20)
    hits = [_hit("a.md", "A", 0.9, first, 0), _hit("b.md", "B", 0.8, second, 0)]
    out = assemble_context(hits, max_chars=1000)
    assert out[0].text == first
    assert out[1].text.endswith(TRIM_MARKER)
    kept = out[1].text.removesuffix(TRIM_MARKER)
    assert second.startswith(kept)
    assert kept.endswith("y")
    assert len(first) + len(kept) <= 1000


def test_assemble_drops_passage_that_would_be_a_stub() -> None:
    hits = [_hit("a.md", "A", 0.9, "x" * 950, 0), _hit("b.md", "B", 0.8, "y" * 500, 0)]
    out = assemble_context(hits, max_chars=1000)
    assert [h.path for h in out] == ["a.md"]


def test_assemble_always_keeps_first_passage() -> None:
    hits = [_hit("a.md", "A", 0.9, "z" * 2000, 0)]
    out = assemble_context(hits, max_chars=500)
    assert len(out) == 1
    assert out[0].text == "z" * 500 + TRIM_MARKER
