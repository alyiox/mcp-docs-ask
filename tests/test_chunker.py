"""Chunker unit tests."""

from __future__ import annotations

from mcp_docs_ask.chunker import chunk_markdown, infer_layer, strip_html_comments


def test_infer_layer_without_rules_is_other() -> None:
    assert infer_layer("docs/guides/getting-started.md") == "other"
    assert infer_layer("docs/api/users.md") == "other"
    assert infer_layer("README.md") == "other"


def test_infer_configured_layer_first_match_wins() -> None:
    layers = {
        "guides": ("docs/guides/**",),
        "api": ("docs/api/**",),
        "reference": ("docs/**",),
    }
    assert infer_layer("docs/guides/getting-started.md", layers) == "guides"
    assert infer_layer("docs/api/users/get.md", layers) == "api"
    assert infer_layer("docs/reference/start.md", layers) == "reference"
    assert infer_layer("README.md", layers) == "other"
    assert infer_layer("docs/guides/a.md", {}) == "other"


def test_strip_html_comments() -> None:
    text = "before\n<!--\nsecret\n-->\nafter"
    assert "secret" not in strip_html_comments(text)
    assert "before" in strip_html_comments(text)
    assert "after" in strip_html_comments(text)


def test_chunk_by_heading() -> None:
    md = """# Home Overview

Preamble paragraph.

## Overview

The home page helps users see business health at a glance. Supports Summary panel.

## Scenarios

| Scenario | Panel |
|----------|--------|
| Big picture | Summary |
"""
    layers = {"guides": ("docs/guides/**",)}
    chunks = chunk_markdown(
        "docs/guides/home-overview.md",
        md,
        chunk_max_chars=1500,
        layers=layers,
    )
    assert any(c.heading == "Overview" for c in chunks)
    assert any("business health" in c.body for c in chunks)
    assert all(c.layer == "guides" for c in chunks)
    assert all("docs/guides" in c.embed_text for c in chunks)


def test_chunk_api_layer() -> None:
    md = """# API Overview

## API mapping

| Function | url |
|----------|-----|
| getSummary | Dashboard |
"""
    layers = {"api": ("docs/api/**",)}
    chunks = chunk_markdown("docs/api/overview.md", md, layers=layers)
    assert chunks
    assert chunks[0].layer == "api"
