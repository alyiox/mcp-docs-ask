"""Chunker unit tests."""

from __future__ import annotations

from mcp_docs_ask.chunker import (
    HEADING_SEP,
    UNNAMED_LAYER,
    chunk_markdown,
    infer_layer,
    strip_html_comments,
)


def test_infer_layer_without_rules_is_unnamed() -> None:
    assert infer_layer("docs/guides/getting-started.md") == UNNAMED_LAYER
    assert infer_layer("docs/api/users.md") == UNNAMED_LAYER
    assert infer_layer("README.md") == UNNAMED_LAYER


def test_infer_configured_layer_first_match_wins() -> None:
    layers = {
        "guides": ("docs/guides/**",),
        "api": ("docs/api/**",),
        "reference": ("docs/**",),
    }
    assert infer_layer("docs/guides/getting-started.md", layers) == "guides"
    assert infer_layer("docs/api/users/get.md", layers) == "api"
    assert infer_layer("docs/reference/start.md", layers) == "reference"
    assert infer_layer("README.md", layers) == UNNAMED_LAYER
    assert infer_layer("docs/guides/a.md", {}) == UNNAMED_LAYER


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
    assert any(c.heading == f"Home Overview{HEADING_SEP}Overview" for c in chunks)
    assert any("business health" in c.body for c in chunks)
    assert all(c.layer == "guides" for c in chunks)
    assert all("docs/guides" in c.search_blob for c in chunks)


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


def test_heading_path_tracks_ancestors() -> None:
    md = """# Guide

## Configuration

Intro.

### Layers

Layer text.

### Embedding

Model text.

## Usage

Usage text.

# Appendix

Appendix text.
"""
    headings = [c.heading for c in chunk_markdown("guide.md", md)]
    assert headings == [
        "Guide > Configuration",
        "Guide > Configuration > Layers",
        "Guide > Configuration > Embedding",
        "Guide > Usage",
        "Appendix",
    ]


def test_heading_path_skips_levels() -> None:
    md = "# Top\n\n### Deep\n\nDeep text.\n\n## Mid\n\nMid text.\n"
    headings = [c.heading for c in chunk_markdown("doc.md", md)]
    assert headings == ["Top > Deep", "Top > Mid"]


def test_split_parts_share_heading_and_number_in_order() -> None:
    body = "\n".join(f"line {i:03d} " + "x" * 40 for i in range(20))
    md = f"Preamble.\n\n# Big\n\n{body}\n\n# Next\n\nTail.\n"
    chunks = chunk_markdown("big.md", md, chunk_max_chars=300)
    assert [c.seq for c in chunks] == list(range(len(chunks)))
    big = [c for c in chunks if c.heading == "Big"]
    assert len(big) > 1
    assert "\n".join(c.body for c in big) == body
    assert chunks[0].heading == ""
    assert chunks[-1].heading == "Next"


def test_hash_lines_in_code_fences_are_not_headings() -> None:
    md = """# Guide

## Install

```bash
# install deps
uv sync
```

~~~
## not a heading either
~~~

## Usage

Run it.
"""
    chunks = chunk_markdown("guide.md", md)
    assert [c.heading for c in chunks] == ["Guide > Install", "Guide > Usage"]
    assert "# install deps" in chunks[0].body
    assert "## not a heading either" in chunks[0].body


def test_unclosed_fence_swallows_rest_of_file() -> None:
    md = "# Guide\n\n```\n# inside\n\n## still inside\n"
    assert [c.heading for c in chunk_markdown("guide.md", md)] == ["Guide"]
