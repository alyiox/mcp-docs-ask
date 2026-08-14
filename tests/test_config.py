"""Config loading tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mcp_docs_ask.config import (
    DEFAULT_CHUNK_MAX_CHARS,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_TOP_K,
    LayerEntry,
    load_config,
)


def _write(path: Path, data: dict) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_load_config(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "config.json",
        {
            "docs": {
                "product": {
                    "source": "https://github.com/example/docs.git",
                    "desc": "Example docs",
                    "ref": "main",
                    "include": ["docs/**/*.md"],
                    "layers": {
                        "API": {
                            "desc": "API reference",
                            "include": ["docs/api/**"],
                        },
                        "guides": {
                            "include": ["docs/guides/**"],
                        },
                    },
                    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                    "top_k": 3,
                }
            },
            "default": {
                "docs": "product",
                "embedding_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                "top_k": 6,
                "chunk_max_chars": 1200,
            },
        },
    )
    cfg = load_config(path)
    assert cfg.default_docs == "product"
    assert cfg.top_k == 6
    assert cfg.chunk_max_chars == 1200
    assert cfg.embedding_model == ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    assert cfg.top_k_for("product") == 3
    assert cfg.embedding_model_for("product") == ("sentence-transformers/all-MiniLM-L6-v2")
    assert cfg.chunk_max_chars_for("product") == 1200
    entry = cfg.doc("product")
    assert entry.include == ("docs/**/*.md",)
    assert entry.desc == "Example docs"
    assert entry.layers == {
        "api": LayerEntry(include=("docs/api/**",), desc="API reference"),
        "guides": LayerEntry(include=("docs/guides/**",), desc=""),
    }


def test_load_config_inherits_defaults(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "config.json",
        {
            "docs": {"notes": {"source": "/docs"}},
            "default": {"docs": "notes"},
        },
    )
    cfg = load_config(path)
    assert cfg.default_docs == "notes"
    assert cfg.embedding_model == DEFAULT_EMBEDDING_MODEL
    assert cfg.top_k == DEFAULT_TOP_K
    assert cfg.chunk_max_chars == DEFAULT_CHUNK_MAX_CHARS
    assert cfg.embedding_model_for("notes") == DEFAULT_EMBEDDING_MODEL
    assert cfg.top_k_for("notes") == DEFAULT_TOP_K
    assert cfg.chunk_max_chars_for("notes") == DEFAULT_CHUNK_MAX_CHARS


def test_load_config_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.json")


def test_load_config_requires_default(tmp_path: Path) -> None:
    path = _write(tmp_path / "config.json", {"docs": {"default": {"source": "/docs"}}})
    with pytest.raises(ValueError, match="config.default"):
        load_config(path)


def test_load_config_defaults_to_all_markdown_without_layers(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "config.json",
        {
            "docs": {"default": {"source": "/docs"}},
            "default": {"docs": "default"},
        },
    )
    entry = load_config(path).doc()
    assert entry.include == ("**/*.md",)
    assert entry.layers == {}
    assert entry.desc == ""


def test_load_config_rejects_reserved_layer_name(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "config.json",
        {
            "docs": {
                "default": {
                    "source": "/docs",
                    "layers": {"all": {"include": ["docs/**"]}},
                }
            },
            "default": {"docs": "default"},
        },
    )
    with pytest.raises(ValueError, match="reserved"):
        load_config(path)


def test_load_config_rejects_layer_list_instead_of_object(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "config.json",
        {
            "docs": {
                "default": {
                    "source": "/docs",
                    "layers": {"guides": ["docs/guides/**"]},
                }
            },
            "default": {"docs": "default"},
        },
    )
    with pytest.raises(ValueError, match="must be an object"):
        load_config(path)


def test_load_config_requires_source(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "config.json",
        {
            "docs": {"default": {"include": ["**/*.md"]}},
            "default": {"docs": "default"},
        },
    )
    with pytest.raises(ValueError, match="source"):
        load_config(path)


def test_load_config_rejects_invalid_per_docs_top_k(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "config.json",
        {
            "docs": {"default": {"source": "/docs", "top_k": 0}},
            "default": {"docs": "default"},
        },
    )
    with pytest.raises(ValueError, match="top_k"):
        load_config(path)
