"""Config loading tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mcp_docs_ask.config import LayerEntry, load_config


def test_load_config(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "docs": {
                    "default": {
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
                    }
                },
                "default_docs": "default",
                "top_k": 6,
            }
        ),
        encoding="utf-8",
    )
    cfg = load_config(path)
    assert cfg.default_docs == "default"
    assert cfg.top_k == 6
    entry = cfg.doc("default")
    assert entry.include == ("docs/**/*.md",)
    assert entry.desc == "Example docs"
    assert entry.layers == {
        "api": LayerEntry(include=("docs/api/**",), desc="API reference"),
        "guides": LayerEntry(include=("docs/guides/**",), desc=""),
    }


def test_load_config_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.json")


def test_load_config_defaults_to_all_markdown_without_layers(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps({"docs": {"default": {"source": "/docs"}}}),
        encoding="utf-8",
    )
    entry = load_config(path).doc()
    assert entry.include == ("**/*.md",)
    assert entry.layers == {}
    assert entry.desc == ""


def test_load_config_rejects_reserved_layer_name(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "docs": {
                    "default": {
                        "source": "/docs",
                        "layers": {"all": {"include": ["docs/**"]}},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="reserved"):
        load_config(path)


def test_load_config_rejects_layer_list_instead_of_object(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "docs": {
                    "default": {
                        "source": "/docs",
                        "layers": {"guides": ["docs/guides/**"]},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="must be an object"):
        load_config(path)


def test_load_config_requires_source(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps({"docs": {"default": {"include": ["**/*.md"]}}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="source"):
        load_config(path)
