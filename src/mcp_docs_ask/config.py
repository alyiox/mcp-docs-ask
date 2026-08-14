"""Load ~/.config/mcp-docs-ask/config.json."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CONFIG_DIR = Path.home() / ".config" / "mcp-docs-ask"
CONFIG_PATH = CONFIG_DIR / "config.json"


def _cache_dir() -> Path:
    override = os.environ.get("MCP_DOCS_ASK_CACHE")
    if override:
        return Path(override).expanduser().resolve()
    return Path.home() / ".cache" / "mcp-docs-ask"


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_INCLUDE = ["**/*.md"]
DEFAULT_TOP_K = 8
DEFAULT_CHUNK_MAX_CHARS = 1500
DEFAULT_REF = "main"


@dataclass(frozen=True)
class LayerEntry:
    include: tuple[str, ...]
    desc: str = ""


@dataclass(frozen=True)
class DocsEntry:
    source: str
    desc: str = ""
    ref: str = DEFAULT_REF
    include: tuple[str, ...] = tuple(DEFAULT_INCLUDE)
    exclude: tuple[str, ...] = ()
    layers: dict[str, LayerEntry] = field(default_factory=dict)
    embedding_model: str | None = None
    top_k: int | None = None
    chunk_max_chars: int | None = None

    def layer_patterns(self) -> dict[str, tuple[str, ...]]:
        """Layer name → include globs (for chunking / fingerprint)."""
        return {name: entry.include for name, entry in self.layers.items()}


@dataclass(frozen=True)
class Config:
    docs: dict[str, DocsEntry]
    default_docs: str
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    top_k: int = DEFAULT_TOP_K
    chunk_max_chars: int = DEFAULT_CHUNK_MAX_CHARS

    def doc(self, name: str | None = None) -> DocsEntry:
        key = name or self.default_docs
        if key not in self.docs:
            known = ", ".join(sorted(self.docs)) or "(none)"
            raise KeyError(f"Unknown docs id {key!r}. Configured: {known}")
        return self.docs[key]

    def embedding_model_for(self, docs_id: str | None = None) -> str:
        entry = self.doc(docs_id)
        return entry.embedding_model or self.embedding_model

    def top_k_for(self, docs_id: str | None = None) -> int:
        entry = self.doc(docs_id)
        return entry.top_k if entry.top_k is not None else self.top_k

    def chunk_max_chars_for(self, docs_id: str | None = None) -> int:
        entry = self.doc(docs_id)
        return entry.chunk_max_chars if entry.chunk_max_chars is not None else self.chunk_max_chars


def default_cache_dir() -> Path:
    return _cache_dir()


def repos_dir() -> Path:
    return default_cache_dir() / "repos"


def indexes_dir() -> Path:
    return default_cache_dir() / "indexes"


def _parse_desc(raw: Any, field_name: str) -> str:
    if raw is None:
        return ""
    if not isinstance(raw, str):
        raise ValueError(f"{field_name} must be a string")
    return raw.strip()


def _parse_layers(raw: Any) -> dict[str, LayerEntry]:
    if not isinstance(raw, dict):
        raise ValueError("docs.layers must be an object mapping layer names to layer objects")
    layers: dict[str, LayerEntry] = {}
    for name, value in raw.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("docs.layers names must be non-empty strings")
        normalized = name.strip().lower()
        if normalized == "all":
            raise ValueError(f"docs.layers name {normalized!r} is reserved")
        if normalized in layers:
            raise ValueError(f"duplicate docs.layers name after normalization: {normalized!r}")
        if not isinstance(value, dict):
            raise ValueError(f"docs.layers.{name} must be an object with 'include'")
        patterns = value.get("include")
        if (
            not isinstance(patterns, (list, tuple))
            or not patterns
            or not all(isinstance(pattern, str) and pattern.strip() for pattern in patterns)
        ):
            raise ValueError(f"docs.layers.{name}.include must be a non-empty list of glob strings")
        layers[normalized] = LayerEntry(
            include=tuple(pattern.strip().replace("\\", "/") for pattern in patterns),
            desc=_parse_desc(value.get("desc"), f"docs.layers.{name}.desc"),
        )
    return layers


def _parse_optional_embedding_model(raw: Any, field_name: str) -> str | None:
    if raw is None:
        return None
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return raw.strip()


def _parse_optional_top_k(raw: Any, field_name: str) -> int | None:
    if raw is None:
        return None
    top_k = int(raw)
    if top_k < 1:
        raise ValueError(f"{field_name} must be >= 1")
    return top_k


def _parse_optional_chunk_max_chars(raw: Any, field_name: str) -> int | None:
    if raw is None:
        return None
    chunk_max_chars = int(raw)
    if chunk_max_chars < 100:
        raise ValueError(f"{field_name} must be >= 100")
    return chunk_max_chars


def _parse_docs_entry(raw: dict[str, Any], docs_id: str) -> DocsEntry:
    source = raw.get("source")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("docs.<id>.source must be a non-empty string (path or git URL)")
    include = raw.get("include", DEFAULT_INCLUDE)
    exclude = raw.get("exclude", [])
    if not isinstance(include, list) or not all(isinstance(x, str) for x in include):
        raise ValueError("docs.<id>.include must be a list of strings")
    if not isinstance(exclude, list) or not all(isinstance(x, str) for x in exclude):
        raise ValueError("docs.<id>.exclude must be a list of strings")
    ref = raw.get("ref", DEFAULT_REF)
    if not isinstance(ref, str) or not ref.strip():
        raise ValueError("docs.<id>.ref must be a non-empty string")
    layers = _parse_layers(raw.get("layers", {}))
    prefix = f"docs.{docs_id}"
    return DocsEntry(
        source=source.strip(),
        desc=_parse_desc(raw.get("desc"), f"{prefix}.desc"),
        ref=ref.strip(),
        include=tuple(include),
        exclude=tuple(exclude),
        layers=layers,
        embedding_model=_parse_optional_embedding_model(
            raw.get("embedding_model"), f"{prefix}.embedding_model"
        ),
        top_k=_parse_optional_top_k(raw.get("top_k"), f"{prefix}.top_k"),
        chunk_max_chars=_parse_optional_chunk_max_chars(
            raw.get("chunk_max_chars"), f"{prefix}.chunk_max_chars"
        ),
    )


def _parse_default_block(raw: Any, docs: dict[str, DocsEntry]) -> tuple[str, str, int, int]:
    if not isinstance(raw, dict):
        raise ValueError("config.default must be an object")
    default_docs = raw.get("docs")
    if not isinstance(default_docs, str) or not default_docs.strip():
        raise ValueError("default.docs must be a non-empty string")
    default_docs = default_docs.strip()
    if default_docs not in docs:
        raise ValueError(f"default.docs {default_docs!r} not in docs {sorted(docs)}")

    embedding_model = raw.get("embedding_model", DEFAULT_EMBEDDING_MODEL)
    if not isinstance(embedding_model, str) or not embedding_model.strip():
        raise ValueError("default.embedding_model must be a non-empty string")

    top_k = _parse_optional_top_k(raw.get("top_k"), "default.top_k")
    if top_k is None:
        top_k = DEFAULT_TOP_K
    chunk_max_chars = _parse_optional_chunk_max_chars(
        raw.get("chunk_max_chars"), "default.chunk_max_chars"
    )
    if chunk_max_chars is None:
        chunk_max_chars = DEFAULT_CHUNK_MAX_CHARS

    return default_docs, embedding_model.strip(), top_k, chunk_max_chars


def _config_from_data(data: dict[str, Any]) -> Config:
    docs_raw = data.get("docs")
    if not isinstance(docs_raw, dict) or not docs_raw:
        raise ValueError("config.docs must be a non-empty object")

    docs: dict[str, DocsEntry] = {}
    for name, raw in docs_raw.items():
        if not isinstance(raw, dict):
            raise ValueError(f"docs.{name} must be an object")
        docs[name] = _parse_docs_entry(raw, name)

    default_docs, embedding_model, top_k, chunk_max_chars = _parse_default_block(
        data.get("default"), docs
    )
    return Config(
        docs=docs,
        default_docs=default_docs,
        embedding_model=embedding_model,
        top_k=top_k,
        chunk_max_chars=chunk_max_chars,
    )


def load_config(path: Path | None = None) -> Config:
    cfg_path = path or CONFIG_PATH
    if not cfg_path.is_file():
        raise FileNotFoundError(
            f"Config not found at {cfg_path}. "
            f"Copy config.example.json to {CONFIG_PATH} and set docs.<id>.source."
        )
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("config root must be a JSON object")
    return _config_from_data(data)


def config_from_dict(data: dict[str, Any]) -> Config:
    """Build Config from an in-memory dict (tests)."""
    return _config_from_data(data)
