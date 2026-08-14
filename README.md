# mcp-docs-ask

[![CI](https://github.com/alyiox/mcp-docs-ask/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/alyiox/mcp-docs-ask/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/mcp-docs-ask.svg)](https://pypi.org/project/mcp-docs-ask/)
[![Python 3.13+](https://img.shields.io/badge/python-3.13%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

<!-- mcp-name: io.github.alyiox/mcp-docs-ask -->

Local RAG MCP for documentation. Point `source` at any markdown repository
(local path or git URL).

The server does **retrieval only** (no answer LLM). `ask_docs` returns grounded
passages and citations; the MCP host (Cursor / Claude) synthesizes the answer.

## Features

- `ask_docs` retrieval with configurable path-based layer filters
- `list_docs` discovery for configured docs collections and layer filters
- `reindex` rebuilds the local vector index; for git URL sources it also fetches updates
- Multilingual embeddings (default model works across languages)
- Light ASCII-token boost for coding lookups (`OrderList`, `UserController`, …)
- Docs source: local path **or** git URL (Homebrew-style cache under `~/.cache/mcp-docs-ask/`)

## Requirements

- Python 3.13+
- [`uv`](https://docs.astral.sh/uv/)
- `git` on PATH (only if `source` is a git URL)
- Git credentials on the machine when `source` is a **private** git URL
  (`gh auth login`, HTTPS credential helper, or SSH). No tokens in config.
- First run downloads the embedding model weights once (sentence-transformers)

## Quick start

```bash
git clone git@github.com:alyiox/mcp-docs-ask.git
cd mcp-docs-ask
uv sync
mkdir -p ~/.config/mcp-docs-ask
cp config.example.json ~/.config/mcp-docs-ask/config.json
# Prefer a local checkout while developing:
#   set docs.<id>.source to your docs repo path
npx -y @modelcontextprotocol/inspector uv run mcp-docs-ask
```

## Configuration

Config path: `~/.config/mcp-docs-ask/config.json`

> **Windows:** `%USERPROFILE%\.config\mcp-docs-ask\config.json`

```json
{
  "docs": {
    "default": {
      "source": "https://github.com/example/docs.git",
      "desc": "Product guides and API reference",
      "ref": "main",
      "include": ["**/*.md"],
      "exclude": ["archive/**"],
      "layers": {
        "guides": {
          "desc": "How-to and onboarding guides",
          "include": ["docs/guides/**"]
        },
        "api": {
          "desc": "HTTP API reference",
          "include": ["docs/api/**"]
        }
      }
    },
    "local": {
      "source": "/path/to/docs",
      "desc": "Local markdown tree (no git; ref unused)",
      "include": ["**/*.md"],
      "exclude": ["archive/**"]
    }
  },
  "default_docs": "default",
  "embedding_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
  "top_k": 8,
  "chunk_max_chars": 1500
}
```

`default` is a git URL (`ref` applies). `local` is a filesystem path (`ref` is unused if present). Optional `desc` on each docs collection and layer helps agents pick the right target.

**Embedding model recommendation**

| Docs / queries | Recommended `embedding_model` |
|---|---|
| **English-only** (default when omitted) | `sentence-transformers/all-MiniLM-L6-v2` |
| **Multilingual** | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (see `config.example.json`) |

Changing `embedding_model` requires a `reindex` (the on-disk index stores the model name).

| Field | Description |
|---|---|
| `docs.<id>.source` | Docs **repo root**: local path or git URL |
| `docs.<id>.desc` | Short description for discovery (`list_docs`) |
| `docs.<id>.ref` | Branch / tag / SHA for git URL sources only (default `main`; ignored for local paths) |
| `docs.<id>.include` | Globs relative to repo root (default `**/*.md`) |
| `docs.<id>.exclude` | Globs to skip |
| `docs.<id>.layers.<name>.include` | Path globs for that layer (first match wins) |
| `docs.<id>.layers.<name>.desc` | Short layer description for discovery |
| `embedding_model` | sentence-transformers model id (see recommendation above) |
| `top_k` | Default retrieval count |
| `chunk_max_chars` | Max body chars per heading chunk |

Omit `layers` (or set `"layers": {}`) for flat repos — everything is `other`
and `ask_docs` uses `layer=all`. Configure any names you need for multi-tree
docs. First match wins. Layer names are case-insensitive; `all` / `other` are reserved.

Cache layout:

- Repos (git URL): `~/.cache/mcp-docs-ask/repos/<docs-id>/`
- Indexes: `~/.cache/mcp-docs-ask/indexes/<docs-id>/`

## Tools

| Tool | Description |
|---|---|
| `list_docs` | List configured docs collections and their layer filters |
| `ask_docs` | Retrieve grounded passages + citations for a question |
| `reindex` | Sync git source (if URL) and rebuild the vector index |

## MCP host examples

### Cursor

Add to `.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "docs-ask": {
      "command": "uvx",
      "args": ["mcp-docs-ask"]
    }
  }
}
```

### Claude Code

Add to your Claude Code MCP config:

```json
{
  "mcpServers": {
    "docs-ask": {
      "command": "uvx",
      "args": ["mcp-docs-ask"]
    }
  }
}
```

### Codex

```toml
[mcp_servers.docs-ask]
command = "uvx"
args = ["mcp-docs-ask"]
```

### OpenCode

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "docs-ask": {
      "type": "local",
      "enabled": true,
      "command": ["uvx", "mcp-docs-ask"]
    }
  }
}
```

### GitHub Copilot

```json
{
  "inputs": [],
  "servers": {
    "docs-ask": {
      "type": "stdio",
      "command": "uvx",
      "args": ["mcp-docs-ask"]
    }
  }
}
```

## Development

```bash
uv sync
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run pyright
uv run pytest
```

## Notes

- **Local path:** `ask_docs` rebuilds the index automatically when file mtimes/sizes
  change (fingerprint check). You do not need `reindex` after editing local docs.
- **Git URL:** `ask_docs` never fetches. Call `reindex` to `git fetch` the configured
  `ref` and rebuild.
- Changing `embedding_model` invalidates the on-disk index (rebuild on next use /
  `reindex`).
