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
    "product": {
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
      },
      "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
    },
    "team-notes": {
      "source": "/path/to/docs",
      "desc": "Internal team notes (local path; ref unused)",
      "include": ["**/*.md"],
      "exclude": ["archive/**"]
    }
  },
  "default": {
    "docs": "product",
    "embedding_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "top_k": 8,
    "chunk_max_chars": 1500
  }
}
```

`product` is a git URL (`ref` applies). `team-notes` is a filesystem path (`ref` unused).
Optional `desc` on each docs collection and layer helps agents pick the right target.

`embedding_model`, `top_k`, and `chunk_max_chars` resolve as:
`docs.<id>.X` → `default.X` → built-in. Omit per-docs keys to inherit.

**Embedding model recommendation**

Any Hugging Face id loadable by `sentence-transformers` works. Pick by language mix:

| Docs / queries | Recommended `embedding_model` |
|---|---|
| **English-only** (built-in when omitted) | `sentence-transformers/all-MiniLM-L6-v2` |
| **Chinese-only** | `BAAI/bge-small-zh-v1.5` |
| **Multilingual** (~50 langs) | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |

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
| `docs.<id>.embedding_model` | Optional override (see recommendation above) |
| `docs.<id>.top_k` | Optional override for default retrieval count |
| `docs.<id>.chunk_max_chars` | Optional override for max body chars per heading chunk |
| `default.docs` | Default docs collection id |
| `default.embedding_model` | Default sentence-transformers model id |
| `default.top_k` | Default retrieval count |
| `default.chunk_max_chars` | Default max body chars per heading chunk |

**Layers** partition indexed files by path glob. First match wins. Names are
case-insensitive; `all` is reserved (cannot be configured as a layer name).

| `ask_docs` `layer` | Meaning |
|---|---|
| `all` (default) | Every indexed chunk (named layers and paths outside them) |
| `<named>` | Only chunks whose path matched that named layer’s `include` globs |

Paths that match no named-layer glob are still indexed and only appear under
`layer=all`. Omit `layers` (or set `"layers": {}`) for flat repos — use
`layer=all`.

Cache layout:

- Repos (git URL): `~/.cache/mcp-docs-ask/repos/<docs-id>/`
- Indexes: `~/.cache/mcp-docs-ask/indexes/<docs-id>/`

## Tools

| Tool | Description |
|---|---|
| `list_docs` | List configured docs collections and their layer filters |
| `ask_docs` | Retrieve grounded passages + citations (`layer`: `all` or a named layer) |
| `reindex` | Sync git source (if URL) and rebuild the vector index |

`list_docs` returns a `default` block with the same keys as the config `default`
block (`docs`, `embedding_model`, `top_k`, `chunk_max_chars`), plus a `docs` list
where each entry carries its resolved values and a `default` flag.
`layer_filters` is `all` plus named layer ids — see **Layers** above.

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
