"""MCPServer entrypoint for Docs Ask (local docs RAG)."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Annotated, Any

from mcp.server.mcpserver import Context, MCPServer
from mcp.types import ToolAnnotations
from pydantic import Field

from .config import Config, load_config
from .embedder import Embedder, SentenceTransformerEmbedder
from .tools.ask_docs import ask_docs_impl
from .tools.list_docs import list_docs_impl
from .tools.reindex import reindex_impl

logger = logging.getLogger(__name__)


@dataclass
class ServerContext:
    config: Config
    embedder: Embedder


@asynccontextmanager
async def _lifespan(app: MCPServer) -> AsyncIterator[ServerContext]:
    config = load_config()
    embedder: Embedder = SentenceTransformerEmbedder(config.embedding_model)
    ctx = ServerContext(config=config, embedder=embedder)
    logger.info(
        "docs-ask ready: docs=%s default=%s model=%s",
        sorted(config.docs),
        config.default_docs,
        config.embedding_model,
    )
    yield ctx


mcp: MCPServer = MCPServer("docs-ask", lifespan=_lifespan)


def _ctx(ctx: Context[ServerContext, Any]) -> ServerContext:
    return ctx.request_context.lifespan_context


def _json(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


@mcp.tool(
    name="list_docs",
    description=(
        "[DocsAsk] List configured documentation collections and available layer filters. "
        "Src: config"
    ),
    annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False),
)
def list_docs(ctx: Context[ServerContext, Any]) -> str:
    return _json(list_docs_impl(_ctx(ctx).config))


@mcp.tool(
    name="ask_docs",
    description=(
        "[DocsAsk] Retrieve grounded documentation passages for a question. "
        "Returns answer_context and citations; host synthesizes the answer. Src: index"
    ),
    annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False),
)
def ask_docs(
    ctx: Context[ServerContext, Any],
    question: Annotated[
        str,
        Field(description="[DocsAsk] Natural-language question. Src: user"),
    ],
    docs: Annotated[
        str | None,
        Field(
            description=("[DocsAsk] Docs collection id (default from config). Src: config"),
        ),
    ] = None,
    top_k: Annotated[
        int | None,
        Field(
            description="[DocsAsk] Max passages to return (default from config). Src: user",
            ge=1,
            le=50,
        ),
    ] = None,
    layer: Annotated[
        str,
        Field(
            description=(
                "[DocsAsk] Layer filter: all, other, or a configured layer name. Src: config"
            ),
        ),
    ] = "all",
) -> str:
    server_ctx = _ctx(ctx)
    try:
        result = ask_docs_impl(
            server_ctx.config,
            server_ctx.embedder,
            question=question,
            docs=docs,
            top_k=top_k,
            layer=layer,
        )
    except Exception as exc:
        return _json({"error": type(exc).__name__, "message": str(exc)})
    return _json(result)


@mcp.tool(
    name="reindex",
    description=(
        "[DocsAsk] Rebuild the local docs vector index. When source is a git URL, "
        "fetch/update to ref first (requires git credentials for private repos). "
        "Src: docs"
    ),
    annotations=ToolAnnotations(
        read_only_hint=False,
        destructive_hint=False,
        idempotent_hint=True,
        open_world_hint=True,
    ),
)
def reindex(
    ctx: Context[ServerContext, Any],
    docs: Annotated[
        str | None,
        Field(description="[DocsAsk] Docs collection id to reindex. Src: config"),
    ] = None,
) -> str:
    server_ctx = _ctx(ctx)
    try:
        result = reindex_impl(server_ctx.config, server_ctx.embedder, docs=docs)
    except Exception as exc:
        return _json({"error": type(exc).__name__, "message": str(exc)})
    return _json(result)


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("MCP_DOCS_ASK_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
