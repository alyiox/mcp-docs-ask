"""Server surface tests: drive the tools through an in-process MCP client."""

from __future__ import annotations

import json

import pytest
from mcp import Client

from mcp_docs_ask import server
from mcp_docs_ask.config import config_from_dict
from mcp_docs_ask.embedder import HashEmbedder

_CONFIG = {
    "docs": {
        "product": {
            "source": "/private/local/docs",
            "desc": "Product docs",
            "layers": {"guides": {"include": ["docs/guides/**"]}},
        }
    },
    "default_docs": "product",
}


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch):
    """An in-process client whose lifespan uses a stub config and embedder."""
    monkeypatch.setattr(server, "load_config", lambda: config_from_dict(_CONFIG))
    monkeypatch.setattr(server, "SentenceTransformerEmbedder", lambda _model: HashEmbedder())
    return Client(server.mcp)


async def test_tools_expose_expected_schemas(client) -> None:
    async with client as session:
        tools = {t.name: t for t in (await session.list_tools()).tools}

    assert set(tools) == {"list_docs", "ask_docs", "reindex"}
    # The injected Context parameter must stay out of the advertised schema.
    assert tools["list_docs"].input_schema.get("properties", {}) == {}
    assert set(tools["ask_docs"].input_schema["properties"]) == {
        "question",
        "docs",
        "top_k",
        "layer",
    }
    assert set(tools["reindex"].input_schema["properties"]) == {"docs"}


async def test_tools_are_annotated(client) -> None:
    async with client as session:
        tools = {t.name: t for t in (await session.list_tools()).tools}

    for name in ("list_docs", "ask_docs"):
        annotations = tools[name].annotations
        assert annotations is not None, name
        assert annotations.read_only_hint is True, name
        assert annotations.open_world_hint is False, name

    reindex = tools["reindex"].annotations
    assert reindex is not None
    assert reindex.read_only_hint is False
    assert reindex.destructive_hint is False
    assert reindex.idempotent_hint is True
    assert reindex.open_world_hint is True


async def test_list_docs_reads_the_lifespan_context(client) -> None:
    async with client as session:
        result = await session.call_tool("list_docs", {})

    assert not result.is_error
    payload = json.loads(result.content[0].text)
    assert payload["default_docs"] == "product"
    assert [p["id"] for p in payload["docs"]] == ["product"]
    assert payload["docs"][0]["desc"] == "Product docs"
