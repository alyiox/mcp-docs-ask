"""Docs sync path / git URL detection."""

from __future__ import annotations

from pathlib import Path

import pytest

from mcp_docs_ask.config import DocsEntry
from mcp_docs_ask.sync import is_git_url, resolve_docs_root


def test_is_git_url() -> None:
    assert is_git_url("https://github.com/example/docs.git")
    assert is_git_url("git@github.com:example/docs.git")
    assert is_git_url("ssh://git@github.com/example/docs.git")
    assert not is_git_url("not-a-url")


def test_resolve_local_path(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    (root / "guides").mkdir()
    cfg = DocsEntry(source=str(root))
    checkout = resolve_docs_root("default", cfg, update=False)
    assert checkout.source == "path"
    assert checkout.root == root.resolve()


def test_resolve_missing_path(tmp_path: Path) -> None:
    missing = tmp_path / "nope"
    cfg = DocsEntry(source=str(missing))
    with pytest.raises(FileNotFoundError, match="does not exist"):
        resolve_docs_root("default", cfg, update=False)


def test_local_dir_not_treated_as_git(tmp_path: Path) -> None:
    root = tmp_path / "repo.git"
    root.mkdir()
    assert not is_git_url(str(root))


def test_checkout_ref_fetches_then_detaches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mcp_docs_ask import sync

    calls: list[list[str]] = []

    def fake_run(args: list[str], cwd: Path | None = None) -> str:
        calls.append(args)
        return "abc1234"

    monkeypatch.setattr(sync, "_run_git", fake_run)
    dest = tmp_path / "repo"
    dest.mkdir()
    sync._checkout_ref(dest, "feature/x")
    assert calls[0] == ["fetch", "--tags", "origin", "feature/x"]
    assert calls[1] == ["checkout", "--detach", "FETCH_HEAD"]


def test_checkout_ref_falls_back_to_branch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from mcp_docs_ask import sync

    calls: list[list[str]] = []

    def fake_run(args: list[str], cwd: Path | None = None) -> str:
        calls.append(args)
        if args[:2] == ["checkout", "--detach"]:
            raise RuntimeError("detach failed")
        return ""

    monkeypatch.setattr(sync, "_run_git", fake_run)
    dest = tmp_path / "repo"
    dest.mkdir()
    sync._checkout_ref(dest, "main")
    assert ["checkout", "-B", "main", "FETCH_HEAD"] in calls
