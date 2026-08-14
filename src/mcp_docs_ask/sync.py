"""Resolve docs repo root: local path or git URL (Homebrew-style cache)."""

from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .config import DocsEntry, repos_dir

logger = logging.getLogger(__name__)

_GIT_URL_RE = re.compile(
    r"^(?:https?://|git@|ssh://|git://).+|.*\.git$",
    re.IGNORECASE,
)


def is_git_url(value: str) -> bool:
    text = value.strip()
    if Path(text).expanduser().is_dir():
        return False
    return bool(_GIT_URL_RE.match(text))


@dataclass(frozen=True)
class DocsCheckout:
    root: Path
    source: str  # "path" | "git"
    docs_rev: str | None


def _run_git(args: list[str], cwd: Path | None = None) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "git is not on PATH. Install git, or set docs.<id>.source to a local path."
        ) from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or exc.stdout or "").strip()
        raise RuntimeError(
            f"git {' '.join(args)} failed"
            + (f": {stderr}" if stderr else "")
            + ". For private repos, run `gh auth login` or use SSH, "
            "or set source to a local clone path."
        ) from exc
    return completed.stdout.strip()


def short_rev(root: Path) -> str | None:
    try:
        return _run_git(["rev-parse", "--short", "HEAD"], cwd=root)
    except RuntimeError:
        return None


def _checkout_ref(dest: Path, ref: str) -> None:
    """Fetch ``ref`` explicitly, then check it out (detached for tags/SHAs).

    Works after ``--single-branch`` clones when the configured ref changes,
    because we always ``git fetch origin <ref>`` before checkout.
    """
    _run_git(["fetch", "--tags", "origin", ref], cwd=dest)
    # Prefer detached HEAD so tags and SHAs work the same as branches.
    try:
        _run_git(["checkout", "--detach", "FETCH_HEAD"], cwd=dest)
        return
    except RuntimeError:
        pass
    # Fallback: local branch name (create/reset to FETCH_HEAD)
    try:
        _run_git(["checkout", "-B", ref, "FETCH_HEAD"], cwd=dest)
    except RuntimeError:
        _run_git(["checkout", ref], cwd=dest)


def resolve_docs_root(
    docs_id: str,
    cfg: DocsEntry,
    *,
    update: bool = False,
) -> DocsCheckout:
    """Return the local docs repo root.

    When ``cfg.source`` is a git URL and ``update`` is True, fetch/update to ``cfg.ref``.
    ``ask_docs`` should call with ``update=False``; ``reindex`` with ``update=True``.
    """
    source = cfg.source.strip()
    path = Path(source).expanduser()

    if path.is_dir():
        return DocsCheckout(root=path.resolve(), source="path", docs_rev=short_rev(path))

    if not is_git_url(source):
        raise FileNotFoundError(
            f"docs source path does not exist: {path}. "
            "Clone the repo, or set source to a git URL "
            "(e.g. https://github.com/example/docs.git)."
        )

    dest = repos_dir() / docs_id
    dest.parent.mkdir(parents=True, exist_ok=True)

    if not (dest / ".git").exists():
        logger.info("Cloning %s into %s (ref=%s)", source, dest, cfg.ref)
        _run_git(["clone", "--branch", cfg.ref, "--single-branch", source, str(dest)])
    elif update:
        logger.info("Updating %s to ref %s", dest, cfg.ref)
        _checkout_ref(dest, cfg.ref)

    if not dest.is_dir():
        raise FileNotFoundError(f"git checkout missing at {dest}")

    return DocsCheckout(root=dest.resolve(), source="git", docs_rev=short_rev(dest))
