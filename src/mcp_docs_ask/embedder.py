"""Embedding backends (sentence-transformers + injectable fake for tests)."""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[A-Za-z0-9_./-]+|[\u4e00-\u9fff]+")


class Embedder(Protocol):
    model_name: str

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return float32 matrix shape (n, dim), L2-normalized rows."""
        ...


class HashEmbedder:
    """Deterministic bag-of-tokens embedder for unit tests (no model download)."""

    def __init__(self, model_name: str = "hash-embedder/v1", dim: int = 64) -> None:
        self.model_name = model_name
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            tokens: list[str] = []
            for token in _WORD_RE.findall(text.lower()):
                tokens.append(token)
                # Also add char unigrams for scripts matched as long tokens
                if re.fullmatch(r"[\u4e00-\u9fff]+", token):
                    tokens.extend(list(token))
            for token in tokens:
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                idx = int.from_bytes(digest[:4], "little") % self.dim
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                out[i, idx] += sign
            norm = float(np.linalg.norm(out[i]))
            if norm > 0:
                out[i] /= norm
        return out


class SentenceTransformerEmbedder:
    """Lazy-loaded sentence-transformers wrapper."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None

    def _ensure(self) -> None:
        if self._model is not None:
            return
        logger.info("Loading embedding model %s", self.model_name)
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self.model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        self._ensure()
        assert self._model is not None
        vectors = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)
