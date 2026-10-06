"""BM25 稀疏检索。
来源：参考 rank_bm25 官方示例。
"""
from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)

BM25_PATH = Path("./data/bm25.pkl")


class BM25Store:
    def __init__(self):
        self._chunks = []  # [(chunk_id, content)]
        self._bm25 = None

    def build(self, chunks):
        self._chunks = [(c.chunk_id, c.content) for c in chunks]
        try:
            from rank_bm25 import BM25Okapi
            tokenized = [self._tokenize(text) for _, text in self._chunks]
            self._bm25 = BM25Okapi(tokenized)
        except ImportError:
            logger.warning("rank_bm25 未安装，BM25 检索禁用")
            self._bm25 = None
        # 持久化
        BM25_PATH.parent.mkdir(parents=True, exist_ok=True)
        with BM25_PATH.open("wb") as f:
            pickle.dump(self._chunks, f)

    def load(self):
        if BM25_PATH.exists():
            with BM25_PATH.open("rb") as f:
                self._chunks = pickle.load(f)
            try:
                from rank_bm25 import BM25Okapi
                tokenized = [self._tokenize(text) for _, text in self._chunks]
                self._bm25 = BM25Okapi(tokenized)
            except ImportError:
                self._bm25 = None
        return self

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        if self._bm25 is None or not self._chunks:
            return []
        tokens = self._tokenize(query)
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            {"id": self._chunks[i][0], "score": float(s), "content": self._chunks[i][1]}
            for i, s in ranked
            if s > 0
        ]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [t for t in text.lower().split() if t.strip()]