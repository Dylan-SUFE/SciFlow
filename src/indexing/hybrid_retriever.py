"""稠密 + 稀疏混合检索。
来源：复用 central-dogma 的融合逻辑。
"""
from __future__ import annotations


class HybridRetriever:
    def __init__(self, milvus_store, bm25_store, alpha: float = 0.7):
        self.milvus = milvus_store
        self.bm25 = bm25_store
        self.alpha = alpha

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        dense_hits = self.milvus.search(query, top_k=top_k)
        sparse_hits = self.bm25.search(query, top_k=top_k)
        return self._fuse(dense_hits, sparse_hits, top_k)

    def _fuse(self, dense_hits, sparse_hits, top_k):
        """加权 RRF 融合。"""
        scores: dict[str, float] = {}
        contents: dict[str, str] = {}

        for rank, hit in enumerate(dense_hits):
            cid = hit["id"]
            scores[cid] = scores.get(cid, 0.0) + self.alpha * (1.0 / (rank + 1))
            contents[cid] = hit.get("content", "")

        for rank, hit in enumerate(sparse_hits):
            cid = hit["id"]
            scores[cid] = scores.get(cid, 0.0) + (1 - self.alpha) * (1.0 / (rank + 1))
            contents.setdefault(cid, hit.get("content", ""))

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [{"id": cid, "score": s, "content": contents.get(cid, "")} for cid, s in ranked]