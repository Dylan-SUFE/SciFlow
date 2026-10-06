"""Milvus 向量存储。
来源：参考 Milvus 官方示例。
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
COLLECTION = os.getenv("MILVUS_COLLECTION", "sciflow")
DIM = int(os.getenv("EMBEDDING_DIM", "384"))


class MilvusStore:
    def __init__(self):
        self._client = None
        self._memory: dict[str, dict] = {}  # 内存回退

    def _connect(self):
        if self._client is not None:
            return self._client
        try:
            from pymilvus import MilvusClient
            self._client = MilvusClient(uri=f"http://{MILVUS_HOST}:{MILVUS_PORT}")
            logger.info(f"已连接 Milvus: {MILVUS_HOST}:{MILVUS_PORT}")
            self._ensure_collection()
        except Exception as e:
            logger.warning(f"Milvus 不可用({e})，使用内存存储")
            self._client = "memory"
        return self._client

    def _ensure_collection(self):
        if self._client == "memory":
            return
        if self._client.has_collection(COLLECTION):
            return
        self._client.create_collection(
            collection_name=COLLECTION,
            dimension=DIM,
            metric_type="COSINE",
            auto_id=False,
        )
        logger.info(f"已创建 collection: {COLLECTION}")

    def reset(self):
        client = self._connect()
        if client == "memory":
            self._memory.clear()
            return
        if client.has_collection(COLLECTION):
            client.drop_collection(COLLECTION)
        self._ensure_collection()

    def upsert(self, chunks):
        client = self._connect()
        if client == "memory":
            for c in chunks:
                self._memory[c.chunk_id] = {
                    "id": c.chunk_id,
                    "vector": c.embedding.tolist() if c.embedding is not None else [0.0] * DIM,
                    "content": c.content,
                    "modality": c.modality,
                    "paper_id": c.paper_id,
                }
            return

        data = [
            {
                "id": c.chunk_id,
                "vector": c.embedding.tolist() if c.embedding is not None else [0.0] * DIM,
                "content": c.content,
                "modality": c.modality,
                "paper_id": c.paper_id,
            }
            for c in chunks
        ]
        client.upsert(collection_name=COLLECTION, data=data)

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        from src.processing.embedder import _get_model, _hash_embed

        model = _get_model()
        if model == "hashing":
            qvec = _hash_embed(query)
        else:
            qvec = model.encode([query], normalize_embeddings=True)[0].astype("float32")

        client = self._connect()
        if client == "memory":
            hits = []
            q = qvec.tolist()
            for item in self._memory.values():
                score = _cosine(q, item["vector"])
                hits.append({"id": item["id"], "score": score, "content": item["content"]})
            hits.sort(key=lambda x: x["score"], reverse=True)
            return hits[:top_k]

        results = client.search(
            collection_name=COLLECTION,
            data=[qvec.tolist()],
            limit=top_k,
            output_fields=["content", "modality", "paper_id"],
        )
        return [
            {"id": hit["id"], "score": float(hit["distance"]), "content": hit["entity"].get("content", "")}
            for hit in results[0]
        ]


def _cosine(a, b) -> float:
    import numpy as np
    a, b = np.array(a), np.array(b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(a @ b / (na * nb))