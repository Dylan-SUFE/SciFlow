"""Embedding 模块。
来源：自研，使用 sentence-transformers。
"""
from __future__ import annotations

import hashlib
import logging

import numpy as np

logger = logging.getLogger(__name__)

_MODEL = None


def _get_model():
    global _MODEL
    if _MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            _MODEL = SentenceTransformer("BAAI/bge-small-en-v1.5")
            logger.info("已加载 bge-small-en-v1.5")
        except Exception as e:
            logger.warning(f"sentence-transformers 不可用({e})，使用哈希向量")
            _MODEL = "hashing"
    return _MODEL


def _hash_embed(text: str, dim: int = 384) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    for tok in text.lower().split():
        h = int.from_bytes(hashlib.md5(tok.encode()).digest()[:8], "little")
        vec[h % dim] += 1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def embed_chunks(chunks):
    """给所有 chunk 添加 embedding（先做分模态预处理）。"""
    from src.processing.preprocessors import preprocess_chunk

    model = _get_model()

    # 分模态预处理
    texts = [preprocess_chunk(c) for c in chunks]

    if model == "hashing":
        for c, t in zip(chunks, texts):
            c.embedding = _hash_embed(t)
        return chunks

    vecs = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
        batch_size=8,
    )
    for c, v in zip(chunks, vecs):
        c.embedding = v.astype(np.float32)
    return chunks