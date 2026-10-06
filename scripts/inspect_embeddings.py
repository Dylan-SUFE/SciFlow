"""查看向量化结果。

用法：
    python3 -m scripts.inspect_embeddings
    python3 -m scripts.inspect_embeddings --dump vectors.npy
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import pickle

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", default=None, help="导出向量到 .npy 文件")
    args = parser.parse_args()

    # 1. 从 BM25 pkl 读 chunk 文本
    chunks = pickle.load(open("data/bm25.pkl", "rb"))
    texts = [c[1] for c in chunks[:20]]   # 前 20 条做示例
    ids = [c[0] for c in chunks[:20]]

    print(f"BM25 共 {len(chunks)} 条 chunk，取前 20 条做示例\n")

    # 2. 加载 embedding 模型
    from src.processing.embedder import _get_model
    model = _get_model()

    if model == "hashing":
        print("⚠️  当前用的是哈希向量（无语义）")
        from src.processing.embedder import _hash_embed
        vecs = np.stack([_hash_embed(t) for t in texts])
    else:
        print("✅ 使用的是 bge-small-en-v1.5（语义向量）")
        vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    print(f"\n向量形状: {vecs.shape}  (数量, 维度)")
    print(f"数据类型: {vecs.dtype}")
    print(f"每行范数（应该都接近 1.0）: {np.linalg.norm(vecs, axis=1)[:5]}")
    print()

    # 3. 展示前 3 条向量
    for i in range(min(3, len(texts))):
        print(f"[{ids[i]}]")
        print(f"  文本: {texts[i][:80]}...")
        print(f"  向量前 8 维: {vecs[i][:8]}")
        print()

    # 4. 相似度测试
    print("=" * 70)
    print("相似度测试：验证语义模型是否工作")
    print("=" * 70)
    print()

    test_pairs = [
        ("dose-response relationship", "dosage effect"),           # 应该高
        ("dose-response relationship", "quantum entanglement"),    # 应该低
        ("machine learning model", "neural network"),              # 应该高
        ("machine learning model", "chocolate cake recipe"),       # 应该低
    ]

    for a, b in test_pairs:
        if model == "hashing":
            from src.processing.embedder import _hash_embed
            va, vb = _hash_embed(a), _hash_embed(b)
        else:
            va, vb = model.encode([a, b], normalize_embeddings=True)

        sim = float(np.dot(va, vb))
        print(f"  '{a}'")
        print(f"    vs '{b}'  →  相似度 {sim:.4f}")
        print()

    # 5. 导出
    if args.dump:
        np.save(args.dump, vecs)
        print(f"已导出到 {args.dump}")


if __name__ == "__main__":
    main()