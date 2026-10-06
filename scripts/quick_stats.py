"""快速统计：只做切块，不向量化。

用途：调试切块逻辑时，跳过阶段1（解析）和阶段3（向量化），
      直接看统计结果。

耗时：约 30 秒（相比完整流程 25 分钟）

用法：
    python3 -m scripts.quick_stats
"""
from __future__ import annotations

import pickle
from collections import Counter, defaultdict
from pathlib import Path


def load_cached_documents():
    """从阶段1缓存读 documents。"""
    cache = Path("data/ingest_cache.pkl")
    if not cache.exists():
        raise FileNotFoundError(
            "找不到 data/ingest_cache.pkl。\n"
            "请先跑一次 main.py 生成缓存：\n"
            "  ENABLE_TABLES=false EMBEDDING_DEVICE=cpu python3 main.py --run-all"
        )
    with cache.open("rb") as f:
        data = pickle.load(f)
    return data["documents"]


def infer_modality(chunk_id: str) -> str:
    """从 chunk_id 判断模态。"""
    suffix = chunk_id.rsplit(":", 1)[-1]
    if suffix.startswith("fig"):
        return "figure"
    if suffix.startswith("tbl"):
        return "table"
    if suffix.startswith("f") and len(suffix) > 1 and suffix[1:].isdigit():
        return "formula"
    return "text"


def main():
    print("=== 从缓存读 documents ===")
    documents = load_cached_documents()
    print(f"读入 {len(documents)} 篇文档")
    print()

    print("=== 切块（不向量化）===")
    from src.processing.chunker import chunk_document

    all_chunks = []
    for doc in documents:
        all_chunks.extend(chunk_document(doc))

    print(f"切出 {len(all_chunks)} 个 chunk")
    print()

    # ===== 统计 =====
    # 1. 模态分布
    by_modality = Counter()
    for c in all_chunks:
        by_modality[c.modality] += 1

    print("=== 模态分布 ===")
    total = len(all_chunks)
    for m in ["text", "figure", "table", "formula"]:
        count = by_modality[m]
        print(f"  {m}: {count} 个 ({count/total:.1%})")
    print()

    # 2. 论文分布
    paper_ids = set()
    chunks_per_paper = defaultdict(int)
    for c in all_chunks:
        pid = c.paper_id
        paper_ids.add(pid)
        chunks_per_paper[pid] += 1

    print("=== 论文分布 ===")
    print(f"  唯一 paper_id 数: {len(paper_ids)}")
    counts = list(chunks_per_paper.values())
    if counts:
        print(f"  平均每篇 chunk 数: {sum(counts)/len(counts):.1f}")
        print(f"  最大: {max(counts)}，最小: {min(counts)}")
    print()

    # 3. 文本长度分布
    text_chunks = [c for c in all_chunks if c.modality == "text"]
    if text_chunks:
        lengths = [len(c.content) for c in text_chunks]
        print("=== 文本 chunk 长度分布 ===")
        print(f"  最小: {min(lengths)}")
        print(f"  最大: {max(lengths)}")
        print(f"  平均: {sum(lengths)//len(lengths)}")
        print()

        buckets = Counter()
        for l in lengths:
            if l < 100:
                buckets["< 100（碎片）"] += 1
            elif l < 500:
                buckets["100-500（短）"] += 1
            elif l < 1200:
                buckets["500-1200（正常）"] += 1
            else:
                buckets["> 1200（长）"] += 1

        for k in ["< 100（碎片）", "100-500（短）", "500-1200（正常）", "> 1200（长）"]:
            print(f"  {k}: {buckets[k]} ({buckets[k]/len(lengths):.1%})")
        print()

    # 4. paper_id 采样（验证修复）
    print("=== paper_id 采样（前 10）===")
    for pid in sorted(paper_ids)[:10]:
        print(f"  {pid}")
    print()

    # 5. 检查是否还有裸 ID
    bad_ids = [pid for pid in paper_ids if not pid.startswith("2")]
    if bad_ids:
        print(f"⚠️  异常 paper_id: {len(bad_ids)} 个")
        for pid in bad_ids[:5]:
            print(f"  {pid}")
    else:
        print("✅ 所有 paper_id 格式正确")


if __name__ == "__main__":
    main()