"""查看 PDF 提取效果。

用法：
    python3 -m scripts.inspect_extraction
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pickle
from collections import Counter


def main():
    chunks = pickle.load(open("data/bm25.pkl", "rb"))
    print(f"共 {len(chunks)} 条 chunk\n")

    # 1. 按论文分组统计
    by_paper: dict[str, list] = {}
    for cid, content in chunks:
        paper_id = cid.split(":")[0]
        by_paper.setdefault(paper_id, []).append((cid, content))

    print("=" * 70)
    print("一、每篇论文的 chunk 数")
    print("=" * 70)
    for pid in sorted(by_paper.keys()):
        print(f"  {pid}: {len(by_paper[pid])} 条")
    print()

    # 2. chunk 长度分布
    lengths = [len(c) for _, c in chunks]
    print("=" * 70)
    print("二、chunk 长度分布")
    print("=" * 70)
    print(f"  最小: {min(lengths)} 字符")
    print(f"  最大: {max(lengths)} 字符")
    print(f"  平均: {sum(lengths) // len(lengths)} 字符")

    # 分桶
    buckets = Counter()
    for l in lengths:
        if l < 100:
            buckets["< 100（碎片）"] += 1
        elif l < 500:
            buckets["100-500（短）"] += 1
        elif l < 1200:
            buckets["500-1200（正常）"] += 1
        elif l < 2000:
            buckets["1200-2000（长）"] += 1
        else:
            buckets["> 2000（超长）"] += 1

    print()
    for k in ["< 100（碎片）", "100-500（短）", "500-1200（正常）", "1200-2000（长）", "> 2000（超长）"]:
        print(f"  {k}: {buckets[k]} 条")
    print()

    # 3. 采样前 5 条看内容质量
    print("=" * 70)
    print("三、采样 5 条 chunk，人工核对提取质量")
    print("=" * 70)
    for cid, content in chunks[:5]:
        print(f"\n[{cid}]  ({len(content)} 字符)")
        print("  " + content[:300].replace("\n", "\n  "))
        print()
        input("  按 Enter 继续...")

    # 4. 检查异常特征
    print("=" * 70)
    print("四、异常检查")
    print("=" * 70)

    # 乱码检查（非 ASCII 字符比例）
    weird = sum(1 for _, c in chunks if sum(1 for ch in c if ord(ch) > 127) / max(len(c), 1) > 0.3)
    print(f"  含大量非 ASCII 字符的 chunk: {weird} 条（可能是乱码）")

    # 过短检查
    too_short = sum(1 for _, c in chunks if len(c) < 50)
    print(f"  长度 < 50 的 chunk: {too_short} 条（可能是碎片）")

    # 公式碎片检查（含大量数学符号但语义弱）
    formula_like = sum(
        1 for _, c in chunks
        if any(sym in c for sym in ["^{", "_{", "\\frac", "\\sum"])
        and len(c) < 300
    )
    print(f"  疑似公式碎片: {formula_like} 条")


if __name__ == "__main__":
    main()