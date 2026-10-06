"""数据质量统计报告。

统计处理后的数据质量指标：
- 每篇论文的 chunk 数、图表数、表格数、公式数
- chunk 长度分布
- 模态分布
- 异常检查（乱码、碎片、过短）

用法：
    python3 -m scripts.quality_report
    python3 -m scripts.quality_report --json report.json
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path


def load_chunks(bm25_path: str = "data/bm25.pkl"):
    if not Path(bm25_path).exists():
        raise FileNotFoundError(f"找不到 {bm25_path}，请先运行 main.py")
    with open(bm25_path, "rb") as f:
        return pickle.load(f)


def infer_modality(chunk_id: str) -> str:
    suffix = chunk_id.rsplit(":", 1)[-1]
    if suffix.startswith("fig"):
        return "figure"
    if suffix.startswith("tbl"):
        return "table"
    if suffix.startswith("r") and len(suffix) > 1 and suffix[1:].isdigit():
        return "table"
    if suffix.startswith("f") and len(suffix) > 1 and suffix[1:].isdigit():
        return "formula"
    return "text"


def check_garbage(text: str) -> bool:
    """检测乱码：非 ASCII 字符占比 > 30%。"""
    if not text:
        return True
    non_ascii = sum(1 for ch in text if ord(ch) > 127)
    return non_ascii / len(text) > 0.3


def check_fragment(text: str, min_len: int = 50) -> bool:
    """检测碎片：长度 < min_len。"""
    return len(text) < min_len


def generate_report(chunks: list[tuple[str, str]]) -> dict:
    """生成数据质量报告。"""
    print(f"总 chunk 数: {len(chunks)}")
    print()

    # 1. 按模态分组
    by_modality: dict[str, list] = defaultdict(list)
    for cid, content in chunks:
        m = infer_modality(cid)
        by_modality[m].append((cid, content))

    # 2. 按论文分组
    by_paper: dict[str, list] = defaultdict(list)
    for cid, content in chunks:
        paper_id = cid.split(":")[0]
        by_paper[paper_id].append((cid, content))

    print(f"=== 模态分布 ===")
    for m in ["text", "figure", "table", "formula"]:
        items = by_modality[m]
        print(f"  {m}: {len(items)} 个 ({len(items)/len(chunks):.1%})")
    print()

    print(f"=== 论文分布 ===")
    print(f"  论文数: {len(by_paper)}")
    chunks_per_paper = [len(v) for v in by_paper.values()]
    print(f"  平均每篇 chunk 数: {sum(chunks_per_paper)/len(chunks_per_paper):.1f}")
    print(f"  最大: {max(chunks_per_paper)}，最小: {min(chunks_per_paper)}")
    print()

    # 3. chunk 长度分布（仅文本）
    text_lengths = [len(c) for _, c in by_modality["text"]]
    if text_lengths:
        print(f"=== 文本 chunk 长度分布 ===")
        print(f"  最小: {min(text_lengths)}")
        print(f"  最大: {max(text_lengths)}")
        print(f"  平均: {sum(text_lengths)//len(text_lengths)}")
        print()

        buckets = Counter()
        for l in text_lengths:
            if l < 100:
                buckets["< 100（碎片）"] += 1
            elif l < 500:
                buckets["100-500（短）"] += 1
            elif l < 1200:
                buckets["500-1200（正常）"] += 1
            else:
                buckets["> 1200（长）"] += 1
        for k in ["< 100（碎片）", "100-500（短）", "500-1200（正常）", "> 1200（长）"]:
            print(f"  {k}: {buckets[k]} ({buckets[k]/len(text_lengths):.1%})")
        print()

    # 4. 异常检查
    print(f"=== 异常检查 ===")
    garbage_count = sum(1 for _, c in chunks if check_garbage(c))
    fragment_count = sum(1 for _, c in chunks if check_fragment(c))
    print(f"  乱码 chunk: {garbage_count} ({garbage_count/len(chunks):.1%})")
    print(f"  碎片 chunk (< 50 字符): {fragment_count} ({fragment_count/len(chunks):.1%})")
    print()

    return {
        "total": len(chunks),
        "paper_count": len(by_paper),
        "modality_distribution": {m: len(by_modality[m]) for m in ["text", "figure", "table", "formula"]},
        "chunks_per_paper": {
            "avg": sum(chunks_per_paper) / len(chunks_per_paper),
            "max": max(chunks_per_paper),
            "min": min(chunks_per_paper),
        },
        "text_length": {
            "min": min(text_lengths) if text_lengths else 0,
            "max": max(text_lengths) if text_lengths else 0,
            "avg": sum(text_lengths) // len(text_lengths) if text_lengths else 0,
        },
        "anomalies": {
            "garbage": garbage_count,
            "fragment": fragment_count,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default=None, help="导出 JSON 报告")
    parser.add_argument("--bm25", default="data/bm25.pkl")
    args = parser.parse_args()

    chunks = load_chunks(args.bm25)
    report = generate_report(chunks)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"已导出 JSON 报告到 {args.json}")


if __name__ == "__main__":
    main()