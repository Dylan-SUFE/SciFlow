"""从真实语料库构建评估集。

工作流程：
1. 随机采样 N 个 chunk（作为候选 gold）
2. 对每个 chunk，反向生成一个问题（用 chunk 首句或让用户手写）
3. 输出到 JSONL，用户人工确认/修改 gold_source_ids

用法：
    python3 scripts/build_eval_set.py --sample 10
    python3 scripts/build_eval_set.py --sample 10 --out evaluation/datasets/real_eval.jsonl
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
from pathlib import Path

# BM25 里存着所有 chunk 的 (chunk_id, content)
BM25_PATH = Path("data/bm25.pkl")
OUTPUT_DIR = Path("evaluation/datasets")


def load_all_chunks():
    """从 BM25 的 pickle 文件里读出所有 chunk。"""
    if not BM25_PATH.exists():
        raise FileNotFoundError(
            f"找不到 {BM25_PATH}。请先运行 python3 main.py --run-all 建立索引。"
        )
    with BM25_PATH.open("rb") as f:
        chunks = pickle.load(f)
    print(f"从 {BM25_PATH} 读取到 {len(chunks)} 个 chunk")
    return chunks


def sample_chunks(chunks, n: int):
    """随机采样 n 个 chunk，避免全是同一篇论文。"""
    # 按 paper_id 分组
    by_paper: dict[str, list] = {}
    for cid, content in chunks:
        paper_id = cid.split(":")[0]
        by_paper.setdefault(paper_id, []).append((cid, content))

    # 每篇论文最多采样 2 个，保证多样性
    sampled = []
    for paper_id, items in by_paper.items():
        k = min(2, len(items))
        sampled.extend(random.sample(items, k))

    # 如果还不够 n 个，从剩余里补
    if len(sampled) < n:
        remaining = [c for c in chunks if c not in sampled]
        random.shuffle(remaining)
        sampled.extend(remaining[: n - len(sampled)])

    random.shuffle(sampled)
    return sampled[:n]


def make_question_from_chunk(content: str) -> str:
    """从 chunk 内容反向生成一个粗略的问题。用户需要人工修改。"""
    # 简单策略：取第一句话，问"这段内容讲了什么"
    first_sentence = content.split(".")[0].strip()
    if len(first_sentence) > 80:
        first_sentence = first_sentence[:80] + "..."
    return f"According to the paper, what does this passage describe: {first_sentence}?"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=10, help="采样多少个 chunk")
    parser.add_argument("--out", default="evaluation/datasets/real_eval.jsonl")
    args = parser.parse_args()

    chunks = load_all_chunks()
    sampled = sample_chunks(chunks, args.sample)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)

    with out_path.open("w", encoding="utf-8") as f:
        for cid, content in sampled:
            case = {
                "question": make_question_from_chunk(content),
                "gold_source_ids": [cid],
                "answerable": True,
                "notes": f"手动核对。原文片段：{content[:100]}...",
            }
            f.write(json.dumps(case, ensure_ascii=False) + "\n")

    print(f"\n已生成 {out_path}")
    print(f"共 {len(sampled)} 条用例")
    print("\n下一步：")
    print(f"1. 打开 {out_path}，逐条检查 question 和 gold_source_ids 是否匹配")
    print("2. 修改 question 让它更像真实用户提问")
    print("3. 修正 gold_source_ids（如果你觉得检索到的更好，可以换）")
    print("4. 保存后，跑评估：python3 scripts/run_eval.py --dataset " + str(out_path))


if __name__ == "__main__":
    main()