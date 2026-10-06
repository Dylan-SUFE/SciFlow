"""用真实评估集跑 RAG 评估（完整版）。

包含 central-dogma 的四种指标：
1. 检索层：recall@k, MRR
2. 引用层：citation precision, citation recall
3. 生成层：groundedness
4. 拒绝层：RefusalTally 混淆矩阵

用法：
    python3 -m scripts.run_eval --dataset evaluation/datasets/llm_eval.jsonl
    python3 -m scripts.run_eval --dataset evaluation/datasets/llm_eval.jsonl --show-cases
    python3 -m scripts.run_eval --dataset evaluation/datasets/llm_eval.jsonl --json out.json
"""
from __future__ import annotations

import sys
from pathlib import Path

# 把项目根目录加入 Python 搜索路径，支持从任意位置执行
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import json
import re

from src.evaluation.metrics import (
    citation_precision,
    citation_recall,
    groundedness,
    mean,
    recall_at_k,
    reciprocal_rank,
    RefusalTally,
)
from src.indexing.bm25_store import BM25Store
from src.indexing.hybrid_retriever import HybridRetriever
from src.indexing.milvus_store import MilvusStore
from src.rag.generation import build_generator
from src.rag.pipeline import answer_question


# ============================================================
# 辅助函数
# ============================================================

def load_cases(path: str) -> list[dict]:
    """从 JSONL 加载评估用例。"""
    cases = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            cases.append(json.loads(line))
    return cases


def extract_cited_ids(answer: str, citations) -> list[str]:
    """从答案文本里提取真正被引用的 source_id。

    支持三种标记格式：
    - [2610.01130v1:c5]     ← chunk_id
    - [Methods]              ← section
    - [Figure 3]             ← figure_label

    如果答案里没有可识别的标记，退回到所有 citations 的 source_id
    （兼容不支持引用的生成器）。
    """
    markers = re.findall(r"\[([^\[\]]+)\]", answer)
    if not markers:
        return [c.source_id for c in citations]

    # 建立三张匹配表
    id_set = {c.source_id for c in citations}
    section_to_id = {
        c.section.lower(): c.source_id
        for c in citations
        if getattr(c, "section", None)
    }
    figure_to_id = {
        c.figure_label.lower(): c.source_id
        for c in citations
        if getattr(c, "figure_label", None)
    }

    cited = []
    for m in markers:
        m_strip = m.strip()
        m_lower = m_strip.lower()

        if m_strip in id_set:
            cited.append(m_strip)
        elif m_lower in section_to_id:
            cited.append(section_to_id[m_lower])
        elif m_lower in figure_to_id:
            cited.append(figure_to_id[m_lower])
        # 其他标记（比如 LLM 编造的）忽略，不计入 citation metric

    # 如果所有标记都无法匹配，退回全量（兼容旧行为）
    if not cited:
        return [c.source_id for c in citations]

    return cited


def fmt_pct(v: float | None) -> str:
    """格式化百分比，None 显示 n/a。"""
    return f"{v * 100:5.1f}%" if v is not None else " n/a "


def print_report(
    results: list[dict],
    tally: RefusalTally,
    *,
    k: int,
    dataset: str,
    show_cases: bool = False,
) -> None:
    """打印评估报告。"""

    # 汇总指标
    avg_recall = mean([r["recall"] for r in results])
    avg_mrr = mean([r["rr"] for r in results if r["answerable"]])
    avg_cite_p = mean([r["cite_p"] for r in results])
    avg_cite_r = mean([r["cite_r"] for r in results])
    avg_grounded = mean([r["grounded"] for r in results])

    print()
    print("=" * 72)
    print(f"评估报告 — {dataset}")
    print("=" * 72)
    print(f"用例数: {len(results)}  (k = {k})")
    print()

    print("| 指标 | 值 |")
    print("| --- | ---: |")
    print(f"| recall@{k} | {fmt_pct(avg_recall)} |")
    print(f"| MRR | {fmt_pct(avg_mrr)} |")
    print(f"| citation precision | {fmt_pct(avg_cite_p)} |")
    print(f"| citation recall | {fmt_pct(avg_cite_r)} |")
    print(f"| groundedness | {fmt_pct(avg_grounded)} |")
    print(f"| refusal recall | {fmt_pct(tally.refusal_recall)} |")
    print(f"| refusal precision | {fmt_pct(tally.refusal_precision)} |")
    print(f"| answer accuracy | {fmt_pct(tally.answer_accuracy)} |")
    print()

    print("### 拒绝混淆矩阵")
    print()
    print("| 类型 | 数量 |")
    print("| --- | ---: |")
    print(f"| 正确回答 (answered & answerable) | {tally.answered_when_answerable} |")
    print(f"| 过度拒绝 (refused & answerable) | {tally.refused_when_answerable} |")
    print(f"| 正确拒绝 (refused & unanswerable) | {tally.refused_when_unanswerable} |")
    print(f"| 幻觉风险 (answered & unanswerable) | {tally.answered_when_unanswerable} |")
    print()

    if show_cases:
        print("### 逐题明细")
        print()
        print("| 问题 | 可答 | 状态 | recall | RR | cite_p | grounded |")
        print("| --- | :-: | :-: | ---: | ---: | ---: | ---: |")
        for r in results:
            q = r["question"][:45] + ("..." if len(r["question"]) > 45 else "")
            rec = fmt_pct(r["recall"]).strip()
            rr = f"{r['rr']:.2f}"
            cp = fmt_pct(r["cite_p"]).strip()
            gd = fmt_pct(r["grounded"]).strip()
            print(
                f"| {q} | {'Y' if r['answerable'] else 'N'} | {r['status']} "
                f"| {rec} | {rr} | {cp} | {gd} |"
            )
        print()


def to_json_dict(
    results: list[dict], tally: RefusalTally, *, k: int, dataset: str
) -> dict:
    """序列化为 JSON 兼容格式。"""
    return {
        "dataset": dataset,
        "k": k,
        "cases": len(results),
        "aggregate": {
            "recall_at_k": mean([r["recall"] for r in results]),
            "mrr": mean([r["rr"] for r in results if r["answerable"]]),
            "citation_precision": mean([r["cite_p"] for r in results]),
            "citation_recall": mean([r["cite_r"] for r in results]),
            "groundedness": mean([r["grounded"] for r in results]),
            "refusal_recall": tally.refusal_recall,
            "refusal_precision": tally.refusal_precision,
            "answer_accuracy": tally.answer_accuracy,
        },
        "refusal_tally": {
            "answered_when_answerable": tally.answered_when_answerable,
            "refused_when_answerable": tally.refused_when_answerable,
            "refused_when_unanswerable": tally.refused_when_unanswerable,
            "answered_when_unanswerable": tally.answered_when_unanswerable,
        },
        "results": [
            {
                "question": r["question"],
                "answerable": r["answerable"],
                "status": r["status"],
                "gold": r["gold"],
                "retrieved": r["retrieved"],
                "cited": r["cited"],
                "recall_at_k": r["recall"],
                "reciprocal_rank": r["rr"],
                "citation_precision": r["cite_p"],
                "citation_recall": r["cite_r"],
                "groundedness": r["grounded"],
            }
            for r in results
        ],
    }


# ============================================================
# 主流程
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="SciFlow 评估执行器")
    parser.add_argument("--dataset", required=True, help="评估集 JSONL 路径")
    parser.add_argument("--k", type=int, default=5, help="top-k 检索数")
    parser.add_argument("--show-cases", action="store_true", help="打印逐题明细")
    parser.add_argument("--json", dest="json_out", default=None, help="导出 JSON 报告")
    args = parser.parse_args()

    cases = load_cases(args.dataset)
    print(f"加载 {len(cases)} 条用例")

    # 构建 retriever 和 generator
    milvus = MilvusStore()
    bm25 = BM25Store().load()
    retriever = HybridRetriever(milvus, bm25)
    generator = build_generator("deepseek")

    # 跑评估
    results: list[dict] = []
    tally = RefusalTally()

    for i, case in enumerate(cases, 1):
        question = case["question"]
        gold = case.get("gold_source_ids", [])
        answerable = case.get("answerable", True)

        # 检索
        hits = retriever.search(question, top_k=args.k)
        retrieved_ids = [h["id"] for h in hits]

        # RAG 生成
        resp = answer_question(question, retriever, generator)

        # 从答案里提取真正被引用的源
        cited_ids = extract_cited_ids(resp.answer, resp.citations)

        # 累计混淆矩阵
        tally.add(answerable=answerable, status=resp.status)

        results.append({
            "question": question,
            "answerable": answerable,
            "status": resp.status,
            "gold": gold,
            "retrieved": retrieved_ids,
            "cited": cited_ids,
            "answer": resp.answer,
            "citations": resp.citations,
            "recall": recall_at_k(retrieved_ids, gold, args.k),
            "rr": reciprocal_rank(retrieved_ids, gold),
            "cite_p": citation_precision(cited_ids, gold) if answerable else None,
            "cite_r": citation_recall(cited_ids, gold) if answerable else None,
            "grounded": groundedness(resp.answer, resp.citations) if answerable else None,
        })

        # 实时进度
        rec = fmt_pct(results[-1]["recall"]).strip()
        print(f"  [{i}/{len(cases)}] [{resp.status:<8}] recall={rec:<6} | {question[:500]} | {resp.answer[:500]}")

    # 打印报告
    print_report(
        results, tally,
        k=args.k,
        dataset=args.dataset,
        show_cases=args.show_cases,
    )

    # 导出 JSON
    if args.json_out:
        report = to_json_dict(results, tally, k=args.k, dataset=args.dataset)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        print(f"已导出 JSON 报告到 {args.json_out}")


if __name__ == "__main__":
    main()