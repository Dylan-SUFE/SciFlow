"""评估执行器。
来源：复用 central-dogma 的 evaluation/harness.py。
"""
from __future__ import annotations

from src.evaluation.metrics import recall_at_k, reciprocal_rank, mean
from src.rag.pipeline import answer_question


class EvalReport:
    def __init__(self, backend: str, results: list[dict]):
        self.backend = backend
        self.results = results

    @property
    def recall_at_k(self):
        return mean([r.get("recall_at_k") for r in self.results])

    @property
    def mrr(self):
        return mean([r.get("rr") for r in self.results if r.get("answerable", True)])

    @property
    def refusal_count(self):
        return sum(1 for r in self.results if r["status"] == "refused")

    def to_markdown(self) -> str:
        r_at_k = self.recall_at_k
        mrr = self.mrr
        lines = [
            f"# 评估报告 — {self.backend}",
            f"",
            f"用例数: {len(self.results)}",
            f"",
            f"| 指标 | 值 |",
            f"| --- | ---: |",
            f"| recall@5 | {f'{r_at_k:.2%}' if r_at_k is not None else 'n/a'} |",
            f"| MRR | {f'{mrr:.2%}' if mrr is not None else 'n/a'} |",
            f"| 拒答数 | {self.refusal_count} |",
            f"",
            f"## 逐题明细",
            f"",
            f"| 问题 | 状态 | recall | RR |",
            f"| --- | :-: | ---: | ---: |",
        ]
        for r in self.results:
            q = r["question"][:50]
            rec = f"{r['recall_at_k']:.2%}" if r.get("recall_at_k") is not None else "n/a"
            lines.append(f"| {q} | {r['status']} | {rec} | {r.get('rr', 0):.2f} |")
        return "\n".join(lines)


def evaluate(retriever, generator, cases, k: int = 5) -> EvalReport:
    results = []
    for case in cases:
        hits = retriever.search(case["question"], top_k=k)
        retrieved_ids = [h["id"] for h in hits]
        resp = answer_question(case["question"], retriever, generator)

        results.append({
            "question": case["question"],
            "answerable": case.get("answerable", True),
            "status": resp.status,
            "recall_at_k": recall_at_k(retrieved_ids, case.get("gold_source_ids", []), k),
            "rr": reciprocal_rank(retrieved_ids, case.get("gold_source_ids", [])),
        })

    return EvalReport(backend="sciflow", results=results)