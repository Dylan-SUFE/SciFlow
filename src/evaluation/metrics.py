"""纯函数指标计算。

来源：参考 central-dogma 的 evaluation/metrics.py。

包含四类指标：
1. 检索层：recall@k, MRR
2. 引用层：citation precision, citation recall
3. 生成层：groundedness
4. 拒绝层：RefusalTally 混淆矩阵

所有函数都是纯函数——不碰 store、不碰模型，只吃 id/string/citation，
返回数字或 None。未定义时返回 None，聚合时跳过 None 而不是计为 0。
"""
from __future__ import annotations

import re
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass


# ============================================================
# 检索层
# ============================================================

def recall_at_k(
    retrieved_ids: Sequence[str], gold_ids: Collection[str], k: int
) -> float | None:
    """top-k 里命中了多少 gold。

    None when there are no gold ids (recall is undefined for an
    unanswerable case).
    """
    gold = set(gold_ids)
    if not gold:
        return None
    topk = set(retrieved_ids[:k])
    return len(gold & topk) / len(gold)


def reciprocal_rank(
    retrieved_ids: Sequence[str], gold_ids: Collection[str]
) -> float:
    """1 / rank of the first retrieved gold source (0.0 if none retrieved)."""
    gold = set(gold_ids)
    for rank, sid in enumerate(retrieved_ids, start=1):
        if sid in gold:
            return 1.0 / rank
    return 0.0


# ============================================================
# 引用层
# ============================================================

def citation_precision(
    cited_ids: Collection[str], gold_ids: Collection[str]
) -> float | None:
    """Of the sources the answer cited, the fraction that are gold.

    None when nothing was cited (precision is undefined with an empty set).
    """
    cited = list(cited_ids)
    if not cited:
        return None
    gold = set(gold_ids)
    return sum(1 for c in cited if c in gold) / len(cited)


def citation_recall(
    cited_ids: Collection[str], gold_ids: Collection[str]
) -> float | None:
    """Of the gold sources, the fraction the answer cited.

    None when there are no gold ids.
    """
    gold = set(gold_ids)
    if not gold:
        return None
    return len(gold & set(cited_ids)) / len(gold)


# ============================================================
# 生成层：groundedness
# ============================================================

_MARKER = re.compile(r"\[([^\[\]]+)\]")


def inline_markers(answer: str) -> list[str]:
    """Bracketed inline citations in an answer.

    Example: '[Figure 3] and [Methods]' -> ['Figure 3', 'Methods']
    """
    return [m.strip() for m in _MARKER.findall(answer)]


def _marker_labels(citations) -> set[str]:
    """从 citations 里提取所有可以作为合法标记的字符串。

    包括：
    - source_id（chunk_id / figure_id）— 支持 ExtractiveGenerator 的格式
    - section（如 "Methods"）
    - figure_label（如 "Figure 3"）
    """
    labels: set[str] = set()
    for c in citations:
        # 新增：把 source_id 也算合法标记
        if getattr(c, "source_id", None):
            labels.add(c.source_id.lower())
        if getattr(c, "section", None):
            labels.add(c.section.lower())
        if getattr(c, "figure_label", None):
            labels.add(c.figure_label.lower())
    return labels


def _marker_supported(marker: str, labels: set[str]) -> bool:
    """Check whether a marker matches any of the labels."""
    m = marker.lower().strip()
    if not m:
        return False
    return any(m == lab or lab in m or m in lab for lab in labels)


def groundedness(answer: str, citations) -> float | None:
    """Fraction of the answer's inline citation markers that name a real source.

    Measures whether the model cites *what it was given*: every
    '[Methods]' / '[Figure 3]' / '[chunk_id]' marker must match a
    section, figure_label, or source_id present in the citations.

    None when the answer carries no markers (undefined).
    """
    markers = inline_markers(answer)
    if not markers:
        return None

    labels = _marker_labels(citations)
    if not labels:
        return None

    supported = sum(1 for m in markers if _marker_supported(m, labels))
    return supported / len(markers)


# ============================================================
# 拒绝层：RefusalTally 混淆矩阵
# ============================================================

@dataclass
class RefusalTally:
    """Confusion counts for the answer/refuse decision.

    Feed it (answerable, status) pairs; read precision/recall/accuracy off it.
    """

    answered_when_answerable: int = 0      # 正确回答
    refused_when_answerable: int = 0       # 过度拒绝（漏答）
    refused_when_unanswerable: int = 0     # 正确拒绝
    answered_when_unanswerable: int = 0    # 幻觉风险

    def add(self, *, answerable: bool, status: str) -> None:
        answered = status == "answered"
        if answerable and answered:
            self.answered_when_answerable += 1
        elif answerable and not answered:
            self.refused_when_answerable += 1
        elif not answerable and not answered:
            self.refused_when_unanswerable += 1
        else:
            self.answered_when_unanswerable += 1

    @property
    def total(self) -> int:
        return (
            self.answered_when_answerable
            + self.refused_when_answerable
            + self.refused_when_unanswerable
            + self.answered_when_unanswerable
        )

    @property
    def answer_accuracy(self) -> float | None:
        """Fraction of cases where the answer/refuse decision was correct."""
        if self.total == 0:
            return None
        correct = self.answered_when_answerable + self.refused_when_unanswerable
        return correct / self.total

    @property
    def refusal_recall(self) -> float | None:
        """Of the unanswerable cases, the fraction correctly refused."""
        unanswerable = (
            self.refused_when_unanswerable + self.answered_when_unanswerable
        )
        if unanswerable == 0:
            return None
        return self.refused_when_unanswerable / unanswerable

    @property
    def refusal_precision(self) -> float | None:
        """Of the cases refused, the fraction that were truly unanswerable."""
        refused = (
            self.refused_when_unanswerable + self.refused_when_answerable
        )
        if refused == 0:
            return None
        return self.refused_when_unanswerable / refused


# ============================================================
# 聚合辅助
# ============================================================

def mean(values: Iterable[float | None]) -> float | None:
    """Mean over the defined (non-None) values; None if none are defined."""
    defined = [v for v in values if v is not None]
    if not defined:
        return None
    return sum(defined) / len(defined)