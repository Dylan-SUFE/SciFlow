"""演示用例。
来源：自研，参考 central-dogma 的 demo_cases。
"""
from __future__ import annotations


def get_demo_cases() -> list[dict]:
    """返回演示用例（不依赖具体语料，仅用于跑通流程）。"""
    return [
        {
            "question": "What is the main finding of this paper?",
            "gold_source_ids": [],
            "answerable": True,
        },
        {
            "question": "What method was used in the experiments?",
            "gold_source_ids": [],
            "answerable": True,
        },
        {
            "question": "Which antibiotic prevents neonatal infection?",
            "gold_source_ids": [],
            "answerable": False,
        },
    ]