"""RAG 数据契约。
来源：复用 central-dogma 的 api/schemas.py。
"""
from dataclasses import dataclass, field


@dataclass
class Citation:
    source_id: str
    score: float
    section: str | None = None
    snippet: str = ""


@dataclass
class AskResponse:
    answer: str
    citations: list[Citation] = field(default_factory=list)
    status: str = "answered"
    confidence: float | None = None