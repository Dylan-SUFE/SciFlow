"""统一多模态文档模型。
来源：自研，参考 MMORE 的 MultimodalSample 设计。
"""
from dataclasses import dataclass, field


@dataclass
class TextChunk:
    chunk_id: str
    text: str
    section: str | None = None
    page: int = 0
    char_offset: int = 0


@dataclass
class Figure:
    fig_id: str
    caption: str = ""
    image_uri: str | None = None
    page: int = 0
    image_bytes: bytes | None = None


@dataclass
class Table:
    table_id: str
    markdown: str = ""
    caption: str | None = None
    page: int = 0


@dataclass
class Formula:
    formula_id: str
    latex: str = ""
    page: int = 0


@dataclass
class MultimodalDocument:
    paper_id: str
    title: str = ""
    text_chunks: list[TextChunk] = field(default_factory=list)
    figures: list[Figure] = field(default_factory=list)
    tables: list[Table] = field(default_factory=list)
    formulas: list[Formula] = field(default_factory=list)