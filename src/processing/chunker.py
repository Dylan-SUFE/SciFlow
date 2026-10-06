"""文本切块 + 图表/表格转 chunk。
来源：参考 central-dogma 的 chunk.py。
"""
from __future__ import annotations
import logging

from dataclasses import dataclass

import numpy as np

from src.schema.document import MultimodalDocument
from src.processing.preprocessors import preprocess_table

logger = logging.getLogger(__name__)

@dataclass
class Chunk:
    chunk_id: str
    paper_id: str
    modality: str  # "text" | "figure" | "table"
    content: str
    section: str | None = None
    image_uri: str | None = None
    embedding: np.ndarray | None = None


def _split_with_overlap(text: str, max_chars: int = 1200, overlap: int = 150) -> list[str]:
    """滑动窗口切分，优先在空格处断开。"""
    if len(text) <= max_chars:
        return [text] if text.strip() else []

    out: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        if end < n:
            window = text.rfind(" ", start + max_chars - overlap, end)
            if window != -1 and window > start:
                end = window
        piece = text[start:end].strip()
        if piece:
            out.append(piece)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return out


def chunk_document(doc: MultimodalDocument) -> list[Chunk]:
    chunks: list[Chunk] = []
    ordinal = 0

    # 文本侧
    for tc in doc.text_chunks:
        for piece in _split_with_overlap(tc.text):
            chunks.append(Chunk(
                chunk_id=f"{doc.paper_id}:c{ordinal}",
                paper_id=doc.paper_id,
                modality="text",
                content=piece,
                section=tc.section,
            ))
            ordinal += 1

    # 图表侧
    for fig in doc.figures:
        content = fig.caption or f"Figure {fig.fig_id}"
        chunks.append(Chunk(
            chunk_id=f"{doc.paper_id}:{fig.fig_id}",
            paper_id=doc.paper_id,
            modality="figure",
            content=content,
            image_uri=fig.image_uri,
        ))
        ordinal += 1

    # 表格侧：改成每行一个 chunk
    for tbl in doc.tables:
        rows = preprocess_table(tbl.markdown, caption=tbl.caption or "")
        for i, row in enumerate(rows):
            chunks.append(Chunk(
                chunk_id=f"{doc.paper_id}:{tbl.table_id}:r{i}",
                paper_id=doc.paper_id,
                modality="table",
                content=row,
            ))
            ordinal += 1

    # 公式侧
    for formula in doc.formulas:
        chunks.append(Chunk(
            chunk_id=f"{doc.paper_id}:{formula.formula_id}",    # ← 加前缀
            paper_id=doc.paper_id,
            modality="formula",
            content=formula.latex,
        ))
        ordinal += 1

    MIN_LEN = {
        "text": 50,       # 文本至少 50 字符
        "figure": 5,      # caption 可以短
        "table": 10,      # 表格行
        "formula": 5,     # 公式
    }
    
    filtered = []
    for c in chunks:
        min_len = MIN_LEN.get(c.modality, 20)
        if len(c.content.strip()) >= min_len:
            filtered.append(c)
    
    logger.info(f"  过滤前 {len(chunks)} 个，过滤后 {len(filtered)} 个（移除 {len(chunks)-len(filtered)} 碎片）")

    return chunks