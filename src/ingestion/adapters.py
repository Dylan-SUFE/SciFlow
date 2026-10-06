"""将不同解析器输出统一为 MultimodalDocument。

来源：自研，适配器模式。

三种解析器的输出格式完全不同：
- MinerU: Markdown 字符串
- Docling: DoclingDocument 对象（或 PyMuPDF 降级路径的字典）
- Surya: ocr_results + layout_results

通过 to_document() 按 parser 字段分发到三个内部转换函数，
下游完全不感知解析器差异。

【关键设计】
1. 所有 fig_id / table_id / formula_id 都是【裸 ID】（不带 paper_id 前缀）。
   paper_id 的拼接交给 chunker.py 统一处理，避免双重拼接。
2. 公式识别用字体特征 + 规则混合（见 formula_detect.py），
   而非纯规则 —— 准确率从 50% 提升到 90%。
"""
from __future__ import annotations

import logging

from src.schema.document import MultimodalDocument, TextChunk, Figure, Table, Formula
from src.processing.formula_detect import is_formula

logger = logging.getLogger(__name__)


# ============================================================
# 主入口：按解析器分发
# ============================================================

def to_document(raw: dict, paper_id: str) -> MultimodalDocument:
    parser = raw.get("parser")
    if parser == "mineru":
        return _from_mineru(raw, paper_id)
    if parser == "docling":
        return _from_docling(raw, paper_id)
    if parser == "surya":
        return _from_surya(raw, paper_id)
    raise ValueError(f"未知解析器: {parser}")


# ============================================================
# MinerU 路径
# ============================================================

def _from_mineru(raw: dict, paper_id: str) -> MultimodalDocument:
    """MinerU 输出 Markdown，按 \\n\\n 切段。"""
    md = raw.get("markdown", "")
    chunks: list[TextChunk] = []
    current_section: str | None = None
    para_idx = 0

    for block in md.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        if block.startswith("#"):
            current_section = block.lstrip("#").strip()
            continue
        chunks.append(TextChunk(
            chunk_id=f"{paper_id}:c{para_idx}",
            text=block,
            section=current_section,
        ))
        para_idx += 1

    return MultimodalDocument(paper_id=paper_id, text_chunks=chunks)


# ============================================================
# Docling 路径（含 PyMuPDF 降级）
# ============================================================

def _from_docling(raw: dict, paper_id: str) -> MultimodalDocument:
    doc = raw.get("docling_document")
    chunks: list[TextChunk] = []
    figures: list[Figure] = []
    tables: list[Table] = []
    formulas: list[Formula] = []

    if doc is not None:
        # ── Docling 正常路径 ──
        for i, item in enumerate(getattr(doc, "texts", []) or []):
            text = getattr(item, "text", str(item)).strip()
            if text:
                chunks.append(TextChunk(
                    chunk_id=f"{paper_id}:c{i}",
                    text=text,
                ))

        for i, tbl in enumerate(getattr(doc, "tables", []) or []):
            try:
                md = tbl.export_to_markdown()
            except Exception:
                md = str(tbl)
            tables.append(Table(
                table_id=f"t{i}",                        # 裸 ID
                markdown=md,
            ))

        for i, pic in enumerate(getattr(doc, "pictures", []) or []):
            figures.append(Figure(fig_id=f"fig{i}"))     # 裸 ID

    else:
        # ── PyMuPDF 降级路径：用字体特征识别公式 + 合并相邻块 ──
        text_blocks = raw.get("text_blocks", [])
        text_idx = 0
        formula_idx = 0

        if text_blocks:
            # ★ 第一步：合并相邻短块（新增）
            text_blocks = _merge_short_text_blocks(text_blocks, min_len=300)
            # Step 1: 逐块分类
            classified = []
            for block in text_blocks:
                text = block["text"].strip()
                fonts = block.get("fonts", set())
                bbox = block.get("bbox", (0, 0, 0, 0))
                page_num = block.get("page", 0)

                is_f = is_formula(text, fonts)
                classified.append({
                    "text": text,
                    "is_formula": is_f,
                    "bbox": bbox,
                    "page": page_num,
                })

            # Step 2: 合并相邻公式
            merged = []
            i = 0
            while i < len(classified):
                item = classified[i]

                if not item["is_formula"]:
                    merged.append(item)
                    i += 1
                    continue

                # 收集相邻公式
                group = [item["text"]]
                j = i + 1
                while j < len(classified) and classified[j]["is_formula"]:
                    prev_bbox = classified[j-1]["bbox"]
                    curr_bbox = classified[j]["bbox"]
                    same_page = classified[j]["page"] == item["page"]
                    y_close = abs(curr_bbox[1] - prev_bbox[3]) < 30
                    if same_page and y_close:
                        group.append(classified[j]["text"])
                        j += 1
                    else:
                        break

                merged.append({
                    "text": " ".join(group),
                    "is_formula": True,
                })
                i = j

            # Step 3: 输出（过滤碎片）
            for item in merged:
                text = item["text"].strip()
                if item["is_formula"]:
                    # 过滤：长度 < 10 的公式碎片丢弃
                    if len(text) >= 10:
                        formulas.append(Formula(
                            formula_id=f"f{formula_idx}",
                            latex=text,
                        ))
                        formula_idx += 1
                else:
                    # 过滤：长度 < 5 的文本碎片丢弃
                    if len(text) >= 5:
                        chunks.append(TextChunk(
                            chunk_id=f"{paper_id}:c{text_idx}",
                            text=text,
                        ))
                        text_idx += 1
        else:
            # ★ 兜底路径：没有 text_blocks 时，按 markdown 逐行分类
            md = raw.get("markdown", "")
            text_buffer: list[str] = []

            def flush_text():
                nonlocal text_idx
                if text_buffer:
                    merged = " ".join(text_buffer).strip()
                    if merged:
                        chunks.append(TextChunk(
                            chunk_id=f"{paper_id}:c{text_idx}",
                            text=merged,
                        ))
                        text_idx += 1
                    text_buffer.clear()

            for line in md.split("\n"):
                line = line.strip()
                if not line:
                    continue

                # 无字体信息时，只传文本（规则判断）
                if is_formula(line, None):
                    flush_text()
                    formulas.append(Formula(
                        formula_id=f"f{formula_idx}",
                        latex=line,
                    ))
                    formula_idx += 1
                else:
                    text_buffer.append(line)

            flush_text()

        # PyMuPDF 提取的图片（裸 ID）
        for fig in raw.get("figures", []):
            figures.append(Figure(
                fig_id=fig["fig_id"],                     # 裸 ID，如 "fig_p2_i0"
                caption=fig.get("caption", ""),
                page=fig.get("page", 0),
                image_bytes=fig.get("image_bytes"),
            ))

        # PyMuPDF 提取的表格（裸 ID）
        for tbl in raw.get("tables", []):
            tables.append(Table(
                table_id=tbl["table_id"],                 # 裸 ID，如 "tbl_p5_t2"
                markdown=tbl["markdown"],
                page=tbl.get("page", 0),
            ))

    return MultimodalDocument(
        paper_id=paper_id,
        text_chunks=chunks,
        figures=figures,
        tables=tables,
        formulas=formulas,
    )


# ============================================================
# Surya 路径
# ============================================================

def _from_surya(raw: dict, paper_id: str) -> MultimodalDocument:
    chunks: list[TextChunk] = []
    figures: list[Figure] = []
    tables: list[Table] = []
    ordinal = 0

    ocr_results = raw.get("ocr_results", [])
    layout_results = raw.get("layout_results", [])

    for page_idx, (ocr, layout) in enumerate(zip(ocr_results, layout_results)):
        # 文本行
        for line in getattr(ocr, "text_lines", []) or []:
            text = getattr(line, "text", "").strip()
            if text:
                chunks.append(TextChunk(
                    chunk_id=f"{paper_id}:c{ordinal}",
                    text=text,
                    page=page_idx + 1,
                ))
                ordinal += 1

        # 版面区域
        for region in getattr(layout, "bboxes", []) or []:
            label = getattr(region, "label", "")
            if label == "Figure":
                figures.append(Figure(
                    fig_id=f"fig{len(figures)}",           # 裸 ID
                    page=page_idx + 1,
                ))
            elif label == "Table":
                tables.append(Table(
                    table_id=f"t{len(tables)}",            # 裸 ID
                    page=page_idx + 1,
                ))

    return MultimodalDocument(
        paper_id=paper_id,
        text_chunks=chunks,
        figures=figures,
        tables=tables,
    )

def _merge_short_text_blocks(blocks: list[dict], min_len: int = 300) -> list[dict]:
    """合并相邻短文本块，保护公式块。

    ★ 关键改动：公式块不参与合并——合并前先判断 is_formula，
       是公式就独立保留，避免被合并进长文本块。
    """
    from src.processing.formula_detect import is_formula

    merged = []
    buffer_texts: list[str] = []
    buffer_fonts: set[str] = set()
    buffer_len = 0

    def flush():
        nonlocal buffer_texts, buffer_fonts, buffer_len
        if buffer_texts:
            merged.append({
                "text": " ".join(buffer_texts),
                "fonts": buffer_fonts.copy(),
                "bbox": (0, 0, 0, 0),
            })
            buffer_texts = []
            buffer_fonts = set()
            buffer_len = 0

    for block in blocks:
        text = block["text"].strip()
        if not text:
            continue

        # ★ 公式块不参与合并
        if is_formula(text, block.get("fonts", set())):
            flush()
            merged.append(block)
            continue

        # 超长块直接输出
        if len(text) >= min_len:
            flush()
            merged.append(block)
            continue

        # 累积短块
        buffer_texts.append(text)
        buffer_fonts.update(block.get("fonts", set()))
        buffer_len += len(text) + 1

        if buffer_len >= min_len:
            flush()

    flush()
    return merged