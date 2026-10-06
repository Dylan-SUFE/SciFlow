"""Docling 解析器封装。

来源：自研，调用 docling-project/docling 官方 API。

降级路径：PyMuPDF 增强版
  - 文本：reconstruct_page_text() 阅读顺序重建
  - 文本块（带字体）：_get_page_text_blocks() —— 用于公式识别
  - 图片：page.get_images() + page.get_image_rects() + bbox caption 关联
  - 表格：page.find_tables()
  - caption 关联策略：方案 C（bbox 位置匹配）+ 方案 A（正则兜底）

性能优化：
  - get_text("dict") 每页只调用一次（通过 _get_page_text_blocks 缓存）
"""
from __future__ import annotations

import logging
import re

from src.processing.reading_order import reconstruct_page_text

logger = logging.getLogger(__name__)


class DoclingParser:
    def parse(self, pdf_path: str) -> dict:
        try:
            from docling.document_converter import DocumentConverter
        except ImportError:
            logger.warning("Docling 未安装，使用 PyMuPDF 增强提取")
            return self._fallback_pymupdf(pdf_path)

        try:
            converter = DocumentConverter()
            result = converter.convert(pdf_path)
            doc = result.document
            return {
                "parser": "docling",
                "docling_document": doc,
                "markdown": doc.export_to_markdown(),
                "pdf_path": pdf_path,
            }
        except Exception as e:
            logger.warning(f"Docling 解析失败({e})，回退 PyMuPDF")
            return self._fallback_pymupdf(pdf_path)

    # ============================================================
    # PyMuPDF 降级路径
    # ============================================================

    @staticmethod
    def _fallback_pymupdf(pdf_path: str) -> dict:
        """PyMuPDF 增强版：文本 + 图片 + 表格 + caption 关联 + 字体信息。

        Returns:
            dict 包含：
            - markdown: 阅读顺序重建后的纯文本
            - text_blocks: 带字体信息的文本块列表（用于公式识别）
            - figures: 图表列表
            - tables: 表格列表
        """
        import pymupdf

        doc = pymupdf.open(pdf_path)
        text_parts: list[str] = []
        all_text_blocks: list[dict] = []       # ← 新增：所有页的文本块（带字体）
        figures: list[dict] = []
        tables: list[dict] = []

        for page_idx, page in enumerate(doc):
            # ---------- ① 阅读顺序重建后的纯文本 ----------
            try:
                page_text = reconstruct_page_text(page)
            except Exception as e:
                logger.debug(f"阅读顺序重建失败，回退 get_text(): {e}")
                page_text = page.get_text()
            text_parts.append(page_text)

            # ---------- ② 提取带字体信息的文本块（每页一次）----------
            page_text_blocks = _get_page_text_blocks(page)
            for block in page_text_blocks:
                block["page"] = page_idx + 1
            all_text_blocks.extend(page_text_blocks)

            # ---------- ③ 图片 + caption 关联 ----------
            try:
                for img_idx, img_info in enumerate(page.get_images(full=True)):
                    xref = img_info[0]
                    try:
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        # 过滤太小的图片（图标、logo、公式小图）
                        if len(image_bytes) <= 5000:
                            continue

                        # 获取图片在页面上的 bbox
                        img_rects = page.get_image_rects(xref)

                        # 方案 C：bbox 精确关联（复用已提取的 text_blocks）
                        caption = ""
                        if img_rects and page_text_blocks:
                            caption = _find_caption_by_bbox(
                                img_rects, page_text_blocks
                            )

                        # 方案 A 兜底：正则按顺序
                        if not caption:
                            caption = _find_caption_by_regex(
                                page_text, img_idx
                            )

                        figures.append({
                            "fig_id": f"fig_p{page_idx}_i{img_idx}",
                            "caption": caption,
                            "page": page_idx + 1,
                            "image_bytes": image_bytes,
                            "ext": base_image["ext"],
                        })
                    except Exception as e:
                        logger.debug(
                            f"图片提取失败 (page={page_idx}, xref={xref}): {e}"
                        )
                        continue
            except Exception as e:
                logger.debug(f"page.get_images 失败 (page={page_idx}): {e}")

            # ---------- ④ 表格 ----------
            try:
                tabs = page.find_tables()
                for tab_idx, tab in enumerate(tabs):
                    md = tab.to_markdown()
                    if md and len(md.strip()) > 20:
                        tables.append({
                            "table_id": f"tbl_p{page_idx}_t{tab_idx}",
                            "markdown": md,
                            "page": page_idx + 1,
                        })
            except Exception as e:
                logger.debug(f"表格提取失败 (page={page_idx}): {e}")

        doc.close()

        return {
            "parser": "docling",
            "docling_document": None,
            "markdown": "\n".join(text_parts),
            "pdf_path": pdf_path,
            "text_blocks": all_text_blocks,        # ← 新增：带字体的文本块
            "figures": figures,
            "tables": tables,
        }


# ============================================================
# 提取带字体信息的文本块（新增）
# ============================================================

def _get_page_text_blocks(page) -> list[dict]:
    """提取页面所有文本块（含 bbox + 字体集合）。

    每页只调用一次，结果复用于：
    - caption 关联（用 bbox）
    - 公式识别（用字体特征，见 adapters.py 的 _from_docling）

    Returns:
        [{"text": str, "fonts": set[str], "bbox": tuple}, ...]
    """
    try:
        text_dict = page.get_text("dict")
    except Exception:
        return []

    blocks = text_dict.get("blocks", [])
    if not blocks:
        return []

    result: list[dict] = []
    for block in blocks:
        if block.get("type") != 0:  # 只看文本块
            continue

        block_text = ""
        fonts: set[str] = set()

        for line in block.get("lines", []):
            for span in line.get("spans", []):
                block_text += span.get("text", "")
                font = span.get("font", "")
                if font:
                    fonts.add(font)

        block_text = block_text.strip()
        if block_text:
            result.append({
                "text": block_text,
                "fonts": fonts,
                "bbox": block.get("bbox", (0, 0, 0, 0)),
            })

    return result


# ============================================================
# Caption 关联：方案 C（bbox 位置匹配）
# ============================================================

_CAPTION_PATTERN = re.compile(
    r"^\s*(?:Figure|Fig\.?|FIGURE|Table|TABLE|Tab\.?)\s*(\d+)[.:\s]",
    re.IGNORECASE,
)


def _find_caption_by_bbox(img_rects: list, text_blocks: list[dict]) -> str:
    """根据图片 bbox，找它下方最近的 Figure/Table caption。

    策略：
    1. 接收已提取的文本块（含 bbox），避免重复调用 page.get_text("dict")
    2. 对每个图片，找位于图片下方 0-250px 内的文本块
    3. 过滤出以 Figure/Table 开头的块
    4. 选择水平方向有重叠、垂直距离最近的块作为 caption

    Args:
        img_rects: 图片在页面上的 bbox 列表（pymupdf.Rect 或 tuple）
        text_blocks: 页面所有文本块（已含 bbox 和字体）

    Returns:
        找到的 caption 文本（截断到 800 字符），找不到返回 ""
    """
    if not text_blocks:
        return ""

    best_caption = ""
    best_distance = float("inf")

    for img_rect in img_rects:
        # img_rect 可能是 pymupdf.Rect，也可能是 tuple
        if hasattr(img_rect, "x0"):
            ix0, iy0, ix1, iy1 = img_rect.x0, img_rect.y0, img_rect.x1, img_rect.y1
        else:
            ix0, iy0, ix1, iy1 = img_rect

        for block in text_blocks:
            bx0, by0, bx1, by1 = block["bbox"]

            # ① 必须位于图片下方
            if by0 < iy1:
                continue

            # ② 距离不能太远（250px 以内）
            distance = by0 - iy1
            if distance > 250:
                continue

            # ③ 水平方向必须有重叠
            #    如果图片和文本块在水平方向完全不重叠，说明是另一栏
            if bx1 < ix0 or bx0 > ix1:
                continue

            # ④ 必须以 Figure/Table 开头
            if not _CAPTION_PATTERN.match(block["text"]):
                continue

            # ⑤ 取最近的一个
            if distance < best_distance:
                best_distance = distance
                best_caption = block["text"]

    # 截断到 800 字符
    return best_caption[:800].strip()


# ============================================================
# Caption 关联：方案 A（正则按顺序，兜底）
# ============================================================

def _find_caption_by_regex(page_text: str, img_idx: int) -> str:
    """正则兜底：从页面文本里按顺序找第 img_idx 个 Figure/Table。

    仅在 bbox 关联失败时使用。
    """
    matches = list(_CAPTION_PATTERN.finditer(page_text))
    if not matches:
        return ""

    # 简单策略：按顺序对应
    m = matches[min(img_idx, len(matches) - 1)]
    start = m.start()

    # 截取从 Figure 开头到下一个换行或 800 字符
    snippet = page_text[start:start + 800]
    # 取到第一个空行为止
    first_para = snippet.split("\n\n")[0].split("\n")[0]

    return first_para[:800].strip()