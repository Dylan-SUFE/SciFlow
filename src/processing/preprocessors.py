"""分模态预处理：让不同模态的内容进入 embedder 前先做特定清洗。

来源：自研。
"""
from __future__ import annotations

import re


# ============================================================
# 公式：LaTeX 规范化
# ============================================================

# LaTeX 符号 → 自然语言
_LATEX_MAP = {
    r"\frac": "divided by",
    r"\sum": "sum of",
    r"\int": "integral of",
    r"\alpha": "alpha",
    r"\beta": "beta",
    r"\gamma": "gamma",
    r"\delta": "delta",
    r"\lambda": "lambda",
    r"\mu": "mu",
    r"\sigma": "sigma",
    r"\theta": "theta",
    r"\pi": "pi",
    r"\infty": "infinity",
    r"\partial": "partial",
    r"\nabla": "gradient",
    r"\approx": "approximately equal to",
    r"\leq": "less than or equal to",
    r"\geq": "greater than or equal to",
    r"\neq": "not equal to",
    r"\times": "times",
    r"\cdot": "dot",
    r"\pm": "plus or minus",
    r"\rightarrow": "yields",
    r"\Rightarrow": "implies",
}


def preprocess_formula(latex: str) -> str:
    """把 LaTeX 公式转成更接近自然语言的形式，便于 bge-small 理解。

    例：
        '\\frac{dN}{dt} = rN(1 - \\frac{N}{K})'
        → 'dN divided by dt = rN(1 - N divided by K)'
    """
    text = latex

    # 1. 替换 LaTeX 符号为自然语言
    for latex_sym, natural in _LATEX_MAP.items():
        text = text.replace(latex_sym, f" {natural} ")

    # 2. 处理上标 x^{2} → x squared / x to the power of 2
    def _replace_superscript(m):
        content = m.group(1)
        return f" to the power of {content}"

    text = re.sub(r"\^\{([^}]+)\}", _replace_superscript, text)
    text = re.sub(r"\^(\w)", r" to the power of \1", text)

    # 3. 处理下标 x_{i} → x subscript i
    text = re.sub(r"_\{([^}]+)\}", r" subscript \1", text)
    text = re.sub(r"_(\w)", r" subscript \1", text)

    # 4. 去掉多余的大括号
    text = text.replace("{", " ").replace("}", " ")

    # 5. 清理多余空格
    text = re.sub(r"\s+", " ", text).strip()

    # 6. 兜底：如果转换后还是纯符号，加一句自然语言前缀
    if not re.search(r"[a-zA-Z]{3,}", text):
        return f"mathematical formula: {text}"

    return f"formula: {text}"


# ============================================================
# 表格：行级拆分 + 表头拼接
# ============================================================

def preprocess_table(markdown: str, caption: str = "") -> list[str]:
    """把 Markdown 表格拆成行级 chunk，每行都带上表头。

    例：
        输入：
        | Method | Accuracy | F1 |
        | --- | --- | --- |
        | BERT | 92.3 | 91.5 |
        | GPT-3 | 94.1 | 93.7 |

        输出：
        [
            "table: Method=BERT, Accuracy=92.3, F1=91.5",
            "table: Method=GPT-3, Accuracy=94.1, F1=93.7",
        ]

    Args:
        markdown: Markdown 格式的表格
        caption: 表格的标题（可选，会拼到每行前面）

    Returns:
        每行一条文本，用于独立 embedding。
    """
    lines = [l.strip() for l in markdown.strip().split("\n") if l.strip()]
    if len(lines) < 2:
        return [f"table: {caption} {markdown}".strip()]

    # 解析表头（第一行）
    header_line = lines[0]
    if not header_line.startswith("|"):
        return [f"table: {caption} {markdown}".strip()]

    headers = [h.strip() for h in header_line.strip("|").split("|")]

    # 跳过第二行的分隔符（| --- | --- |）
    data_lines = lines[2:] if len(lines) > 2 and "---" in lines[1] else lines[1:]

    rows = []
    for line in data_lines:
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        # 拼接表头和数据
        parts = []
        for h, c in zip(headers, cells):
            if c:
                parts.append(f"{h}={c}")
        if parts:
            prefix = f"table ({caption}): " if caption else "table: "
            rows.append(prefix + ", ".join(parts))

    # 兜底：如果解析失败，返回原 Markdown
    if not rows:
        return [f"table: {caption} {markdown}".strip()]

    return rows


# ============================================================
# 图表：caption + 图内文字 + 关键词
# ============================================================

def preprocess_figure(
    caption: str,
    figure_label: str | None = None,
    extra_ocr_text: str = "",
) -> str:
    """图表的预处理：把 caption、图注、图内 OCR 文字组合起来。

    Args:
        caption: 图注文本（来自 XML/解析器）
        figure_label: 图表的标签，如 "Figure 3"
        extra_ocr_text: 从图像 OCR 出的图内文字（可选）

    Returns:
        组合后的文本，用于 embedding。
    """
    parts = []

    if figure_label:
        parts.append(f"figure {figure_label}")

    if caption:
        parts.append(caption)

    if extra_ocr_text:
        parts.append(f"figure content: {extra_ocr_text}")

    if not parts:
        return "figure"

    return ": ".join(parts[:2]) + (f" {parts[2]}" if len(parts) > 2 else "")


# ============================================================
# 统一入口
# ============================================================

def preprocess_chunk(chunk) -> str:
    """根据 chunk 的 modality，分发到对应的预处理器。

    Args:
        chunk: 一个 Chunk 对象，有 .modality 和 .content 属性

    Returns:
        预处理后的文本，用于 embedding。
    """
    modality = getattr(chunk, "modality", "text")
    content = getattr(chunk, "content", "")

    if modality == "formula":
        return preprocess_formula(content)
    if modality == "table":
        # 表格返回的是多行，这里只取第一行用于主 embedding
        rows = preprocess_table(content)
        return rows[0] if rows else content
    if modality == "figure":
        return preprocess_figure(
            caption=content,
            figure_label=getattr(chunk, "figure_label", None),
        )
    # text 和其他：原样返回
    return content