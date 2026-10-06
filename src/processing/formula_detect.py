"""公式识别：字体特征 + 规则混合（v3 最终版）。

核心原理：
PDF 里公式通常用特殊字体（Computer Modern 系列、Symbol 系列）。
PyMuPDF 的 get_text("dict") 能拿到每个 span 的字体名。

判据（按强度排序）：
① 多字母单词数 >= 3 → 直接排除（自然语言特征）
② 短文本（<= 25 字符）+ 任意数学信号 → 公式
③ 强数学运算符（∫ ∑ √ 等）>= 2 → 公式
④ LaTeX 残留 → 公式
⑤ Unicode 数学符号 >= 3 + 自然语言词 < 2 → 公式
⑥ ASCII 符号密度 >= 0.2 + 无多字母单词 → 公式
⑦ 字体全为数学字体 + 无多字母单词 → 公式

关键改进（相比 v2）：
- 新增"多字母单词数"快速排除
- 新增"短文本快速判定"（解决 E = mc² 之类）
- 新增"ASCII 符号密度"（解决 dN/dt 之类纯 ASCII 公式）
"""
from __future__ import annotations

import re


# ============================================================
# 字体匹配
# ============================================================

_MATH_FONT_PATTERN = re.compile(
    r"("
    r"cmr|cmmi|cmsy|cmex|cmbx|cmti|cmss|cmb|cmdunh|"
    r"symbol|mtextra|stix|stixgeneral|mathjax|"
    r"mtmi|mtsy|mtm|mtext|"
    r"euclid|euler|rsfs|"
    r"math"
    r")",
    re.IGNORECASE,
)

# Unicode 数学符号集
_MATH_SYMBOLS = set(
    "∫∑∏√∛±×÷∂∇∞"
    "αβγδεζηθικλμνξπρστυφχψω"
    "ΓΔΘΛΞΠΣΦΨΩ"
    "≈≠≤≥≡∝∈∉⊂⊃∪∩∀∃∅"
    "→←⇒⇔↦↑↓"
    "⁰¹²³⁴⁵⁶⁷⁸⁹₀₁₂₃₄₅₆₇₈₉"
)

# 强数学运算符（更高置信度，不含希腊字母）
_STRONG_MATH_OPERATORS = set(
    "∫∑∏√∛±×÷∂∇∞"
    "≈≠≤≥≡∝∈∉⊂⊃∪∩∀∃∅"
    "→←⇒⇔↦"
    "⁰¹²³⁴⁵⁶⁷⁸⁹₀₁₂₃₄₅₆₇₈₉"
)

# ASCII 数学符号（用于纯 ASCII 公式）
_ASCII_MATH_CHARS = set("=+-*/^_(){}[]<>|")

# LaTeX 残留
_LATEX_RESIDUE = re.compile(r"\\[a-zA-Z]+|\^\{|_\{")

# 多字母单词（>= 4 个连续字母）
_LONG_WORD = re.compile(r"[a-zA-Z]{4,}")

# 自然语言常见词
_ENGLISH_WORDS = [
    "the", "is", "are", "was", "were", "we", "this", "that", "these", "those",
    "and", "or", "with", "from", "have", "has", "had", "been",
    "which", "what", "how", "does", "do", "did", "can", "could",
    "will", "would", "not", "but", "all", "any", "one", "two",
    "a", "an", "of", "to", "in", "on", "for", "by", "as", "at",
]


# ============================================================
# 核心函数
# ============================================================

def is_math_font(font_name: str) -> bool:
    """判断字体是否是数学字体。"""
    if not font_name:
        return False
    return bool(_MATH_FONT_PATTERN.search(font_name))


def _count_english_words(text: str) -> int:
    """统计自然语言常见词数量。"""
    text_lower = f" {text.lower()} "
    return sum(1 for w in _ENGLISH_WORDS if f" {w} " in text_lower)


def _count_math_symbols(text: str) -> int:
    """统计 Unicode 数学符号数。"""
    return sum(1 for ch in text if ch in _MATH_SYMBOLS)


def _count_strong_operators(text: str) -> int:
    """统计强数学运算符数。"""
    return sum(1 for ch in text if ch in _STRONG_MATH_OPERATORS)


def _count_ascii_math(text: str) -> int:
    """统计 ASCII 数学符号数。"""
    return sum(1 for ch in text if ch in _ASCII_MATH_CHARS)


def is_formula(text: str, fonts: set[str] | None = None) -> bool:
    """公式识别主函数（v5）。

    改进（相比 v4）：
    - 过滤数字标注（4.7×fewer）
    - 过滤占位符（{writing_prompt}）
    - 短文本需 ≥ 2 个数学信号，或含数学字体
    """
    text = text.strip()
    if not text or len(text) > 200:
        return False

    # ★ 快速过滤：多字母单词 ≥ 3 → 自然语言
    long_words = re.findall(r"[a-zA-Z]{4,}", text)
    if len(long_words) >= 3:
        return False

    # ★ 新增过滤 1：纯数字 + × 的标注（图的标注）
    # 例："4.7×fewer samples", "11.0×", "62.6×"
    if re.match(r"^[\d\.]+×", text):
        return False

    # ★ 新增过滤 2：纯占位符/标点
    # 例："{writing_prompt}", "{", "---"
    if re.match(r"^[\{\}\-_\.\s]+$", text):
        return False

    # ★ 新增过滤 3：LaTeX 提示词占位符
    if re.match(r"^\{[a-z_]+\}$", text):
        return False

    word_count = _count_english_words(text)
    symbol_count = _count_math_symbols(text)
    strong_op_count = _count_strong_operators(text)
    ascii_count = _count_ascii_math(text)
    has_latex = bool(_LATEX_RESIDUE.search(text))

    # ============================================================
    # 短文本判定（≤ 25 字符）
    # ============================================================
    if len(text) <= 25:
        # 统计数学信号数
        math_signals = (
            (1 if strong_op_count >= 1 else 0)
            + (1 if symbol_count >= 1 else 0)
            + (1 if has_latex else 0)
            + (1 if ascii_count >= 3 else 0)
        )

        # ★ 收紧：需 ≥ 2 个数学信号，或含数学字体
        if math_signals >= 2:
            return True
        if math_signals == 1:
            # 有字体信息时，需含数学字体
            if fonts and any(is_math_font(f) for f in fonts):
                return True
            # 无字体信息时，不判公式（避免误判）
            if not fonts:
                return False

    # ============================================================
    # 长文本判定
    # ============================================================
    if strong_op_count >= 2:
        return True
    if has_latex:
        return True
    if symbol_count >= 3 and word_count < 2:
        return True

    ascii_ratio = ascii_count / len(text)
    if ascii_ratio >= 0.2 and len(long_words) == 0:
        return True

    # 字体特征：全数学字体 + 无多字母单词
    if fonts:
        math_ratio = sum(1 for f in fonts if is_math_font(f)) / len(fonts)
        if math_ratio >= 0.8 and len(long_words) == 0:
            return True

    return False


def is_formula_by_font(font_name: str, text: str) -> bool:
    """单 span 判断。"""
    return is_formula(text, {font_name} if font_name else None)