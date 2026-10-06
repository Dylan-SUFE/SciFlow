"""对比自研阅读顺序重建 vs Surya LayoutPredictor。

用法：
    python3 -m scripts.compare_reading_order data/papers/xxx.pdf
"""
from __future__ import annotations

import sys
import time
from pathlib import Path


def method_self_pymupdf(pdf_path: str) -> tuple[str, float]:
    """方法 1：自研 PyMuPDF 阅读顺序重建。"""
    import pymupdf
    from src.processing.reading_order import reconstruct_page_text

    doc = pymupdf.open(pdf_path)
    start = time.time()
    parts = []
    for page in doc:
        parts.append(reconstruct_page_text(page))
    elapsed = time.time() - start
    doc.close()
    return "\n".join(parts), elapsed


def method_surya(pdf_path: str) -> tuple[str, float] | None:
    """方法 2：Surya LayoutPredictor（需要 GPU）。"""
    try:
        from surya.layout import LayoutPredictor
        from surya.recognition import RecognitionPredictor
        from surya.detection import DetectionPredictor
        import pypdfium2 as pdfium
    except ImportError:
        print("Surya 未安装，跳过")
        return None

    try:
        pdf = pdfium.PdfDocument(pdf_path)
        images = [page.render(scale=2.0).to_pil() for page in pdf]
        pdf.close()

        det = DetectionPredictor()
        rec = RecognitionPredictor()
        layout = LayoutPredictor()

        start = time.time()
        ocr_results = rec(images, det, langs=["en"])
        layout_results = layout(images)
        elapsed = time.time() - start

        # 按版面区域重建文本
        parts = []
        for ocr, lay in zip(ocr_results, layout_results):
            for region in lay.bboxes:
                # 找该区域内的文本行
                rx0, ry0, rx1, ry1 = region.bbox
                region_lines = []
                for line in ocr.text_lines:
                    lx0, ly0, lx1, ly1 = line.bbox
                    cx, cy = (lx0 + lx1) / 2, (ly0 + ly1) / 2
                    if rx0 <= cx <= rx1 and ry0 <= cy <= ry1:
                        region_lines.append(line)
                region_lines.sort(key=lambda l: l.bbox[1])
                parts.append(" ".join(l.text for l in region_lines))
        return "\n".join(parts), elapsed
    except Exception as e:
        print(f"Surya 调用失败: {e}")
        return None


def simple_reading_order(pdf_path: str) -> tuple[str, float]:
    """基线方法：PyMuPDF 默认 get_text()。"""
    import pymupdf
    doc = pymupdf.open(pdf_path)
    start = time.time()
    parts = [page.get_text() for page in doc]
    elapsed = time.time() - start
    doc.close()
    return "\n".join(parts), elapsed


def main():
    if len(sys.argv) < 2:
        print("用法: python3 -m scripts.compare_reading_order <pdf_path>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    print(f"对比文件: {pdf_path}")
    print()

    # 基线
    print("=== 方法 1: PyMuPDF 默认 get_text() ===")
    text1, t1 = simple_reading_order(pdf_path)
    print(f"  耗时: {t1:.3f}s")
    print(f"  文本前 300 字: {text1[:300]}")
    print()

    # 自研
    print("=== 方法 2: 自研阅读顺序重建 ===")
    text2, t2 = method_self_pymupdf(pdf_path)
    print(f"  耗时: {t2:.3f}s")
    print(f"  文本前 300 字: {text2[:300]}")
    print()

    # Surya
    print("=== 方法 3: Surya LayoutPredictor ===")
    result = method_surya(pdf_path)
    if result:
        text3, t3 = result
        print(f"  耗时: {t3:.3f}s")
        print(f"  文本前 300 字: {text3[:300]}")
        print()
        print(f"=== 对比总结 ===")
        print(f"  PyMuPDF 默认: {t1:.3f}s")
        print(f"  自研重建: {t2:.3f}s ({t1/t2:.1f}x)")
        print(f"  Surya: {t3:.3f}s ({t1/t3:.1f}x)")
    else:
        print("  (Surya 不可用)")


if __name__ == "__main__":
    main()