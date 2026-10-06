"""Surya OCR 解析器封装。
来源：自研，调用 VikParuchuri/surya。
适用场景：扫描件、图片型 PDF。
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class SuryaParser:
    def __init__(self, langs: list[str] | None = None):
        self.langs = langs or ["en"]
        self._det = None
        self._rec = None
        self._layout = None

    def _load_models(self):
        try:
            from surya.detection import DetectionPredictor
            from surya.recognition import RecognitionPredictor
            from surya.layout import LayoutPredictor
        except ImportError:
            raise ImportError("请安装 Surya: pip install surya-ocr")

        if self._det is None:
            self._det = DetectionPredictor()
        if self._rec is None:
            self._rec = RecognitionPredictor()
        if self._layout is None:
            self._layout = LayoutPredictor()

    def parse(self, pdf_path: str) -> dict:
        self._load_models()
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(pdf_path)
        images = []
        for page in pdf:
            bitmap = page.render(scale=2.0)
            images.append(bitmap.to_pil())
        pdf.close()

        ocr_results = self._rec(images, self._det, langs=self.langs)
        layout_results = self._layout(images)

        return {
            "parser": "surya",
            "pdf_path": pdf_path,
            "ocr_results": ocr_results,
            "layout_results": layout_results,
            "images": images,
        }