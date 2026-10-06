"""根据文档特征选择最佳解析器。
来源：自研。
"""
from __future__ import annotations

import logging
from enum import Enum

logger = logging.getLogger(__name__)


class ParserType(Enum):
    MINERU = "mineru"
    DOCLING = "docling"
    SURYA = "surya"


class DocumentRouter:
    def __init__(self):
        self._mineru = None
        self._docling = None
        self._surya = None

    def select(self, pdf_path: str):
        if self._is_scanned(pdf_path):
            logger.info(f"[Router] 扫描件 → Surya: {pdf_path}")
            return self._get_surya()

        density = self._sample_formula_density(pdf_path)
        if density > 0.05:
            logger.info(f"[Router] 公式密集({density:.2%}) → MinerU: {pdf_path}")
            return self._get_mineru()

        logger.info(f"[Router] 通用文档 → Docling: {pdf_path}")
        return self._get_docling()

    def _is_scanned(self, pdf_path: str, sample_pages: int = 3) -> bool:
        try:
            import pymupdf
            doc = pymupdf.open(pdf_path)
            sample = min(sample_pages, len(doc))
            total = sum(len(doc[i].get_text().strip()) for i in range(sample))
            doc.close()
            return (total / max(sample, 1)) < 50
        except Exception:
            return False

    def _sample_formula_density(self, pdf_path: str, pages: int = 3) -> float:
        try:
            import pymupdf
            doc = pymupdf.open(pdf_path)
            sample = min(pages, len(doc))
            total_chars, formula_chars = 0, 0
            for i in range(sample):
                text = doc[i].get_text()
                total_chars += len(text)
                for line in text.split("\n"):
                    if any(s in line for s in ["\\frac", "\\sum", "_{", "^{", "\\int"]):
                        formula_chars += len(line)
            doc.close()
            return formula_chars / max(total_chars, 1)
        except Exception:
            return 0.0

    def _get_mineru(self):
        if self._mineru is None:
            from src.ingestion.mineru_parser import MinerUParser
            self._mineru = MinerUParser()
        return self._mineru

    def _get_docling(self):
        if self._docling is None:
            from src.ingestion.docling_parser import DoclingParser
            self._docling = DoclingParser()
        return self._docling

    def _get_surya(self):
        if self._surya is None:
            from src.ingestion.surya_parser import SuryaParser
            self._surya = SuryaParser(langs=["en"])
        return self._surya