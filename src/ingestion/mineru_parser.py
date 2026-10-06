"""MinerU 解析器封装。
来源：自研，调用 opendatalab/MinerU 官方 API。
"""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class MinerUParser:
    def parse(self, pdf_path: str) -> dict:
        try:
            from mineru.cli.client import parse_pdf
        except ImportError:
            logger.warning("MinerU 未安装，回退到 Docling")
            from src.ingestion.docling_parser import DoclingParser
            return DoclingParser().parse(pdf_path)

        output_dir = Path("./data/parsed") / Path(pdf_path).stem
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            result = parse_pdf(
                pdf_path,
                output_dir=str(output_dir),
                lang="en",
                formula_enable=True,
                table_enable=True,
            )
            markdown = result.get("markdown", "") if isinstance(result, dict) else str(result)
        except Exception as e:
            logger.warning(f"MinerU 解析失败({e})，回退 Docling")
            from src.ingestion.docling_parser import DoclingParser
            return DoclingParser().parse(pdf_path)

        return {
            "parser": "mineru",
            "markdown": markdown,
            "pdf_path": pdf_path,
            "output_dir": str(output_dir),
        }