"""从 arXiv 批量下载 PDF。

用法：
    python3 -m scripts.download_arxiv --categories cs.CL,cs.LG --num 200
    python3 -m scripts.download_arxiv --query "retrieval augmented generation" --num 100
    python3 -m scripts.download_arxiv --categories cs.CV --num 300 --out-dir data/papers_large
"""
from __future__ import annotations

import argparse
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ARXIV_API = "http://export.arxiv.org/api/query"
ARXIV_PDF = "https://arxiv.org/pdf"
_USER_AGENT = "sciflow/0.1 (research; contact: your@email.com)"


def fetch_arxiv_ids(
    categories: list[str] | None = None,
    query: str | None = None,
    max_results: int = 200,
) -> list[str]:
    """从 arXiv API 获取论文 ID 列表。

    Args:
        categories: arXiv 分类，如 ["cs.CL", "cs.LG"]
        query: 关键词查询
        max_results: 最大获取数

    Returns:
        arXiv ID 列表，如 ["2305.14314v1", ...]
    """
    # 构造搜索条件
    parts = []
    if categories:
        cat_query = " OR ".join(f"cat:{c}" for c in categories)
        parts.append(f"({cat_query})")
    if query:
        parts.append(f'all:"{query}"')

    search_query = " AND ".join(parts) if parts else "cat:cs.CL"

    ids = []
    batch_size = 100  # arXiv 单次最多返回 100
    start = 0

    while len(ids) < max_results:
        params = urllib.parse.urlencode({
            "search_query": search_query,
            "start": start,
            "max_results": min(batch_size, max_results - len(ids)),
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        })
        url = f"{ARXIV_API}?{params}"
        print(f"  [arXiv API] 拉取 start={start} ...")

        req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        with urllib.request.urlopen(req, timeout=60) as resp:
            xml_data = resp.read()

        root = ET.fromstring(xml_data)
        ns = {"atom": "http://www.w3.org/2005/Atom"}

        batch_ids = []
        for entry in root.findall("atom:entry", ns):
            id_url = entry.find("atom:id", ns).text
            # 从 URL 提取 ID，如 http://arxiv.org/abs/2305.14314v1
            arxiv_id = id_url.rsplit("/", 1)[-1]
            batch_ids.append(arxiv_id)

        if not batch_ids:
            break

        ids.extend(batch_ids)
        start += batch_size

        # arXiv 建议 3 秒间隔
        time.sleep(3)

    return ids[:max_results]


def download_pdf(arxiv_id: str, out_dir: Path) -> bool:
    """下载单篇 PDF。"""
    # 去掉版本号：2305.14314v1 → 2305.14314
    clean_id = arxiv_id.split("v")[0] if "v" in arxiv_id else arxiv_id
    pdf_url = f"{ARXIV_PDF}/{clean_id}.pdf"
    out_path = out_dir / f"{arxiv_id}.pdf"

    if out_path.exists():
        return True

    try:
        req = urllib.request.Request(pdf_url, headers={"User-Agent": _USER_AGENT})
        with urllib.request.urlopen(req, timeout=60) as resp:
            pdf_data = resp.read()
        out_path.write_bytes(pdf_data)
        return True
    except Exception as e:
        print(f"    ✗ 下载失败 {arxiv_id}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="从 arXiv 批量下载 PDF")
    parser.add_argument("--categories", default="cs.CL,cs.LG",
                        help="arXiv 分类，逗号分隔")
    parser.add_argument("--query", default=None, help="关键词查询")
    parser.add_argument("--num", type=int, default=200, help="下载数量")
    parser.add_argument("--out-dir", default="data/papers_large", help="输出目录")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 解析分类
    categories = [c.strip() for c in args.categories.split(",")] if args.categories else None

    print(f"=== 阶段1: 获取 arXiv ID ===")
    print(f"  分类: {categories}")
    print(f"  关键词: {args.query}")
    print(f"  目标数量: {args.num}")
    print()

    ids = fetch_arxiv_ids(categories, args.query, args.num)
    print(f"  ✓ 获取 {len(ids)} 个 arXiv ID")

    print()
    print(f"=== 阶段2: 下载 PDF ===")
    print(f"  输出目录: {out_dir}")
    print()

    success = 0
    for i, arxiv_id in enumerate(ids, 1):
        if download_pdf(arxiv_id, out_dir):
            success += 1
        if i % 10 == 0:
            print(f"  进度: {i}/{len(ids)}，成功 {success}")
        # 礼貌间隔，避免被限流
        time.sleep(1)

    print()
    print(f"=== 完成 ===")
    print(f"  成功下载 {success}/{len(ids)} 篇")
    print(f"  保存位置: {out_dir}")


if __name__ == "__main__":
    main()