"""
SciFlow: 多模态科学文献数据管道与 RAG 系统
统一入口：
  python main.py --run-all
  python main.py --query "What is the main finding?"
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sciflow")


def stage_ingest(pdf_dir: str):
    """阶段1：解析 PDF → MultimodalDocument（带缓存）。"""
    import pickle
    from pathlib import Path

    cache_file = Path("data/ingest_cache.pkl")

    # ★ 如果缓存存在且文件数匹配，直接读缓存
    pdf_paths = sorted(Path(pdf_dir).glob("*.pdf"))
    if cache_file.exists():
        try:
            with cache_file.open("rb") as f:
                cached = pickle.load(f)
            if cached.get("pdf_count") == len(pdf_paths):
                logger.info(f"从缓存读取 {len(cached['documents'])} 篇文档")
                return cached["documents"]
        except Exception:
            pass

    # 正常解析流程
    from src.ingestion.router import DocumentRouter
    from src.ingestion.adapters import to_document

    router = DocumentRouter()
    logger.info(f"找到 {len(pdf_paths)} 篇 PDF")
    documents = []
    for pdf_path in pdf_paths:
        try:
            parser = router.select(str(pdf_path))
            raw = parser.parse(str(pdf_path))
            doc = to_document(raw, paper_id=pdf_path.stem)
            documents.append(doc)
            logger.info(
                f"  ✓ {pdf_path.name}: "
                f"{len(doc.text_chunks)} 文本块, "
                f"{len(doc.figures)} 图表, "
                f"{len(doc.tables)} 表格, "
                f"{len(doc.formulas)} 公式"
            )
            # logger.info(f"  ✓ {pdf_path.name}: {len(doc.text_chunks)} 文本块")
        except Exception as e:
            logger.warning(f"  ✗ {pdf_path.name} 解析失败: {e}")
            continue

    # ★ 写入缓存
    try:
        with cache_file.open("wb") as f:
            pickle.dump({
                "pdf_count": len(pdf_paths),
                "documents": documents,
            }, f)
        logger.info(f"已缓存到 {cache_file}")
    except Exception as e:
        logger.warning(f"缓存写入失败: {e}")

    return documents


def stage_process(documents):
    """阶段2：切块 + 向量化"""
    from src.processing.chunker import chunk_document
    from src.processing.embedder import embed_chunks

    all_chunks = []
    for doc in documents:
        chunks = chunk_document(doc)
        all_chunks.extend(chunks)
    logger.info(f"切块完成：共 {len(all_chunks)} 个 chunk")

    all_chunks = embed_chunks(all_chunks)
    logger.info(f"向量化完成：{len(all_chunks)} 条")
    return all_chunks


def stage_index(chunks):
    """阶段3：写入 Milvus + BM25"""
    from src.indexing.milvus_store import MilvusStore
    from src.indexing.bm25_store import BM25Store

    milvus = MilvusStore()
    milvus.upsert(chunks)
    logger.info(f"Milvus 写入 {len(chunks)} 条")

    bm25 = BM25Store()
    bm25.build(chunks)
    logger.info(f"BM25 索引 {len(chunks)} 条")

    return milvus, bm25


def stage_query(milvus, bm25, question: str):
    """阶段4：RAG 查询"""
    from src.indexing.hybrid_retriever import HybridRetriever
    from src.rag.pipeline import answer_question
    from src.rag.generation import build_generator

    retriever = HybridRetriever(milvus, bm25)
    # generator = build_generator("extractive")
    generator = build_generator("deepseek")

    response = answer_question(question, retriever, generator)
    print(f"\n{'=' * 60}")
    print(f"问题: {question}")
    print(f"答案: {response.answer}")
    print(f"状态: {response.status}")
    print(f"引用数: {len(response.citations)}")
    for c in response.citations[:3]:
        print(f"  - {c.source_id} (score={c.score:.4f})")
    print(f"{'=' * 60}\n")
    return response


def stage_evaluate(milvus, bm25):
    """阶段5：评估"""
    from src.evaluation.harness import evaluate
    from src.evaluation.datasets.demo_cases import get_demo_cases
    from src.rag.generation import build_generator
    from src.indexing.hybrid_retriever import HybridRetriever

    retriever = HybridRetriever(milvus, bm25)
    generator = build_generator("deepseek")
    cases = get_demo_cases()

    report = evaluate(retriever, generator, cases)
    print(report.to_markdown())

def stage_process(documents=None, pdf_dir: str = "./data/papers", use_ray: bool = False):
    """阶段2：切块 + 向量化（支持 Ray 分布式）。"""
    if use_ray:
        from src.processing.ray_pipeline import ingest_with_auto_backend
        chunks = ingest_with_auto_backend(pdf_dir, use_ray=True)
        logger.info(f"Ray 分布式处理完成：{len(chunks)} 个 chunk")
        return chunks

    # 单机路径（原逻辑）
    from src.processing.chunker import chunk_document
    from src.processing.embedder import embed_chunks

    all_chunks = []
    for doc in documents:
        all_chunks.extend(chunk_document(doc))
    logger.info(f"切块完成：共 {len(all_chunks)} 个 chunk")
    return embed_chunks(all_chunks)

def main():
    parser = argparse.ArgumentParser(description="SciFlow 多模态 RAG 管道")
    parser.add_argument("--pdf-dir", default="./data/papers", help="PDF 目录")
    parser.add_argument("--run-all", action="store_true", help="跑完整流程")
    parser.add_argument("--query", default=None, help="只跑查询")
    parser.add_argument("--reset", action="store_true", help="重置向量库")
    parser.add_argument(
    "--use-ray",
    action="store_true",
    default=False,          # 显式写出，等价于默认行为
    help="启用 Ray 分布式处理（默认关闭）",
)
    args = parser.parse_args()

    if args.run_all:
        logger.info("═══ 阶段1: 文档摄取 ═══")
        documents = stage_ingest(args.pdf_dir)

        logger.info("═══ 阶段2: 切块与向量化 ═══")
        chunks = stage_process(documents)

        logger.info("═══ 阶段3: 索引 ═══")
        if args.reset:
            from src.indexing.milvus_store import MilvusStore
            MilvusStore().reset()
        milvus, bm25 = stage_index(chunks)

        logger.info("═══ 阶段4: RAG 查询演示 ═══")
        stage_query(milvus, bm25, "What is the main finding of this paper?")

        logger.info("═══ 阶段5: 评估 ═══")
        stage_evaluate(milvus, bm25)

    elif args.query:
        from src.indexing.milvus_store import MilvusStore
        from src.indexing.bm25_store import BM25Store
        milvus = MilvusStore()
        bm25 = BM25Store()
        bm25.load()
        stage_query(milvus, bm25, args.query)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()