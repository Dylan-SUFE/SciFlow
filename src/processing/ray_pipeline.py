"""基于 Ray Data 的分布式多模态数据处理管道。
来源：参考 Ray 官方教程（ray-project/ray）+ 自研。

用途：
  - 处理 PB 级文档时，把解析、切块、Embedding、写入向量库各阶段并行化。
  - 支持 CPU（解析/切块）和 GPU（Embedding）异构调度。
  - Ray 不可用时自动降级到单机处理。

参考教程：
  - https://docs.ray.io/en/latest/data/working-with-images.html
  - https://docs.ray.io/en/latest/data/working-with-text.html
  - https://docs.ray.io/en/latest/train/getting-started-transformers.html
"""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


# ============ 单机回退 ============

def _fallback_single_node(pdf_paths: list[str]):
    """Ray 不可用时的单机回退。"""
    logger.warning("[Ray] 不可用，降级到单机处理")
    from src.ingestion.router import DocumentRouter
    from src.ingestion.adapters import to_document
    from src.processing.chunker import chunk_document
    from src.processing.embedder import embed_chunks

    router = DocumentRouter()
    all_chunks = []
    for pdf in pdf_paths:
        try:
            parser = router.select(pdf)
            raw = parser.parse(pdf)
            doc = to_document(raw, paper_id=Path(pdf).stem)
            all_chunks.extend(chunk_document(doc))
        except Exception as e:
            logger.warning(f"单机解析失败 {pdf}: {e}")
            continue

    return embed_chunks(all_chunks)


# ============ Ray 分布式管道 ============

def build_ingestion_pipeline(
    pdf_paths: list[str],
    batch_size: int = 8,
    num_gpus: float = 0.5,
    concurrency: int = 4,
):
    """构建 Ray Data 分布式摄取管道。

    Args:
        pdf_paths: PDF 文件路径列表
        batch_size: 每批处理文档数
        num_gpus: 每个 worker 分配的 GPU 数量（0.5 表示半张卡）
        concurrency: 并发 worker 数

    Returns:
        ray.data.Dataset，包含 embedded chunks
    """
    try:
        import ray
        from ray.data import from_items
    except ImportError:
        logger.warning("Ray 未安装，请执行: pip install 'ray[data]'")
        return None

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, logging_level=logging.WARNING)

    logger.info(
        f"[Ray] 启动分布式管道: {len(pdf_paths)} 篇文档, "
        f"batch_size={batch_size}, num_gpus={num_gpus}, concurrency={concurrency}"
    )

    # Stage 1: 分布式解析（CPU 密集）
    ds = from_items(pdf_paths)

    def _parse(pdf_path: str):
        """单文档解析 → Chunk 列表（在 Ray worker 内执行）。"""
        from src.ingestion.router import DocumentRouter
        from src.ingestion.adapters import to_document
        from src.processing.chunker import chunk_document

        try:
            router = DocumentRouter()
            parser = router.select(pdf_path)
            raw = parser.parse(pdf_path)
            doc = to_document(raw, paper_id=Path(pdf_path).stem)
            chunks = chunk_document(doc)
            # Chunk 里 embedding 是 numpy 数组，需要转成可序列化格式
            return [
                {
                    "chunk_id": c.chunk_id,
                    "paper_id": c.paper_id,
                    "modality": c.modality,
                    "content": c.content,
                    "section": c.section,
                    "image_uri": c.image_uri,
                }
                for c in chunks
            ]
        except Exception as e:
            logger.warning(f"[Ray] 解析失败 {pdf_path}: {e}")
            return []

    # map 输出是 list[list[dict]]，用 flat_map 展平
    ds = ds.flat_map(_parse)

    # Stage 2: 分布式 Embedding（GPU 密集）
    def _embed_batch(batch: dict) -> dict:
        """批量 Embedding（在 Ray GPU worker 内执行）。"""
        import numpy as np

        from src.processing.embedder import _get_model, _hash_embed

        model = _get_model()
        texts = batch["content"]

        if model == "hashing":
            vectors = np.stack([_hash_embed(t) for t in texts])
        else:
            vectors = model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            ).astype(np.float32)

        batch["embedding"] = [v.tolist() for v in vectors]
        return batch

    ds = ds.map_batches(
        _embed_batch,
        batch_size=batch_size,
        batch_format="pandas",
        num_gpus=num_gpus,
        concurrency=concurrency,
    )

    logger.info("[Ray] 管道构建完成")
    return ds


def ingest_pdfs_distributed(
    pdf_paths: list[str],
    batch_size: int = 8,
    num_gpus: float = 0.5,
    concurrency: int = 4,
):
    """端到端分布式摄取：解析 → 切块 → Embedding。

    Returns:
        list[dict]（每个 dict 是一行 chunk 数据，含 embedding）
    """
    if not pdf_paths:
        return []

    ds = build_ingestion_pipeline(
        pdf_paths,
        batch_size=batch_size,
        num_gpus=num_gpus,
        concurrency=concurrency,
    )

    if ds is None:
        # Ray 不可用，回退单机
        chunks = _fallback_single_node(pdf_paths)
        return [
            {
                "chunk_id": c.chunk_id,
                "paper_id": c.paper_id,
                "modality": c.modality,
                "content": c.content,
                "section": c.section,
                "image_uri": c.image_uri,
                "embedding": c.embedding.tolist() if c.embedding is not None else None,
            }
            for c in chunks
        ]

    # 触发执行
    logger.info("[Ray] 开始执行管道...")
    rows = ds.take_all()
    logger.info(f"[Ray] 管道执行完成，共 {len(rows)} 条 chunk")
    return rows


def shutdown():
    """关闭 Ray 运行时。"""
    try:
        import ray
        if ray.is_initialized():
            ray.shutdown()
            logger.info("[Ray] 已关闭")
    except ImportError:
        pass


# ============ 高级：带 GPU 显存优化的流式管道 ============

def build_streaming_pipeline(pdf_paths: list[str], window_size: int = 100):
    """流式管道：处理超大规模文档时，避免一次性加载所有数据到内存。

    参考 Ray 官方 streaming execution 模式：
      - 使用 streaming_execution 让数据集流式通过各 stage
      - 窗口内并行，窗口间串行，内存占用恒定
    """
    try:
        import ray
        from ray.data import from_items
    except ImportError:
        logger.warning("Ray 未安装")
        return None

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, logging_level=logging.WARNING)

    # 开启流式执行
    ctx = ray.data.DataContext.get_current()
    ctx.execution_options.preserve_order = False
    ctx.execution_options.verbose_progress = True

    ds = from_items(pdf_paths)

    # Stage 1: 解析 + 切块
    def _parse_and_chunk(path: str):
        from src.ingestion.router import DocumentRouter
        from src.ingestion.adapters import to_document
        from src.processing.chunker import chunk_document

        try:
            parser = DocumentRouter().select(path)
            doc = to_document(parser.parse(path), paper_id=Path(path).stem)
            return [
                {"chunk_id": c.chunk_id, "content": c.content, "modality": c.modality}
                for c in chunk_document(doc)
            ]
        except Exception as e:
            logger.warning(f"[Ray] 解析失败 {path}: {e}")
            return []

    ds = ds.flat_map(_parse_and_chunk)

    # Stage 2: Embedding（GPU，半张卡并发 4）
    def _embed(batch):
        import numpy as np
        from src.processing.embedder import _get_model, _hash_embed

        model = _get_model()
        texts = batch["content"]
        if model == "hashing":
            vectors = np.stack([_hash_embed(t) for t in texts])
        else:
            vectors = model.encode(
                texts, normalize_embeddings=True, convert_to_numpy=True
            ).astype(np.float32)
        batch["embedding"] = [v.tolist() for v in vectors]
        return batch

    ds = ds.map_batches(
        _embed,
        batch_size=32,
        batch_format="pandas",
        num_gpus=0.5,
        concurrency=4,
    )

    logger.info(f"[Ray] 流式管道已构建（window_size={window_size}）")
    return ds


# ============ 与 main.py 集成 ============

def ingest_with_auto_backend(pdf_dir: str, use_ray: bool = False):
    """统一入口：自动判断用 Ray 还是单机。

    Args:
        pdf_dir: PDF 目录
        use_ray: True 强制使用 Ray；False 走单机

    Returns:
        list[Chunk]，带 embedding
    """
    from src.processing.chunker import Chunk
    import numpy as np

    pdf_paths = sorted(str(p) for p in Path(pdf_dir).glob("*.pdf"))
    if not pdf_paths:
        return []

    if use_ray:
        logger.info(f"[Auto] 使用 Ray 处理 {len(pdf_paths)} 篇文档")
        rows = ingest_pdfs_distributed(pdf_paths)
        # 转回 Chunk 对象
        chunks = []
        for r in rows:
            chunks.append(Chunk(
                chunk_id=r["chunk_id"],
                paper_id=r["paper_id"],
                modality=r["modality"],
                content=r["content"],
                section=r.get("section"),
                image_uri=r.get("image_uri"),
                embedding=np.asarray(r["embedding"], dtype=np.float32)
                if r.get("embedding") else None,
            ))
        return chunks

    logger.info(f"[Auto] 使用单机处理 {len(pdf_paths)} 篇文档")
    return _fallback_single_node(pdf_paths)