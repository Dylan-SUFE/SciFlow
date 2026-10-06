"""上下文组装。
来源：复用 central-dogma 的 api/context.py。
"""
from __future__ import annotations

from src.rag.schemas import Citation


def assemble_context(
    hits: list[dict],
    char_budget: int = 4000,
    per_source_cap: int = 2,
) -> list[Citation]:
    """去重 + 预算裁剪。"""
    chosen: list[Citation] = []
    seen_sources: dict[str, int] = {}
    used = 0

    for hit in hits:
        sid = hit["id"]
        if seen_sources.get(sid, 0) >= per_source_cap:
            continue
        snippet = hit.get("content", "")
        if chosen and used + len(snippet) > char_budget:
            break
        seen_sources[sid] = seen_sources.get(sid, 0) + 1
        used += len(snippet)
        chosen.append(Citation(
            source_id=sid,
            score=hit["score"],
            snippet=snippet[:500],
        ))

    return chosen