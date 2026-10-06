"""RAG 总编排，含三处拒绝机制。
来源：复用 central-dogma 的 api/pipeline.py。
"""
from __future__ import annotations

from src.rag.schemas import AskResponse
from src.rag.context import assemble_context

MIN_RELEVANCE_SCORE = 0.3

def _looks_like_soft_refusal(answer: str) -> bool:
    """检测'软拒绝'——模型用自然语言表达'不知道'。"""
    if not answer:
        return True
    
    answer_lower = answer.lower().strip()
    
    # 常见的软拒绝模式
    refusal_patterns = [
        "cannot find information",
        "cannot find any information",
        "can't find information",
        "no information",
        "not directly address",
        "does not directly address",
        "sources are largely unrelated",
        "sources are unrelated",
        "not addressed in the",
        "the provided sources do not",
        "the sources do not contain",
        "i don't have information",
        "there is no information",
    ]
    
    # 前 200 字符里出现拒绝模式 + 答案很短 → 判为软拒绝
    prefix = answer_lower[:200]
    for pattern in refusal_patterns:
        if pattern in prefix:
            # 再检查：如果答案里没有明确的引用标记，很可能是软拒绝
            has_citations = bool(re.findall(r"\[[\w\.\-:]+:\w+\]", answer))
            if not has_citations:
                return True
    
    return False

def answer_question(question: str, retriever, generator) -> AskResponse:
    # 1. 检索
    hits = retriever.search(question, top_k=10)

    # 拒绝一：检索为空
    if not hits:
        return AskResponse(
            answer="I couldn't find anything in the corpus that supports an answer.",
            citations=[],
            status="refused",
        )

    # 拒绝二：分数太低
    best = hits[0]["score"]
    if best < MIN_RELEVANCE_SCORE:
        return AskResponse(
            answer="I couldn't find anything relevant enough to answer that question.",
            citations=[],
            status="refused",
        )

    # 上下文组装
    citations = assemble_context(hits)

    # 生成
    answer = generator.generate(question, citations)



    # 拒绝三：生成器无法接地
    if not answer.strip():
        return AskResponse(
            answer="The sources don't support a reliable answer.",
            citations=[],
            status="refused",
        )

    # # ★ 新增：软拒绝检测
    # if _looks_like_soft_refusal(answer):
    #     return _refused(store, "The sources don't support a reliable answer.")

    return AskResponse(
        answer=answer,
        citations=citations,
        status="answered",
        confidence=best,
    )