"""用 LLM 从真实语料库构建分模态评估集。

工作流程：
1. 从 BM25 索引里分层采样 N 个 chunk（文本/图表/表格/公式）
2. 根据 chunk 模态选择对应的 GENERATE_PROMPT 生成问题
3. 用 VERIFY_PROMPT 二次校验（宽松版）
4. 生成 M 个"语料库无关"的问题测试拒绝机制
5. 输出 JSONL 供 run_eval.py 使用

支持 OpenAI 兼容 API（OpenAI、DeepSeek、通义千问、Kimi 等）。

用法：
    export LLM_API_KEY=sk-xxx
    export LLM_BASE_URL=https://api.deepseek.com/v1
    export LLM_MODEL=deepseek-chat
    python3 -m scripts.llm_eval_builder --sample 100 --unanswerable 10
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import time
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    raise SystemExit("请先安装 openai: pip3 install openai")


# ============================================================
# 配置
# ============================================================

BM25_PATH = Path("data/bm25.pkl")
OUTPUT_DIR = Path("evaluation/datasets")

API_KEY = os.getenv("LLM_API_KEY", "")
BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")


# ============================================================
# 分模态 Prompt
# ============================================================

GENERATE_PROMPTS = {
    # ---- 文本 chunk：自然语言段落 ----
    "text": """You are helping build an evaluation set for a scientific RAG system.

Below is a passage from a scientific paper:

---
{chunk}
---

Task: Generate ONE natural question in English that:
1. Can ONLY be answered using this passage (not from general knowledge)
2. Sounds like what a real scientist would ask (NOT "what does this passage say")
3. Is specific and self-contained — the question must make sense without seeing the passage
4. Uses terms and concepts that appear in the passage
5. Length: 10-25 words
6. If the passage is a formula, definition, or theorem:
   - Ask about its meaning, conditions, or implications
7. If the passage describes a method or experiment:
   - Ask about the mechanism, parameters, or expected outcome
8. If the passage is a result or conclusion:
   - Ask about the finding, evidence, or its implications

Output ONLY the question. No quotes, no prefix, no explanation.""",

    # ---- 图表 chunk：caption ----
    "figure": """You are helping build an evaluation set for a scientific RAG system.

Below is a figure caption from a scientific paper:

---
{chunk}
---

Task: Generate ONE question that a scientist might ask about this figure.
The question should:
1. Ask about what the figure SHOWS (trends, comparisons, values, patterns)
2. Be answerable from the figure's caption
3. Sound natural, not like "what does this figure show"
4. Length: 8-20 words

Examples (to guide your style):
- If caption says "Dose-response curve showing plateau at high doses"
  → "What happens to the response at high doses?"
- If caption says "Comparison of BERT and GPT-3 accuracy on MMLU"
  → "How does GPT-3's accuracy compare to BERT's on MMLU?"
- If caption says "The red dots represent maxima of the quadratic form"
  → "What do the red dots represent in this figure?"

Output ONLY the question. No quotes, no prefix, no explanation.""",

    # ---- 表格 chunk：行级数据 ----
    "table": """You are helping build an evaluation set for a scientific RAG system.

Below is a row from a table in a scientific paper (formatted as key=value pairs):

---
{chunk}
---

Task: Generate ONE question that can be answered by this table row.
The question should:
1. Ask about a specific value, comparison, or attribute
2. Be answerable from this row alone
3. Sound natural
4. Length: 6-15 words

Examples (to guide your style):
- If row says "Method=BERT, Accuracy=92.3, F1=91.5"
  → "What is BERT's accuracy and F1 score?"
- If row says "Temperature=37°C, Yield=85%, Time=2h"
  → "What yield is achieved at 37°C?"
- If row says "Model=ResNet-50, Top-1=76.1, Params=25.6M"
  → "How many parameters does ResNet-50 have?"

Output ONLY the question. No quotes, no prefix, no explanation.""",

    # ---- 公式 chunk：LaTeX ----
    "formula": """You are helping build an evaluation set for a scientific RAG system.

Below is a mathematical formula from a scientific paper:

---
{chunk}
---

Task: Generate ONE question that this formula can help answer.
The question should:
1. Ask about what the formula DESCRIBES, COMPUTES, or MODELS
2. Be answerable by understanding the formula's structure
3. Sound natural to a scientist
4. Length: 6-15 words

Examples (to guide your style):
- If formula is "dN/dt = rN(1 - N/K)"
  → "What equation describes logistic population growth?"
- If formula is "E = mc²"
  → "What equation relates mass and energy?"
- If formula is "cos(θ) = (a · b) / (|a| |b|)"
  → "How is the cosine of the angle between two vectors computed?"

Output ONLY the question. No quotes, no prefix, no explanation.""",
}


VERIFY_PROMPT = """You are evaluating whether a question can be answered using a passage.

Question: {question}

Passage:
---
{chunk}
---

Task: Judge whether the passage contains enough information to answer the question.

Guidelines:
- YES: The passage directly contains the answer, OR partial information is 
  sufficient to give a reasonable answer.
- NO: The passage lacks key information, OR the question is about a completely 
  different topic.

Note: The passage can be text, a figure caption, a table row, or a formula. 
Judge based on what it provides — partial answers count as YES.

Examples:
- Question: "What is the melting point of iron?"
  Passage: "Iron melts at 1538°C under standard pressure."
  → YES (directly stated)

- Question: "What is the melting point of iron?"
  Passage: "Iron is a transition metal with symbol Fe."
  → NO (passage doesn't mention melting point)

- Question: "How does the algorithm handle edge cases?"
  Passage: "We propose Algorithm 1 which iteratively refines the solution 
            for each region, including edge cases."
  → YES (partial but relevant)

Output ONLY "YES" or "NO"."""


UNANSWERABLE_PROMPT = """You are helping build a "refusal test set" for a scientific RAG system.

The system's corpus covers these topics:
{topics}

Questions you have ALREADY generated (do NOT generate anything similar to these):
{previous_questions}

Task: Generate ONE scientific question that:
1. Sounds like a legitimate scientific question a researcher might ask
2. Is from a DIFFERENT scientific domain than the corpus topics above
3. Is from a DIFFERENT scientific domain than ALL previously generated questions
4. Has absolutely NO answer in the corpus — the system must refuse
5. Length: 8-20 words

Available domains to pick from (choose a DIFFERENT one each time):
- Medicine / clinical research
- Neuroscience / brain
- Microbiology / immunology
- Genomics / molecular biology
- Chemistry / materials science
- Physics / quantum mechanics
- Astronomy / astrophysics
- Geology / earth science
- Ecology / environmental science
- Psychology / cognitive science
- Economics / social science
- Linguistics / language
- Agriculture / food science
- Engineering / robotics

Output ONLY the question. No quotes, no prefix, no explanation."""


# ============================================================
# LLM 调用
# ============================================================

def make_client() -> OpenAI:
    if not API_KEY:
        raise SystemExit(
            "缺少 LLM_API_KEY 环境变量。\n"
            "示例：export LLM_API_KEY=sk-xxx\n"
            "如需自定义 API：export LLM_BASE_URL=https://api.deepseek.com/v1"
        )
    return OpenAI(api_key=API_KEY, base_url=BASE_URL)


def call_llm(client: OpenAI, prompt: str, max_retries: int = 3) -> str:
    """调 LLM，带重试和退避。"""
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=200,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt
            print(f"  LLM 调用失败({e})，{wait}秒后重试...")
            time.sleep(wait)
    return ""


# ============================================================
# 数据加载与模态判断
# ============================================================

def load_all_chunks():
    if not BM25_PATH.exists():
        raise FileNotFoundError(
            f"找不到 {BM25_PATH}。请先运行 python3 main.py --run-all 建立索引。"
        )
    with BM25_PATH.open("rb") as f:
        chunks = pickle.load(f)
    print(f"从 {BM25_PATH} 读取到 {len(chunks)} 个 chunk")
    return chunks


def _infer_modality(chunk_id: str) -> str:
    """从 chunk_id 判断模态（兼容双重/三重拼接）。

    支持两种格式：
      - 单次拼接：{paper_id}:c0 / {paper_id}:f0 / {paper_id}:fig_p2_i0 / {paper_id}:tbl_p5_t2
      - 双重拼接：{paper_id}:{paper_id}:fig_p2_i0 / {paper_id}:{paper_id}:tbl_p5_t2:r0

    关键策略：
    1. 用 rsplit(":", 1) 从最右边取后缀
    2. 长前缀优先（fig > f）
    3. 新增 r 前缀（表格行）
    4. 兜底检查完整 id 的关键词
    """
    # 从最右边取后缀
    suffix = chunk_id.rsplit(":", 1)[-1]

    # 长前缀优先判断
    if suffix.startswith("fig"):
        return "figure"
    if suffix.startswith("tbl"):
        return "table"
    if suffix.startswith("r") and len(suffix) > 1 and suffix[1:].isdigit():
        return "table"       # 表格行格式：r0, r1, r2
    if suffix.startswith("f") and len(suffix) > 1 and suffix[1:].isdigit():
        return "formula"
    if suffix.startswith("c") and len(suffix) > 1 and suffix[1:].isdigit():
        return "text"

    # 兜底 1：从完整 id 里找关键词
    cid_lower = chunk_id.lower()
    if "fig" in cid_lower:
        return "figure"
    if "tbl" in cid_lower or "table" in cid_lower:
        return "table"

    # 兜底 2：默认当文本
    return "text"


# ============================================================
# 分层采样
# ============================================================

def _sample_per_paper(pool: list, k: int, per_paper_cap: int = 2) -> list:
    """从 pool 里采样，每篇论文最多采 per_paper_cap 个。"""
    by_paper: dict[str, list] = {}
    for cid, content in pool:
        # 用第一个冒号前的部分作为 paper_id
        paper_id = cid.split(":")[0]
        by_paper.setdefault(paper_id, []).append((cid, content))

    sampled = []
    for paper_id, items in by_paper.items():
        n = min(per_paper_cap, len(items))
        sampled.extend(random.sample(items, n))
        if len(sampled) >= k:
            break

    return sampled[:k]


def sample_chunks_stratified(chunks, n: int):
    """分层采样：文本 60%，图表 15%，表格 15%，公式 10%。

    按模态设置不同的最小长度阈值：
      - 文本段落：200 字符（过滤碎片）
      - 图表 caption：10 字符（caption 通常短）
      - 表格行：10 字符
      - 公式：5 字符
    """
    # 不同模态的最小长度
    MIN_LENGTH = {
        "text": 200,
        "figure": 10,
        "table": 10,
        "formula": 5,
    }

    # 1. 按模态分组
    by_modality: dict[str, list] = {
        "text": [], "figure": [], "table": [], "formula": [],
    }
    for cid, content in chunks:
        modality = _infer_modality(cid)
        min_len = MIN_LENGTH.get(modality, 100)
        if len(content) >= min_len:
            by_modality[modality].append((cid, content))

    print()
    print("=== 各模态 chunk 数量 ===")
    for m, items in by_modality.items():
        print(f"  {m}: {len(items)} 个")

    # 2. 按比例分配配额
    quotas = {
        "text":    int(n * 0.60),
        "figure":  int(n * 0.15),
        "table":   int(n * 0.15),
        "formula": int(n * 0.10),
    }

    # 3. 每类内部分层采样
    sampled = []
    print()
    print(f"=== 分层采样 {n} 个 chunk ===")
    for modality, quota in quotas.items():
        pool = by_modality[modality]
        if not pool:
            print(f"  ⚠️  {modality} 无可用 chunk，跳过")
            continue
        k = min(quota, len(pool))
        sampled.extend(_sample_per_paper(pool, k))
        print(f"  采样 {modality}: {k} 个")

    random.shuffle(sampled)
    return sampled[:n]


# ============================================================
# 生成评估集
# ============================================================

def extract_topics(chunks, limit: int = 20) -> str:
    """从 chunk 里提取代表性内容片段，作为"语料库话题"提示。"""
    by_paper: dict[str, str] = {}
    for cid, content in chunks:
        paper_id = cid.split(":")[0]
        if paper_id not in by_paper and len(content) > 100:
            by_paper[paper_id] = content[:150]
    topics = list(by_paper.values())[:limit]
    return "\n".join(f"- {t}..." for t in topics)


def build_eval_set(sample_size: int, unanswerable_count: int, out_path: Path):
    client = make_client()
    chunks = load_all_chunks()

    sampled = sample_chunks_stratified(chunks, sample_size)

    print()
    print(f"开始生成 {len(sampled)} 条可答用例...\n")

    cases = []
    stats = {"text": 0, "figure": 0, "table": 0, "formula": 0, "failed": 0}

    for i, (cid, content) in enumerate(sampled, 1):
        modality = _infer_modality(cid)
        print(f"[{i}/{len(sampled)}] chunk={cid} (modality={modality})")

        generate_prompt = GENERATE_PROMPTS.get(modality, GENERATE_PROMPTS["text"])

        try:
            question = call_llm(client, generate_prompt.format(chunk=content[:1500]))
            question = question.strip().strip('"').strip("'")
        except Exception as e:
            print(f"  ✗ LLM 调用失败：{e}")
            stats["failed"] += 1
            continue

        if not question or len(question) < 10:
            print(f"  ✗ 问题太短，跳过")
            stats["failed"] += 1
            continue

        try:
            verify = call_llm(
                client,
                VERIFY_PROMPT.format(question=question, chunk=content[:1500]),
            ).upper()
        except Exception as e:
            print(f"  ✗ 校验调用失败：{e}")
            stats["failed"] += 1
            continue

        if "YES" not in verify:
            print(f"  ✗ 校验失败（{verify}），跳过")
            stats["failed"] += 1
            continue

        print(f"  ✓ {question[:80]}")

        cases.append({
            "question": question,
            "gold_source_ids": [cid],
            "answerable": True,
            "modality": modality,
            "notes": f"LLM 生成（{modality}）。原文：{content[:100]}...",
        })
        stats[modality] += 1

    # ========== Step 3: 生成不可答问题（逐条 + 去重）==========
    print()
    print(f"开始生成 {unanswerable_count} 条不可答用例...\n")
    topics = extract_topics(chunks, limit=20)

    generated_questions: list[str] = []

    def _is_duplicate(new_q: str, existing: list[str]) -> bool:
        """判断新问题是否和已有的重复（基于实词 Jaccard 相似度）。"""
        import re
        def tokens(s):
            return set(re.findall(r"[a-z]{4,}", s.lower()))
        new_tokens = tokens(new_q)
        for old_q in existing:
            old_tokens = tokens(old_q)
            if not new_tokens or not old_tokens:
                continue
            jaccard = len(new_tokens & old_tokens) / len(new_tokens | old_tokens)
            if jaccard > 0.5:
                return True
        return False

    i = 0
    max_attempts = unanswerable_count * 3

    while i < unanswerable_count and max_attempts > 0:
        max_attempts -= 1

        prev_text = (
            "\n".join(f"- {q}" for q in generated_questions)
            if generated_questions
            else "(none yet)"
        )

        try:
            question = call_llm(
                client,
                UNANSWERABLE_PROMPT.format(
                    topics=topics,
                    previous_questions=prev_text,
                ),
            ).strip().strip('"').strip("'")
        except Exception as e:
            print(f"  ✗ LLM 调用失败：{e}")
            continue

        if not question or len(question) < 10:
            continue

        if _is_duplicate(question, generated_questions):
            print(f"  ⚠️  重复，跳过：{question[:60]}...")
            continue

        i += 1
        generated_questions.append(question)
        print(f"[{i}/{unanswerable_count}] {question}")

        cases.append({
            "question": question,
            "gold_source_ids": [],
            "answerable": False,
            "modality": "none",
            "notes": "LLM 生成，语料库无关，应被拒绝",
        })

    # ========== 输出 ==========
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")

    answerable_count = sum(1 for c in cases if c["answerable"])
    print()
    print("=" * 60)
    print(f"已生成 {out_path}")
    print(f"  可答用例：{answerable_count} 条")
    print(f"    - 文本：  {stats['text']}")
    print(f"    - 图表：  {stats['figure']}")
    print(f"    - 表格：  {stats['table']}")
    print(f"    - 公式：  {stats['formula']}")
    print(f"  不可答用例：{len(cases) - answerable_count} 条")
    print(f"  失败跳过：  {stats['failed']} 条")
    success_rate = answerable_count / max(len(sampled), 1)
    print(f"  成功率：    {success_rate:.1%}")
    print()
    print(f"下一步：python3 -m scripts.run_eval --dataset {out_path}")

# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="用 LLM 自动构建评估集（分模态）")
    parser.add_argument("--sample", type=int, default=100, help="采样多少个 chunk")
    parser.add_argument("--unanswerable", type=int, default=10, help="生成多少个不可答问题")
    parser.add_argument(
        "--out",
        default="evaluation/datasets/llm_eval.jsonl",
        help="输出路径",
    )
    args = parser.parse_args()

    print("LLM 配置：")
    print(f"  Base URL: {BASE_URL}")
    print(f"  Model: {MODEL}")
    print(f"  采样数: {args.sample}")
    print(f"  不可答数: {args.unanswerable}")

    build_eval_set(args.sample, args.unanswerable, Path(args.out))


if __name__ == "__main__":
    main()