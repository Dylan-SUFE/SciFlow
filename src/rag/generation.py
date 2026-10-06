"""答案生成器。
来源：复用 central-dogma 的 api/generation.py。
"""
from __future__ import annotations

SYSTEM_PROMPT = (
    "You are a scientific literature assistant. Answer the question using ONLY the "
    "provided sources. After each claim, cite its source inline in brackets. "
    "If the sources do not support an answer, reply with exactly: NO_ANSWER."
)


# class ExtractiveGenerator:
#     """无模型拼接版（用于测试和 Demo）。"""

#     def generate(self, question: str, citations) -> str:
#         if not citations:
#             return ""
#         top = citations[0]
#         return (
#             f"Based on the retrieved source [{top.source_id}], "
#             f"here is the relevant content: {top.snippet[:200]}..."
#         )

class ExtractiveGenerator:
    """无模型拼接版，用于测试和 Demo。

    严格模式：
    - 只有检索结果中有"强相关"的证据才生成答案
    - 强相关的判定：query 和 chunk 有足够的关键词重叠
    - 否则返回空字符串，触发 pipeline 的拒绝三
    """

    def __init__(self, min_overlap: int = 2):
        self.min_overlap = min_overlap

    def generate(self, question: str, citations) -> str:
        if not citations:
            return ""

        # 只取最高分的 citation
        top = citations[0]
        snippet = top.snippet or ""

        # 计算 query 和 chunk 的关键词重叠
        q_terms = set(self._tokenize(question))
        c_terms = set(self._tokenize(snippet))
        overlap = len(q_terms & c_terms)

        # 弱相关 → 拒绝回答
        if overlap < self.min_overlap:
            return ""

        return (
            f"Based on the retrieved source [{top.source_id}], "
            f"here is the relevant content: {snippet[:200]}..."
        )

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """简单分词：小写 + 按空格切分 + 过滤短词。"""
        import re
        tokens = re.findall(r"[a-zA-Z]{3,}", text.lower())
        # 过滤常见停用词
        stopwords = {
            "the", "and", "for", "are", "was", "with", "this", "that",
            "from", "have", "has", "been", "which", "what", "how",
            "does", "can", "will", "not", "but", "all", "any", "one",
        }
        return [t for t in tokens if t not in stopwords]


class QwenVisionGenerator:
    """本地 Qwen2.5-VL 生成器（需要 GPU）。"""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_name)

    def generate(self, question: str, citations) -> str:
        sources = "\n\n".join(f"[{c.source_id}]: {c.snippet}" for c in citations)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Sources:\n{sources}\n\nQuestion: {question}"},
        ]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[prompt], return_tensors="pt").to(self.model.device)
        generated = self.model.generate(**inputs, max_new_tokens=512)
        trimmed = generated[:, inputs.input_ids.shape[1]:]
        answer = self.processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
        return "" if answer.upper().startswith("NO_ANSWER") else answer

class DeepSeekGenerator:
    """通过 DeepSeek API 生成答案（OpenAI 兼容）。

    需要环境变量：
      - LLM_API_KEY    DeepSeek 的 API key
      - LLM_BASE_URL   默认 https://api.deepseek.com/v1
      - LLM_MODEL      默认 deepseek-chat
    """

    def __init__(self, model_name: str | None = None):
        import os
        from openai import OpenAI

        # 从 .env 读取
        try:
            from pathlib import Path
            env_file = Path(".env")
            if env_file.exists():
                for line in env_file.read_text().splitlines():
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        os.environ.setdefault(k.strip(), v.strip())
        except Exception:
            pass

        api_key = os.getenv("LLM_API_KEY")
        if not api_key:
            raise ValueError("缺少 LLM_API_KEY 环境变量或 .env 文件")

        self._client = OpenAI(
            api_key=api_key,
            base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1"),
        )
        self._model = model_name or os.getenv("LLM_MODEL", "deepseek-chat")

    def generate(self, question: str, citations) -> str:
        if not citations:
            return ""

        # 把 citations 拼成上下文
        context_parts = []
        for c in citations:
            context_parts.append(
                f"[{c.source_id}] (score={c.score:.3f}):\n{c.snippet}"
            )
        context = "\n\n".join(context_parts)

        system_prompt = (
            "You are a scientific literature assistant.\n\n"
            "Your task: answer the question using ONLY the provided sources.\n\n"
            "Guidelines:\n"
            "1. For EVERY claim, cite the source inline using its exact ID...\n"
            "2. If the sources contain PARTIAL information, provide what you can "
            "and note what's missing — do NOT refuse just because incomplete.\n"
            "3. If the source is a formula, definition, or theorem, explain it.\n"
            "4. ONLY reply 'NO_ANSWER' if the sources are COMPLETELY unrelated — "
            "meaning a scientist would gain ZERO useful information.\n"
            "5. Do NOT refuse because sources are 'complex' or 'technical'.\n"
            "6. Do NOT refuse just because the answer is partial or indirect.\n"
        )
        user_prompt = (
            f"Sources:\n{context}\n\n"
            f"Question: {question}\n\n"
            f"Answer with inline citations:"
        )

        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.5,
            max_tokens=500,
        )
        answer = resp.choices[0].message.content.strip()

        # 模型返回 NO_ANSWER 时触发拒绝
        if answer.upper().startswith("NO_ANSWER"):
            return ""

        return answer

def build_generator(kind: str = "extractive"):
    if kind == "extractive":
        return ExtractiveGenerator()
    if kind == "deepseek":
        return DeepSeekGenerator()
    if kind == "qwen-vision":
        return QwenVisionGenerator()
    if kind == "bedrock":
        return BedrockGenerator()
    raise ValueError(f"未知 generator: {kind}")