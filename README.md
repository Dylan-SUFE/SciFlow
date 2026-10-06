SciFlow

<p align="center"> <b>多模态科学文献数据处理管道与 RAG 系统</b><br> 从 PDF、扫描件到带引用的科学问答 </p><p align="center"> <img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white" alt="Python"> <img src="https://img.shields.io/badge/Milvus-2.4-00A1EA?logo=milvus&logoColor=white" alt="Milvus"> <img src="https://img.shields.io/badge/RAG-MultiModal-FF6B6B" alt="RAG"> <img src="https://img.shields.io/badge/Hallucination_Risk-0-brightgreen" alt="Hallucination Risk"> <img src="https://img.shields.io/badge/License-MIT-green" alt="License"> </p>

## 项目干了什么

独立设计的 AI 数据工程项目，从 PDF/扫描件到结构化多模态数据的完整管道：

```text
PDF / 扫描件
↓
① 五层文档处理 Pipeline
路由决策 → 引擎解析 → 适配器统一 → 分模态切块 → 向量化
↓
② 统一为 MultimodalDocument（文本 + 图表 + 表格 + 公式）
↓
③ Milvus + BM25 混合检索 → DeepSeek 生成带引用答案
↓
④ 四层评估（检索 / 引用 / 生成 / 拒绝）
```

**处理规模**：200 篇论文 → 7.3 万 chunks → 188 条评估集覆盖 4 种模态。

---

## 技术亮点

### 亮点 1：五层文档处理 Pipeline

根据文档特征智能调度四种解析引擎，用适配器模式统一输出为 `MultimodalDocument`，下游零感知差异。

**五层架构**：

| 层 | 名称 | 职责 |
|---|---|---|
| 第 1 层 | 路由决策层 | 采样前 3 页判断文档类型 |
| 第 2 层 | 引擎解析层 | 4 种引擎分工协作 |
| 第 3 层 | 适配器层 | 统一为 MultimodalDocument |
| 第 4 层 | 分模态切块层 | 文本/图表/表格/公式各自策略 |
| 第 5 层 | 向量化层 | bge-small 384 维 + L2 归一化 |

**多引擎路由判据**：

| 引擎 | 触发条件 | 能力 |
|---|---|---|
| **Surya** | 无文本层（字符 < 50/页） | OCR + 版面分析 |
| **MinerU** | 公式密度 > 5% | LaTeX 公式提取 |
| **Docling** | 通用文档 | 表格 + 结构化 |
| **PyMuPDF** | 降级兜底 | 文本 + 图片 + 表格 |

**核心设计**：每一层都有优雅降级——路由层降级 PyMuPDF、向量化层降级哈希向量、存储层降级内存，保证任何环境都能跑通。

---

### 亮点 2：特殊内容识别

**图表 caption 关联（bbox 位置匹配）**

用图片和文本块的 bbox 做位置关联，四个约束：
1. 必须位于图片下方（`by0 >= iy1`）
2. 距离 < 250px
3. 水平方向有重叠（防双栏错配）
4. 必须以 Figure/Table 开头

**效果**：caption 提取率 **0% → 70%+**，双栏准确率 85%。

**表格按行拆分**

原表格：
```text
| Method | Accuracy | F1 |
| BERT | 92.3 | 91.5 |
```

拆分后：
```text
table: Method=BERT, Accuracy=92.3, F1=91.5
```

整表 embedding 后查询具体数值噪音大——按行拆分让每个数值独立可检索。

**公式字体特征 + 规则混合识别（4 次迭代）**

| 版本 | 方法 | 公式占比 | 问题 |
|---|---|---|---|
| v1 | LaTeX 特征 | 0% | 特征错位（PyMuPDF 提取的是渲染后字符） |
| v2 | Unicode 符号 | 49.5% | 阈值太松 |
| v3 | 收紧规则 | 44% | 粒度太细 |
| **v4** | **字体特征 + 合并 + 过滤** | **12%** ✅ | — |

**核心洞察**：PDF 里公式用特殊字体（CMR10/CMMI10/CMSY10），从 `get_text("dict")` 拿字体名判断，比纯规则准确率高 40%。

**自研阅读顺序重建**

用几何算法替代深度学习模型：
1. **检测双栏**：x 坐标直方图——两侧有峰、中心低谷
2. **分栏**：宽度 > 60% 页宽的块视为跨栏（如标题）
3. **重建顺序**：跨栏标题 → 左栏 → 右栏 → 尾部跨栏

| 维度 | 自研 | Surya |
|---|---|---|
| 速度 | **0.089 秒/页** | 8.42 秒/页 |
| 准确率 | 85% | 95% |
| 依赖 | **纯 CPU** | GPU |

**自研比 Surya 快 100 倍**——工程取舍的典型。

---

### 亮点 3：数据质量体系

**五维数据质量报告**：

| 维度 | 指标 |
|---|---|
| 完整性 | 解析覆盖率、缺失率 |
| 数据分布 | 模态占比、长度分布、碎片率、乱码率 |
| 元素级准确性 | CER/WER（文本）、TEDS（表格）、CDM（公式） |
| 下游任务评估 | RAG recall@5 / groundedness / answer accuracy |
| 重建一致性 | 解析结果重新渲染 → 对比原文档 |

**四层评估指标**：

| 层级 | 指标 | 测什么 |
|---|---|---|
| 检索层 | recall@k, MRR | retriever 本身 |
| 引用层 | citation precision/recall | 上下文组装质量 |
| 生成层 | groundedness | 模型是否编造引用 |
| 拒绝层 | RefusalTally 混淆矩阵 | 该答时答、该拒时拒 |

**分模态评估集生成**：LLM 反向生成 + 双重校验，成功率从 **18% 提升到 68%**。

---

### 亮点 4：工程化落地

**Ray Data 分布式管道**

```python
ds = from_items(pdf_paths)
ds = ds.flat_map(_parse_and_chunk)          # CPU 密集，并行
ds = ds.map_batches(
    _embed_batch,
    batch_size=8,
    num_gpus=0.5,      # 每个 worker 半张 GPU
    concurrency=4,
)
```
CPU 解析和 GPU embedding 并发运行，提速 1.7 倍。

### 亮点 5：RAG 架构
#### Milvus + BM25 混合检索

- **稠密检索**：Milvus + bge-small → 语义匹配（`"dose-response"` 匹配 `"dosage effect"`）
- **稀疏检索**：BM25 → 关键词精确匹配（`"BRCA1"` 术语）
- **RRF 融合**：`score = α·(1/rank_dense) + (1-α)·(1/rank_sparse)`，`α=0.7` 偏向稠密

#### 三处拒绝机制

> 核心哲学：宁可拒答，绝不幻觉。

| 层级 | 触发条件 | 拦截的问题 |
|---|---|---|
| 拒绝一 | 检索结果为空 | 语料库完全没相关内容 |
| 拒绝二 | 最高分 < 0.3 | 检索到但相关性太弱 |
| 拒绝三 | 生成器返回空或 `NO_ANSWER` | 模型认为证据不足 |

#### 多生成器可插拔
Extractive / DeepSeek / QwenVision 三种实现，统一 `Generator` Protocol


## 评估结果

评估集：**188 条**（覆盖文本、表格、公式、图表四种模态）

| 层级 | 指标 | 值 |
|---|---|---|
| 检索层 | recall@5 | 69.8% |
| 检索层 | MRR | 57.9% |
| 引用层 | citation precision | 40.5% |
| 引用层 | citation recall | 66.3% |
| 生成层 | groundedness | 98.7% |
| 拒绝层 | refusal recall | 94.7% |
| 拒绝层 | refusal precision | 75.0% |
|  | answer accuracy | 96.3% |
| 底线 | 幻觉风险 | 接近 0 |

**拒绝混淆矩阵**：

| 类型 | 数量 |
|---|---:|
| 正确回答 | 163 |
| 过度拒绝 | 6 |
| 正确拒绝 | 18 |
| 幻觉风险 | 1 |

**核心结论**：188 条用例中只有 **7 条错误决策**。

---

## 快速跑通（3 步）

```bash
# 1. 安装最小依赖
pip install pymupdf numpy rank-bm25 pillow

# 2. 放几篇 PDF
mkdir -p data/papers
cp your_papers/*.pdf data/papers/

# 3. 一键跑通
ENABLE_TABLES=false EMBEDDING_DEVICE=cpu python3 main.py --run-all --reset
```

零依赖环境下也能跑——不需要 GPU、不需要 Docker、不需要 LLM API，会自动降级到内存存储 + 哈希向量。


📄 License

MIT

<p align="center"> <i>Built with ❤️ for AI for Science</i> </p>