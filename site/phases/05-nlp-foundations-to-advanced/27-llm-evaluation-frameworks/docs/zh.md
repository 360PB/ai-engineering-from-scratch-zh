# LLM 评估 — RAGAS、DeepEval、G-Eval

> 精确匹配和 F1 丢失语义等价。人工审查不可扩展。LLM 即评判者是生产答案——带有足够的校准来信任数字。

**类型：** 动手实现
**语言：** Python
**前置知识：** Phase 5 · 13（问答），Phase 5 · 14（信息检索）
**时长：** 约75分钟

## 问题

你的 RAG 系统回答："2007年6月29日。"
标准答案是："June 29, 2007."
精确匹配得分 0。F1 约 75%。人类会评 100%。

现在乘以 10,000 个测试用例。再乘以每次改变检索器、分块、提示或模型。你需要一个理解语义、廉价规模化运行、不谎报回归、并暴露正确失败模式的评估器。

2026年有三个框架解决了这个问题。

- **RAGAS。** 检索增强生成评估。四个 RAG 指标（faithfulness、answer-relevance、context-precision、context-recall），带 NLI + LLM-judge 后端。研究支撑，轻量级。
- **DeepEval。** LLMs 的 pytest。G-Eval、任务完成、幻觉、偏差指标。CI/CD 原生。
- **G-Eval。** 一种方法（也是一个 DeepEval 指标）：带思维链的 LLM 即评判者，自定义标准，0-1 分。

三个都依赖 LLM 即评判者。本课建立对方法和信任层的感觉。

## 核心概念

**LLM 即评判者。** 用评分 Rubric 评判输出的 LLM 替换静态指标。给定 `(query, context, answer)`，提示评判 LLM：" faithfulness 上评分 0-1。"返回分数。

为什么有效：LLM 以很小一部分成本近似人类判断。GPT-4o-mini 约 $0.003 每评分案例，使 1000 样本回归评估运行成本低于 $5。

为什么静默失败：

1. **评判者偏差。** 评判者偏好更长的答案、来自同一模型家族的答案、与提示风格匹配的答案。
2. **JSON 解析失败。** 坏 JSON → NaN 分数 → 被静默排除在聚合之外。RAGAS 用户了解这种痛苦。用 try/except + 显式失败模式做门控。
3. **跨模型版本漂移。** 升级评判者会改变每个指标。冻结评判者模型 + 版本。

**RAG 四指标。**

| 指标 | 问题 | 后端 |
|--------|----------|---------|
| Faithfulness | 答案中的每个声明都来自检索上下文吗？ | 基于 NLI 的蕴含 |
| Answer relevance | 答案解决了问题吗？ | 从答案生成假设性问题；与真实问题比较 |
| Context precision | 检索的 chunk 中有多少是相关的？ | LLM-judge |
| Context recall | 检索是否返回了所有需要的内容？ | LLM-judge 对标准答案 |

**G-Eval。** 定义自定义标准："答案是否引用了正确的来源？"框架自动展开为思维链评估步骤，然后评分 0-1。适用于 RAGAS 未覆盖的领域特定质量维度。

**校准。** 在信任原始评判分数之前，永不缺少与人工标签的相关性。运行 100 个人工标注的例子。绘制评判者 vs 人类。计算 Spearman rho。如果 rho < 0.7，你的评判者 rubric 需要改进。

## 动手实现

### 步骤 1：用 NLI 做 faithfulness（RAGAS 风格）

```python
from typing import Callable
from transformers import pipeline

nli = pipeline("text-classification",
               model="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
               top_k=None)

def atomic_claims(answer: str, llm: LLM) -> list[str]:
    prompt = f"""将这个答案分解为简单的事实声明（每行一个）：
{answer}
"""
    return llm(prompt).splitlines()


def faithfulness(answer: str, context: str, llm: LLM) -> float:
    claims = atomic_claims(answer, llm)
    if not claims:
        return 0.0
    supported = 0
    for claim in claims:
        result = nli({"text": context, "text_pair": claim})[0]
        entail = next((s for s in result if s["label"] == "entailment"), None)
        if entail and entail["score"] > 0.5:
            supported += 1
    return supported / len(claims)
```

将答案分解为原子声明。用 NLI 检查每个声明对检索上下文的蕴含。Faithfulness = 支持的比例。

### 步骤 2：答案相关性

```python
import numpy as np
from sentence_transformers import SentenceTransformer

def answer_relevance(question: str, answer: str, encoder, llm: LLM, n: int = 3) -> float:
    prompt = f"Write {n} questions this answer could be the answer to:\n{answer}"
    generated = [line for line in llm(prompt).splitlines() if line.strip()][:n]
    if not generated:
        return 0.0
    q_emb = np.asarray(encoder.encode([question], normalize_embeddings=True)[0])
    g_embs = np.asarray(encoder.encode(generated, normalize_embeddings=True))
    sims = [float(q_emb @ g_emb) for g_emb in g_embs]
    return sum(sims) / len(sims)
```

### 步骤 3：G-Eval 自定义指标

```python
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams, LLMTestCase

metric = GEval(
    name="Correctness",
    criteria="The answer should be factually accurate and match the expected output.",
    evaluation_steps=[
        "Read the expected output.",
        "Read the actual output.",
        "List factual claims in the actual output.",
        "For each claim, mark supported or unsupported by the expected output.",
        "Return score = fraction supported.",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
)

test = LLMTestCase(input="When was the first iPhone released?",
                   actual_output="June 29th, 2007.",
                   expected_output="June 29, 2007.")
metric.measure(test)
print(metric.score, metric.reason)
```

评估步骤就是 rubric。显式步骤比隐式"评分 0-1"提示更稳定。

### 步骤 4：CI 门控

```python
import deepeval
from deepeval.metrics import FaithfulnessMetric, ContextualRelevancyMetric


def test_rag_system():
    cases = load_regression_cases()
    faith = FaithfulnessMetric(threshold=0.85)
    rel = ContextualRelevancyMetric(threshold=0.7)
    for case in cases:
        faith.measure(case)
        assert faith.score >= 0.85, f"faithfulness regression on {case.id}"
        rel.measure(case)
        assert rel.score >= 0.7, f"relevancy regression on {case.id}"
```

作为 pytest 文件交付。每次 PR 都运行。回归时阻止合并。

## 陷阱

- **无校准。** 与人工标签相关性 0.3 的评判者是噪声。交付前要求校准运行。
- **自我评估。** 用同一 LLM 生成和评判会将分数膨胀 10-20%。用不同模型家族做评判者。
- **配对评判中的位置偏差。** 评判者偏好第一个选项。总随机化顺序并运行双向。
- **原始聚合隐藏失败。** 平均分 0.85 通常隐藏 5% 的灾难性失败。总检查底部分位数。
- **黄金数据集腐烂。** 未版本化的评估集随时间漂移会破坏纵向比较。每次变更时标记数据集。
- **LLM 成本。** 规模化后，评判调用主导成本。用满足校准阈值的最便宜模型。GPT-4o-mini、Claude Haiku、Mistral-small。

## 用现成库

2026年技术栈：

| 用例 | 框架 |
|---------|-----------|
| RAG 质量监控 | RAGAS（4指标） |
| CI/CD 回归门控 | DeepEval + pytest |
| 自定义领域标准 | DeepEval 内的 G-Eval |
| 在线实时流量监控 | 无参考模式的 RAGAS |
| 人工介入抽查 | LangSmith 或 Phoenix 带标注 UI |
| 红队 / 安全评估 | Promptfoo + DeepEval |

## 产出

保存为 `outputs/skill-eval-architect.md`：

```markdown
---
name: eval-architect
description: 设计带校准评判和 CI 门控的 LLM 评估计划。
version: 1.0.0
phase: 5
lesson: 27
tags: [nlp, evaluation, rag]
---

给定用例（RAG / Agent / 生成任务），输出：

1. 指标。Faithfulness / relevance / context-precision / context-recall + 带标准的任何自定义 G-Eval 指标。
2. 评判模型。命名模型 + 版本，成本 vs 准确率的原因。
3. 校准。人工标注集规模，vs 人工的目标 Spearman rho > 0.7。
4. 数据集版本控制。标记策略，变更日志，分层。
5. CI 门控。每指标阈值，回归窗口逻辑，底部告警。

拒绝依赖未在 ≥50 个人工标注示例上测试的评判者。拒绝自我评估（同一模型生成 + 评判）。拒绝无底部 10% 暴露的纯聚合报告。
```

## 练习

1. **简单。** 在10个有已知幻觉的 RAG 示例上使用 RAGAS。验证 faithfulness 指标捕获了每个。
2. **中等。** 人工标注50个 QA 答案正确性 0-1。用 G-Eval 评分。测量评判者与人类间的 Spearman rho。
3. **困难。** 用 DeepEval 构建 pytest CI 门控。故意回归检索器。验证门控失败。在底部 10% 阈值检查添加底部告警。

## 关键术语

| 术语 | 行话 | 实际含义 |
|------|-----------------|-----------------------|
| LLM-as-judge | 用 LLM 评分 | 提示评判模型按 rubric 评分输出 0-1。 |
| RAGAS | RAG 指标库 | 带4个无参考 RAG 指标的开源评估框架。 |
| Faithfulness | 答案有据吗？ | 由检索上下文蕴含的答案声明比例。 |
| Context precision | 检索的 chunk 相关吗？ | top-K chunk 中实际重要的比例。 |
| Context recall | 检索到所有内容了吗？ | 由检索 chunk 支持的标准答案声明比例。 |
| G-Eval | 自定义 LLM 评判 | Rubric + 思维链评估步骤 + 0-1 分。 |
| Calibration | 信任但验证 | 评判分数与人工分数间的 Spearman 相关性。 |

## 扩展阅读

- [Es et al. (2023). RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217) — RAGAS 论文。
- [Liu et al. (2023). G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment](https://arxiv.org/abs/2303.16634) — G-Eval 论文。
- [DeepEval 文档](https://deepeval.com/docs/metrics-introduction) — 开源生产栈。
- [Zheng et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685) — 偏差、校准、局限。
- [MLflow GenAI Scorer](https://mlflow.org/blog/third-party-scorers) — 整合 RAGAS、DeepEval、Phoenix 的统一框架。