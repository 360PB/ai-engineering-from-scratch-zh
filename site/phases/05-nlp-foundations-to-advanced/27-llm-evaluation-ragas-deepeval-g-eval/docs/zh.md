# LLM 评估 — RAGAS、DeepEval、G-Eval

> 精确匹配和 F1 无法捕捉语义等价。人工审核无法规模化。LLM 即评判者是生产环境的答案——配合足够的校准才能信任这个数字。

**类型：** 构建
**语言：** Python
**前置要求：** Phase 5 · 13（问答）、Phase 5 · 14（信息检索）
**时长：** 约 75 分钟

## 问题

你的 RAG 系统回答："2007 年 6 月 29 日。"
标准答案是："June 29, 2007."
精确匹配得分 0。F1 得分约 75%。人工评分 100%。

现在乘以 10000 个测试用例。再乘以每一次对检索器、分块、提示词或模型的改动。你需要一个评估器，能理解语义、成本低廉且可规模化运行、不会谎报回归问题，并暴露正确的失败模式。

2026 年有三个框架解决这个问题。

- **RAGAS。** 检索增强生成评估（Retrieval-Augmented Generation ASsessment）。四个 RAG 指标（忠实度、答案相关性、上下文精确率、上下文召回率），后端基于 NLI + LLM 评判。研究支撑、轻量级。
- **DeepEval。** LLM 的 pytest。G-Eval、任务完成度、幻觉、偏见指标。CI/CD 原生。
- **G-Eval。** 一种方法（也是 DeepEval 的指标）：带思维链的 LLM 即评判者、自定义标准、0-1 得分。

三者都依赖 LLM 即评判者。本课程建立对该方法及其信任层的基本认识。

## 概念

![Four evaluation dimensions, LLM-as-judge architecture](../assets/llm-evaluation.svg)

**LLM 即评判者。** 用一个 LLM 替代静态指标来对输出进行评分。给定 `(query, context, answer)`，向评判 LLM 提问："对忠实度评分 0-1。"返回得分。

为什么有效：LLM 以极低成本近似人类判断。GPT-4o-mini 每个评分案例约 $0.003，使 1000 样本的回归评估运行费用低于 $5。

为什么会静默失效：

1. **评判者偏差。** 评判者偏爱更长的答案、同一个模型家族的答案、与提示风格一致的答案。
2. **JSON 解析失败。** 格式错误的 JSON → NaN 得分 → 被静默排除在聚合之外。RAGAS 用户深受其苦。用 try/except + 显式失败模式来拦截。
3. **模型版本漂移。** 升级评判者会改变所有指标。冻结评判模型 + 版本。

**RAG 四项指标。**

| 指标 | 问题 | 后端 |
|------|------|------|
| Faithfulness（忠实度） | 答案中的每个声明是否来自检索到的上下文？ | 基于 NLI 的蕴含判断 |
| Answer relevance（答案相关性） | 答案是否回答了问题？ | 从答案生成假设问题；与真实问题比较 |
| Context precision（上下文精确率） | 检索到的 chunk 中有多少是相关的？ | LLM 评判 |
| Context recall（上下文召回率） | 检索是否返回了所需的一切？ | 针对标准答案的 LLM 评判 |

**G-Eval。** 定义自定义标准："答案是否引用了正确的来源？"框架自动扩展为思维链评估步骤，然后评分 0-1。适用于 RAGAS 未覆盖的领域特定质量维度。

**校准。** 在有真人标签做相关验证之前，不要信任原始评判得分。运行 100 个人工标注的例子。绘制评判者 vs 人工得分图。计算 Spearman rho。如果 rho < 0.7，你的评判标准需要改进。

## 构建

### 步骤 1：基于 NLI 的忠实度（RAGAS 风格）

```python
from typing import Callable
from transformers import pipeline

nli = pipeline("text-classification",
               model="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
               top_k=None)

# `llm` 是任意可调用对象：prompt str -> 生成的 str。
# 示例: llm = lambda p: client.messages.create(model="claude-haiku-4-5", ...).content[0].text
LLM = Callable[[str], str]


def atomic_claims(answer: str, llm: LLM) -> list[str]:
    prompt = f"""Break this answer into simple factual claims (one per line):
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

将答案分解为原子声明。用 NLI 检查每个声明与检索上下文的一致性。忠实度 = 支持声明比例。

### 步骤 2：答案相关性

```python
import numpy as np
from sentence_transformers import SentenceTransformer

# encoder: 任意实现 .encode(texts, normalize_embeddings=True) -> ndarray 的模型
# 例如: encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")

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

如果答案暗示的问题与所问的问题不同，相关性就会下降。

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

评估步骤就是评分标准。显式步骤比隐式"评分 0-1"提示更稳定。

### 步骤 4：CI 门槛

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

作为 pytest 文件交付。每次 PR 都运行。回归则阻止合并。

### 步骤 5：从零构建演示评估

参见 `code/main.py`。忠实度（答案声明与上下文的重叠）和相关性（答案 token 与问题 token 的重叠）的纯标准库近似实现。非生产用。仅展示形态。

## 陷阱

- **无校准。** 与人工标签相关性 0.3 的评判者是噪音。上线前必须有一次校准运行。
- **自评。** 用同一个 LLM 生成和评判会使得分膨胀 10-20%。评判者使用不同的模型家族。
- **配对评判中的位置偏差。** 评判者偏爱第一个选项。始终随机排序并运行两个方向。
- **原始聚合掩盖失败。** 平均得分 0.85 往往掩盖了 5% 的灾难性失败。始终检查底部分位数。
- **标准数据集腐坏。** 随时间漂移的未版本化评估集破坏长期比较。每次变更都给数据集打标签。
- **LLM 成本。** 规模化下，评判调用是成本大头。使用满足校准阈值的最便宜模型。GPT-4o-mini、Claude Haiku、Mistral-small。

## 使用

2026 年技术栈：

| 用例 | 框架 |
|------|------|
| RAG 质量监控 | RAGAS（4 项指标） |
| CI/CD 回归门槛 | DeepEval + pytest |
| 自定义领域标准 | DeepEval 中的 G-Eval |
| 在线实时流量监控 | 无参考模式的 RAGAS |
| 人工介入抽查 | 带标注 UI 的 LangSmith 或 Phoenix |
| 红队 / 安全评估 | Promptfoo + DeepEval |

典型组合：RAGAS 用于监控，DeepEval 用于 CI，G-Eval 用于新维度。三个都跑；它们会有效地产生分歧。

## 上线

保存为 `outputs/skill-eval-architect.md`：

```markdown
---
name: eval-architect
description: 设计带校准评判器和 CI 门槛的 LLM 评估计划。
version: 1.0.0
phase: 5
lesson: 27
tags: [nlp, evaluation, rag]
---

给定用例（RAG / agent / 生成式任务），输出：

1. 指标。忠实度 / 相关性 / 上下文精确率 / 上下文召回率 + 任何带标准定义的自定义 G-Eval 指标。
2. 评判模型。指定模型 + 版本，成本 vs 准确率的理由。
3. 校准。人工标注集规模，目标 Spearman rho vs 人工 > 0.7。
4. 数据集版本化。打标签策略、变更日志、分层策略。
5. CI 门槛。每个指标阈值、回归窗口逻辑、底部分位数告警。

拒绝未经 ≥50 个人工标注样本测试的评判者。拒绝自评（同模型生成 + 评判）。拒绝无底部 10% 暴露的纯聚合报告。标记任何评判模型升级落地时未并行运行基线评估的管道。
```

## 练习

1. **简单。** 在 10 个已知幻觉的 RAG 例子上使用 RAGAS。验证忠实度指标能捕捉到每一个。
2. **中等。** 人工标注 50 个 QA 答案的正确性 0-1。用 G-Eval 评分。测量评判者与人工的 Spearman rho。
3. **困难。** 用 DeepEval 构建 pytest CI 门槛。故意让检索器回归。验证门槛失败。通过检查最低 10% 的阈值来添加底部分位数告警。

## 关键术语

| 术语 | 人们通常说 | 实际含义 |
|------|-----------|---------|
| LLM-as-judge | 用 LLM 评分 | 给定评分标准，提示评判模型对输出评分 0-1。 |
| RAGAS | RAG 指标库 | 开源评估框架，包含 4 个无参考 RAG 指标。 |
| Faithfulness | 答案有依据吗？ | 答案声明中被检索上下文蕴含的比例。 |
| Context precision | 检索的 chunk 相关吗？ | top-K chunk 中实际重要的比例。 |
| Context recall | 检索找到一切了吗？ | 被检索 chunk 支持的标准答案声明比例。 |
| G-Eval | 自定义 LLM 评判 | 评分标准 + 思维链评估步骤 + 0-1 得分。 |
| Calibration | 信任但验证 | 评判者得分与人工得分的 Spearman 相关性。 |

## 延伸阅读

- [Es et al. (2023). RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217) — RAGAS 论文。
- [Liu et al. (2023). G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment](https://arxiv.org/abs/2303.16634) — G-Eval 论文。
- [DeepEval docs](https://deepeval.com/docs/metrics-introduction) — 开源生产技术栈。
- [Zheng et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685) — 偏差、校准与局限。
- [MLflow GenAI Scorer](https://mlflow.org/blog/third-party-scorers) — 统一框架，整合 RAGAS、DeepEval、Phoenix。