# 结构化输出与约束解码

> 要求 LLM 返回 JSON。大部分时间得到 JSON。在生产中，"大部分"是问题。约束解码通过在采样前编辑 logits 将"大部分"变成"总是"。

**类型：** 构建
**语言：** Python
**先修课程：** Phase 5 · 17（聊天机器人）、Phase 5 · 19（子词分词）
**耗时：** 约 60 分钟

## 问题

分类器提示 LLM："返回 {positive, negative, neutral} 之一。"模型返回"The sentiment is positive — this review is overwhelmingly favorable because the customer explicitly states that they ..."。你的解析器崩溃。你的分类器 F1 为 0.0。

自由形式生成不是契约。是建议。生产系统需要契约。

2026 年存在三层。

1. **提示。** 礼貌请求。"只返回 JSON 对象。"在前沿模型上约 80% 有效，在较小模型上较少。
2. **原生结构化输出 API。** OpenAI `response_format`、Anthropic 工具使用、Gemini JSON mode。在支持的模式上可靠。供应商锁定。
3. **约束解码。** 在每个生成步骤修改 logits，使模型*不能*发出无效 token。按结构 100% 有效。在任何本地模型上工作。

本课为三者构建直觉并命名何时用哪个。

## 概念

![Constrained decoding masking invalid tokens at each step](../assets/constrained-decoding.svg)

**约束解码如何工作。** 在每个生成步骤，LLM 对整个词汇表（约 100k token）产生一个 logit 向量。*logit 处理器*坐在模型和采样器之间。它计算给定目标语法当前位置——JSON Schema、正则表达式、上下文无关语法——哪些 token 是有效的，并将所有无效 token 的 logits 设置为负无穷。剩余 logits 上的 softmax 将概率质量仅放在有效延续上。

2026 年的实现：

- **Outlines。** 将 JSON Schema 或正则表达式编译为有限状态机。每个 token 得到 O(1) 有效下一步查找。基于 FSM，所以递归模式需要扁平化。
- **XGrammar / llguidance。** 上下文无关语法引擎。处理递归 JSON Schema。解码开销接近零。OpenAI 在 2025 年结构化输出实现中 credited llguidance。
- **vLLM guided decoding。** 内置 `guided_json`、`guided_regex`、`guided_choice`、`guided_grammar`，通过 Outlines、XGrammar 或 lm-format-enforcer 后端。
- **Instructor。** 跨任何 LLM 的基于 Pydantic 的包装器。验证失败时重试。跨提供商，但不修改 logits——依靠重试 + 结构化输出感知提示。

### 反直觉的结果

约束解码通常比无约束生成*更快*。两个原因。首先，它缩小了下一个 token 的搜索空间。其次，聪明的实现完全跳过强制 token 的 token 生成（脚手架如 `{"name": "`——每个字节都是确定的）。

### 让你付出代价的陷阱

字段顺序重要。将 `answer` 放在 `reasoning` 前面，模型在思考之前就提交了答案。JSON 有效。答案是错的。没有验证抓住它。

```json
// BAD
{"answer": "yes", "reasoning": "because ..."}

// GOOD
{"reasoning": "... therefore ...", "answer": "yes"}
```

Schema 字段顺序是逻辑，不是格式。

## 构建

### 步骤 1：从零构建正则约束生成

见 `code/main.py` 获取独立 FSM 实现。30 行中的核心思想：

```python
def mask_logits(logits, valid_token_ids):
    mask = [float("-inf")] * len(logits)
    for tid in valid_token_ids:
        mask[tid] = logits[tid]
    return mask


def generate_constrained(model, tokenizer, prompt, fsm):
    ids = tokenizer.encode(prompt)
    state = fsm.initial_state
    while not fsm.is_accept(state):
        logits = model.next_token_logits(ids)
        valid = fsm.valid_tokens(state, tokenizer)
        logits = mask_logits(logits, valid)
        tok = sample(logits)
        ids.append(tok)
        state = fsm.transition(state, tok)
    return tokenizer.decode(ids)
```

FSM 跟踪我们满足了多少语法部分。`valid_tokens(state, tokenizer)` 计算哪些词汇表 token 可以推进 FSM 而不离开接受路径。

### 步骤 2：用 Outlines 处理 JSON Schema

```python
from pydantic import BaseModel
from typing import Literal
import outlines


class Review(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float
    evidence_span: str


model = outlines.models.transformers("meta-llama/Llama-3.2-3B-Instruct")
generator = outlines.generate.json(model, Review)

result = generator("Classify: 'The wait staff was attentive and the food arrived hot.'")
print(result)
# Review(sentiment='positive', confidence=0.93, evidence_span='attentive ... hot')
```

零验证错误。从不。FSM 使无效输出不可达。

### 步骤 3：用 Instructor 处理提供商无关的 Pydantic

```python
import instructor
from anthropic import Anthropic
from pydantic import BaseModel, Field


class Invoice(BaseModel):
    vendor: str
    total_usd: float = Field(ge=0)
    line_items: list[str]


client = instructor.from_anthropic(Anthropic())
invoice = client.messages.create(
    model="claude-opus-4-7",
    max_tokens=1024,
    response_model=Invoice,
    messages=[{"role": "user", "content": "Extract from: 'Acme Corp $420. Widget, Gizmo.'"}],
)
```

不同机制。Instructor 不触碰 logits。它将 schema 格式化到提示中，解析输出，验证失败时重试（默认 3 次）。适用于任何提供商。重试增加延迟和成本。跨提供商可移植性是卖点。

### 步骤 4：原生供应商 API

```python
from openai import OpenAI

client = OpenAI()
response = client.responses.create(
    model="gpt-5",
    input=[{"role": "user", "content": "Classify: 'The food was cold.'"}],
    text={"format": {"type": "json_schema", "name": "sentiment",
          "schema": {"type": "object", "required": ["sentiment"],
                     "properties": {"sentiment": {"type": "string",
                                                  "enum": ["positive", "negative", "neutral"]}}}}},
)
print(response.output_parsed)
```

服务器端约束解码。对于支持的模式，可靠性与 Outlines 平齐。不需要本地模型管理。锁定供应商。

## 陷阱

- **递归模式。** Outlines 将递归扁平化为固定深度。树结构输出（嵌套评论、AST）需要 XGrammar 或 llguidance（CFG-based）。
- **超大枚举。** 10,000 选项枚举编译慢或超时。切换到检索器：先预测 top-k 候选，约束为那些。
- **语法太严格。** 强制 `date: "YYYY-MM-DD"` 正则表达式，模型无法输出缺失日期的 `"unknown"`。模型通过捏造日期来补偿。允许 `null` 或哨兵。
- **过早提交。** 见上面的字段顺序陷阱。始终将推理放在前面。
- **无模式的供应商 JSON 模式。** 纯 JSON 模式只保证有效 JSON，不保证对你的用例有效*。始终提供完整 schema。

## 使用

2026 年技术栈：

| 场景 | 选择 |
|------|------|
| OpenAI/Anthropic/Google 模型，简单模式 | 原生供应商结构化输出 |
| 任何提供商，Pydantic 工作流，可以容忍重试 | Instructor |
| 本地模型，需要 100% 有效性，平模式 | Outlines (FSM) |
| 本地模型，递归模式 | XGrammar 或 llguidance |
| 自托管推理服务器 | vLLM guided decoding |
| 可容忍重试的批处理 | Instructor + 最便宜的模型 |

## 交付

保存为 `outputs/skill-structured-output-picker.md`：

```markdown
---
name: structured-output-picker
description: 选择结构化输出方法、schema 设计和验证计划。
version: 1.0.0
phase: 5
lesson: 20
tags: [nlp, llm, structured-output]
---

给定用例（提供商、延迟预算、模式复杂性、失败容忍度），输出：

1. 机制。原供应商结构化输出、Instructor 重试、Outlines FSM 或 XGrammar CFG。一句话理由。
2. Schema 设计。字段顺序（推理第一，答案最后）、"unknown"的可空字段、枚举 vs 正则表达式、必需字段。
3. 失败策略。最大重试次数、回退模型、优雅 `null` 处理、分布外拒绝。
4. 验证计划。Schema 合规率（目标 100%）、语义有效性（LLM 判断）、字段覆盖率、延迟 p50/p99。

拒绝将 `answer` 或 `decision` 放在推理字段之前的任何设计。拒绝使用裸 JSON 模式而不用 schema。标记递归模式在仅 FSM 库后面。
```

## 练习

1. **简单。** 提示小开放权重模型（例如 Llama-3.2-3B）不带约束解码获取 `Review(sentiment, confidence, evidence_span)`。在 100 条评论上测量解析为有效 JSON 的比例。
2. **中等。** 同语料库用 Outlines JSON 模式。比较合规率、延迟和语义准确率。
3. **困难。** 从零为电话号码（`\d{3}-\d{3}-\d{4}`）实现正则约束解码器。在 1000 个样本上验证 0 无效输出。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 约束解码 | 强制有效输出 | 在每个生成步骤将无效 token logits 屏蔽。 |
| Logit 处理器 | 约束的东西 | 函数：`(logits, state) -> masked_logits`。 |
| FSM | 有限状态机 | 编译的语法表示；O(1) 有效下一步查找。 |
| CFG | 上下文无关语法 | 处理递归的语法；比 FSM 慢但更表达。 |
| Schema 字段顺序 | 重要吗？ | 是——第一个字段提交；始终将推理放在答案之前。 |
| 引导解码 | vLLM 的名字 | 相同概念，集成到推理服务器中。 |
| JSON 模式 | OpenAI 早期版本 | 保证 JSON 语法；不保证 schema 匹配。 |

## 延伸阅读

- [Willard, Louf (2023). Efficient Guided Generation for LLMs](https://arxiv.org/abs/2307.09702) —— Outlines 论文。
- [XGrammar paper (2024)](https://arxiv.org/abs/2411.15100) —— 快速 CFG 约束解码。
- [vLLM — Structured Outputs](https://docs.vllm.ai/en/latest/features/structured_outputs.html) —— 推理服务器集成。
- [OpenAI — Structured Outputs guide](https://platform.openai.com/docs/guides/structured-outputs) —— API 参考 + gotchas。
- [Instructor library](https://python.useinstructor.com/) —— Pydantic + 跨提供商重试。
- [JSONSchemaBench (2025)](https://arxiv.org/abs/2501.10868) —— 基准测试 6 个约束解码框架。