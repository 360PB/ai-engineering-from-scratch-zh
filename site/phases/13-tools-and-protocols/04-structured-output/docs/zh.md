# 结构化输出 —— JSON Schema、Pydantic、Zod、约束解码

> "请模型好好返回 JSON" 在前沿模型上也有 5% 到 15% 的失败率。结构化输出通过约束解码缩小这个差距：模型在字面上被阻止发出违反 schema 的 token。OpenAI 的严格模式、Anthropic 的 schema 类型工具使用、Gemini 的 `responseSchema`、Pydantic AI 的 `output_type` 和 Zod 的 `.parse` 是同一思想的五种表面形式。本课构建 schema 验证器和严格模式合约，学习者将用于每个生产提取管道。

**类型：** 构建
**语言：** Python（标准库、JSON Schema 2020-12 子集）
**前置要求：** Phase 13 · 02（函数调用深度解析）
**时间：** 约 75 分钟

## 学习目标

- 使用正确的约束（enum、min/max、required、pattern）为提取目标编写 JSON Schema 2020-12。
- 解释严格模式和约束解码为什么与"生成后验证"提供不同的保证。
- 区分三种失败模式：解析错误、schema 违规、模型拒绝。
- 发布一个带类型化修复和类型化拒绝处理的提取管道。

## 问题

读取采购订单邮件的 Agent 需要将自由文本转换为 `{customer, line_items, total_usd}`。三种方法。

**方法一：提示生成 JSON。** "请以 JSON 回复，包含 customer、line_items、total_usd 字段。" 在前沿模型上 85% 到 95% 的时间有效。六种失败方式：缺少括号、尾随逗号、错误类型、幻觉字段、在 token 限制处截断、泄漏散文如"这是你的 JSON："。

**方法二：生成后验证。** 自由生成，解析，根据 schema 验证，失败则重试。可靠但昂贵——每次重试都要付费，而且截断 bug 每次发生都会额外消耗一轮。

**方法三：约束解码。** 提供商在解码时强制执行 schema。无效 token 被屏蔽在采样分布之外。输出保证可解析，保证通过验证。失败归结为一种模式：拒绝（模型判定输入不符合 schema）。

2026 年的每个前沿提供商都提供某种形式的方法三。

- **OpenAI。** `response_format: {type: "json_schema", strict: true}` 加响应的 `refusal`（模型拒绝时）。
- **Anthropic。** 对 `tool_use` 输入的 schema 强制执行；`stop_reason: "refusal"` 不存在，但无工具调用的 `end_turn` 是信号。
- **Gemini。** 请求级别的 `responseSchema`；2026 年 Gemini 为选定类型提供 token 级语法约束。
- **Pydantic AI。** `output_type=InvoiceModel` 发出类型化为 `InvoiceModel` 的结构化 `RunResult`。
- **Zod（TypeScript）。** 运行时解析器，根据 Zod schema 验证提供商输出；与 OpenAI 的 `beta.chat.completions.parse` 配对。

共同点：声明 schema 一次，端到端强制执行。

## 概念

### JSON Schema 2020-12 —— 通用语言

每个提供商都接受 JSON Schema 2020-12。你最常用的构造：

- `type`：object、array、string、number、integer、boolean、null 之一。
- `properties`：字段名到子 schema 的映射。
- `required`：必须出现的字段名列表。
- `enum`：允许值的闭集。
- `minimum` / `maximum`（数字）、`minLength` / `maxLength` / `pattern`（字符串）。
- `items`：应用于每个数组元素的子 schema。
- `additionalProperties`：`false` 禁止额外字段（不同模式默认值不同）。

OpenAI 严格模式增加了三个要求：每个属性必须列在 `required` 中、到处 `additionalProperties: false`、无未解析 `$ref`。违反这些，API 在请求时返回 400。

### Pydantic，Python 绑定

Pydantic v2 通过 `model_json_schema()` 从 dataclass 形状的模型生成 JSON Schema。Pydantic AI 包装了这一点，所以你可以写：

```python
class Invoice(BaseModel):
    customer: str
    line_items: list[LineItem]
    total_usd: Decimal
```

Agent 框架将 schema 转换到边缘的 OpenAI 严格模式、Anthropic `input_schema` 或 Gemini `responseSchema`。模型输出作为类型化 `Invoice` 实例返回。验证错误抛出带类型化错误路径的 `ValidationError`。

### Zod，TypeScript 绑定

Zod（`z.object({customer: z.string(), ...})`）是 TS 等价物。OpenAI 的 Node SDK 暴露 `zodResponseFormat(Invoice)`，转换为 API 的 JSON Schema 载荷。

### 拒绝

严格模式无法强制模型回答。如果输入不符合 schema（"邮件是一首诗，不是发票"），模型发出包含原因的 `refusal` 字段。你的代码必须将其作为一级结果处理，而非失败。拒绝作为安全信号也很有用：模型被要求从受保护内容邮件中提取信用卡号码时，会返回带有附加安全原因的拒绝。

### 开源的约束解码

开源实现使用三种技术：

1. **基于语法的解码**（`outlines`、`guidance`、`lm-format-enforcer`）：从 schema 构建确定性有限自动机；每一步，屏蔽违反 FSM 的 token 的 logit。
2. **带 JSON 解析器的 Logit 屏蔽**：与模型同步运行流式 JSON 解析器；每一步，计算有效下一 token 集合。
3. **带验证器的推测解码**：廉价草稿模型提议 token，验证器强制执行 schema。

商业提供商在幕后选择其中之一。2026 年的最新技术比纯生成在短结构化输出上更快，在长输出上速度大致相同。

### 三种失败模式

1. **解析错误。** 输出不是有效 JSON。严格模式下不可能发生。非严格提供商上仍可能发生。
2. **Schema 违规。** 输出可解析但违反 schema。严格模式下不可能发生。在严格模式之外常见。
3. **拒绝。** 模型拒绝。必须作为类型化结果处理。

### 重试策略

在严格模式之外（Anthropic 工具使用、非严格 OpenAI、旧版 Gemini），恢复模式是：

```
generate -> parse -> validate -> if fail, inject error and retry, max 3x
```

一次重试通常足够。三次重试捕获弱模型偶发问题。超过三次说明 schema 有问题：模型对于某些输入无法满足它，提示或 schema 需要修复。

### 小模型支持

约束解码适用于小模型。带语法强制的 3B 参数开源模型在结构化任务上优于 70B 参数模型加原始提示。这是结构化输出对生产有意义的主要原因：它将可靠性与模型大小解耦。

## 使用它

`code/main.py` 发货一个纯标准库的最小 JSON Schema 2020-12 验证器（类型、required、enum、min/max、pattern、items、additionalProperties）。它包装 `Invoice` schema 并用假 LLM 输出运行验证器，展示解析错误、schema 违规和拒绝路径。在生产中将假输出替换为任意提供商的真实响应。

要注意的点：

- 验证器返回带路径和消息的类型化 `[ValidationError]` 列表。这是你想在重试提示中暴露的形状。
- 拒绝分支不重试。记录并返回类型化拒绝。Phase 14 · 09 将拒绝用作安全信号。
- `additionalProperties: false` 检查在对抗性测试输入上触发，展示为什么严格模式关闭了幻觉字段的大门。

## 发布它

本课生成 `outputs/skill-structured-output-designer.md`。给定一个自由文本提取目标（发票、支持工单、简历等），该 Skill 生成一个严格模式兼容的 JSON Schema 2020-12 和镜像它的 Pydantic 模型，并填入类型化拒绝和重试处理存根。

## 练习

1. 运行 `code/main.py`。添加一个 `total_usd` 为负数的第四个测试用例。确认验证器用 `minimum` 约束路径拒绝它。

2. 扩展验证器以支持带判别器的 `oneOf`。常见情况：`line_item` 要么是产品要么是服务，按 `kind` 标记。严格模式在这里有微妙规则；检查 OpenAI 的结构化输出指南。

3. 将同一 Invoice schema 写为 Pydantic BaseModel，并将 `model_json_schema()` 输出与你手写的 schema 进行比较。找出 Pydantic 默认设置的一个字段而手写版本遗漏的。

4. 测量拒绝率。构造十个不可提取的输入（一首歌词、数学证明、空邮件）并用严格模式对真实提供商运行。统计拒绝 vs 幻觉输出。这是你拒绝感知重试的基础事实。

5. 从头到尾阅读 OpenAI 的结构化输出指南。找出它在严格模式中明确禁止的一个构造，而普通 JSON Schema 允许。然后设计一个非必要地使用禁止构造的 schema，并将其重构为严格兼容。

## 关键术语

| 术语 | 人们通常说 | 实际含义 |
|------|-----------|----------|
| JSON Schema 2020-12 | "Schema 规范" | 每个现代提供商使用的 IETF 草案 schema 方言 |
| Strict mode（严格模式） | "保证 schema" | 通过约束解码强制执行 schema 的 OpenAI 标志 |
| Constrained decoding（约束解码） | "Logit 屏蔽" | 解码时强制执行，屏蔽无效下一 token |
| Refusal（拒绝） | "模型拒绝" | 输入不符合 schema 时的类型化结果 |
| Parse error（解析错误） | "无效 JSON" | 输出未解析为 JSON；严格模式下不可能 |
| Schema violation（Schema 违规） | "形状错误" | 可解析但违反类型 / required / enum / 范围 |
| `additionalProperties: false` | "不允许额外项" | 禁止未知字段；OpenAI 严格模式必需 |
| Pydantic BaseModel | "类型化输出" | 发出和验证 JSON Schema 的 Python 类 |
| Zod schema | "TypeScript 输出类型" | 用于提供商输出验证的 TS 运行时 schema |
| Grammar enforcement（语法强制） | "开源约束解码" | 基于 FSM 的 logit 屏蔽，如 outlines / guidance |

## 延伸阅读

- [OpenAI — 结构化输出](https://platform.openai.com/docs/guides/structured-outputs) — 严格模式、拒绝和 schema 要求
- [OpenAI — 引入结构化输出](https://openai.com/index/introducing-structured-outputs-in-the-api/) — 2024 年 8 月发布帖，解释解码保证
- [Pydantic AI — 输出](https://ai.pydantic.dev/output/) — 序列化为每个提供商的类型化 output_type 绑定
- [JSON Schema — 2020-12 发布说明](https://json-schema.org/draft/2020-12/release-notes) — 规范规范
- [Microsoft — Azure OpenAI 中的结构化输出](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/structured-outputs) — 企业部署说明和严格模式注意事项