# 工具使用与函数调用

> Toolformer（Schick et al., 2023）开创了自监督工具标注。Berkeley Function Calling Leaderboard V4（Patil et al., 2025）设定了 2026 年的标准：40% Agent 场景，30% 多轮对话，10% 实时，10% 非实时，10% 幻觉检测。单轮调用已解决。记忆、动态决策和长程工具链尚未解决。

**类型：** 动手实现
**语言：** Python（标准库）
**前置要求：** Phase 14 · 01（Agent 循环）、Phase 13 · 01（函数调用深度解析）
**时间：** 约 60 分钟

## 学习目标

- 解释 Toolformer 的自监督训练信号：只有在执行能减少下一个 token 损失时才保留工具标注。
- 说出 BFCL V4 的五个评估类别以及每个类别衡量的内容。
- 实现一个带模式验证、参数强制转换和执行沙箱的标准库工具注册表。
- 诊断 2026 年三个未解决的开放问题：长程工具链、动态决策和记忆。

## 问题背景

早期工具使用问的是：模型能预测正确的函数调用吗？现代工具使用问的是：模型能在 40 步中链式调用工具、带记忆、在部分可观测性下、在工具失败后恢复、不产生不存在工具的幻觉吗？

Toolformer 建立了基线：模型可以通过自监督学习何时调用工具。BFCL V4 定义了 2026 年的评估目标。两者之间的差距是生产环境 Agent 实际面对的空间。

## 核心概念

### Toolformer（Schick et al., NeurIPS 2023）

核心理念：让模型在自己的预训练语料上标注候选 API 调用。对每个候选执行它。只有当工具结果能减少下一个 token 的损失时才保留标注。在过滤后的语料上进行微调。

覆盖的工具：计算器、问答系统、搜索引擎、翻译器、日历。自监督信号纯粹基于工具是否有助于预测文本——无人工标签。

规模结果：工具使用在规模上涌现。较小的模型受工具标注拖累；较大的模型从中受益。这就是为什么 2026 年前沿模型内置了强大的工具使用能力，而大多数 7B 模型需要显式的工具使用微调才能可靠。

### Berkeley Function Calling Leaderboard V4（Patil et al., ICML 2025）

BFCL 是 2026 年的事实评估标准。V4 组成：

- **Agent 场景（40%）** — 完整 Agent 轨迹：记忆、多轮、动态决策。
- **多轮对话（30%）** — 带工具链的交互式对话。
- **实时（10%）** — 用户提交的实时提示（更难的分发）。
- **非实时（10%）** — 合成测试用例。
- **幻觉检测（10%）** — 检测何时不应调用工具。

V3 引入了基于状态的评估：工具序列后，检查 API 的实际状态（例如"文件是否已创建？"），而不是匹配工具调用的 AST。V4 添加了 Web 搜索、记忆和格式敏感度类别。

2026 年关键发现：单轮函数调用已接近解决。失败集中在记忆（跨轮携带上下文）、动态决策（根据先前结果选择工具）、长程链（20+ 步后漂移）和幻觉检测（当没有工具适用时拒绝调用）。

### 工具模式

每个提供商都有模式。它们在细节上有所不同，但共享相同的结构：

```json
{
  "name": "string",
  "description": "string（工具做什么，何时使用）",
  "input_schema": JSON Schema（properties, required, types, enums）
}
```

Anthropic 直接使用 `input_schema`。OpenAI 使用 `function.parameters`。两者都接受 JSON Schema。描述是至关重要的——模型通过它们来选择正确的工具。糟糕的工具描述是选错工具失败的首要根源。

### 参数验证

不要信任任何工具调用。验证：

1. **类型强制转换。** 模型可能返回字符串 "5"，而模式说是 int。如果明确则强制转换；否则拒绝。
2. **枚举验证。** 如果模式说 `status in {"open", "closed"}` 而模型发出 `"in_progress"`，用描述性错误拒绝。
3. **必需字段。** 缺少必需字段 → 立即向模型返回错误观察，而不是崩溃。
4. **格式验证。** 日期、邮箱、URL——用具体解析器而非正则表达式验证。

每个验证失败都应返回结构化观察，让模型能用正确格式重试。

### 并行工具调用

现代提供商支持在一个 assistant 回合中并行调用工具。循环：

1. 模型发出 3 个带不同 `tool_use_id` 的工具调用。
2. 运行时执行它们（如果独立则并行）。
3. 每个结果作为通过 `tool_use_id` 关联的 `tool_result` 块返回。

工程规则：将关联 ID 视为关键。交换它们会导致错误的工具匹配到错误的结果。

### 沙箱

工具执行是沙箱边界。见课程 09 的详细说明。简而言之：每个工具应指定读/写表面、网络访问、超时和内存上限。通用的 `run_shell(cmd)` 是红旗；具体的 `git_status()` 更安全。

## 动手实现

`code/main.py` 实现了一个生产形态的工具注册表：

- JSON Schema 子集验证器（仅用标准库）。
- 工具注册：描述、输入模式、超时和执行器。
- 参数强制转换和枚举验证。
- 带关联 ID 的并行工具分发。
- 结构化字符串形式的错误观察。

运行：

```bash
python3 code/main.py
```

追踪显示一个迷你 Agent 在一轮中调用三个工具，其中一个故意格式错误的调用被带有描述性错误的拒绝，模型可以据此操作。

## 用现成库

每个提供商都有自己的工具模式——Anthropic、OpenAI、Gemini、Bedrock。如果需要多提供商，使用翻译层（OpenAI Agents SDK、Vercel AI SDK、LangChain 工具适配器）。BFCL 是参考基准——如果工具使用是产品的核心，在发布前用它运行评估。

## 产出

`outputs/skill-tool-registry.md` 为给定任务领域生成工具目录、模式和注册表。包括描述质量检查（每个工具的描述是否告诉模型何时使用它？）。

## 练习

1. 添加一个"空操作"工具，让模型能明确拒绝使用任何其他工具。在类似 BFCL 的幻觉测试上测量效果。
2. 实现 int-as-string 和 float-as-string 的参数强制转换。强制转换在何时开始掩盖真正的 bug？
3. 添加每工具超时和断路器（连续 3 次失败后拒绝该工具 60 秒）。这如何改变模型的恢复方式？
4. 阅读 BFCL V4 描述。选择一个类别（例如"多轮"），用你的 Agent 运行 10 个示例提示。报告通过率。
5. 将标准库验证器移植到 Pydantic 或 Zod。Pydantic/Zod 抓住了玩具实现遗漏了什么？

## 术语表

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Function calling | "工具使用" | 带验证模式的结构化输出工具调用 |
| Toolformer | "自监督工具标注" | Schick 2023——保留结果能减少下一个 token 损失的 tool calls |
| BFCL | "Berkeley Function Calling Leaderboard" | 2026 基准：40% Agent，30% 多轮，10% 实时，10% 非实时，10% 幻觉 |
| Tool schema | "模型的函数签名" | name, description, JSON Schema of arguments |
| tool_use_id | "关联 ID" | 将工具调用与其结果绑定；并行分发时必不可少 |
| Hallucination detection | "知道何时不调用" | V4 类别：当没有工具适用时拒绝调用 |
| Argument coercion | "字符串转 int 修复" | 对可预测的模式不匹配进行窄修复；模糊时拒绝 |
| Sandboxing | "工具执行边界" | 每工具读/写表面、网络、超时、内存上限 |

## 扩展阅读

- [Schick et al., Toolformer (arXiv:2302.04761)](https://arxiv.org/abs/2302.04761) — 自监督工具标注
- [Berkeley Function Calling Leaderboard (V4)](https://gorilla.cs.berkeley.edu/leaderboard.html) — 2026 评估基准
- [Anthropic, Tool use documentation](https://platform.claude.com/docs/en/agent-sdk/overview) — Claude Agent SDK 中的生产工具模式
- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/) — 函数工具类型和护栏