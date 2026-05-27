# Self-Refine and CRITIC：迭代式输出改进

> Self-Refine（Madaan et al., 2023）让同一个 LLM 扮演三个角色——生成、反馈、改进——形成循环。在 7 个任务上平均提升 +20 分。CRITIC（Gou et al., 2023）通过外部工具进行验证来强化反馈步骤。2026 年这一模式成为所有框架的默认实现：Anthropic 称为"评估器-优化器"（evaluator-optimizer），OpenAI Agents SDK 称为输出护栏（guardrail loop）。

**类型：** 动手实现
**语言：** Python（标准库）
**前置要求：** Phase 14 · 01（Agent 循环）、Phase 14 · 03（Reflexion）
**时间：** 约 60 分钟

## 学习目标

- 陈述 Self-Refine 的三条核心提示（生成、反馈、改进）并解释为什么历史记录对改进提示至关重要。
- 解释 CRITIC 的关键洞察：LLM 在没有外部基础的情况下进行自我验证是不可靠的。
- 用标准库实现一个带历史的 Self-Refine 循环，可选带外部验证器。
- 将这一模式映射到 Anthropic 的"评估器-优化器"工作流和 OpenAI Agents SDK 的输出护栏。

## 问题背景

Agent 产生的答案几乎正确。可能一行代码有语法错误。可能摘要太长了。可能计划遗漏了边缘情况。你想要的是：Agent 自我批评输出，然后修复它。

Self-Refine 证明这可以用单一模型实现，无需训练数据，无需强化学习。但有一个问题：LLM 在自我验证硬性事实方面表现很差。CRITIC 给出了解决方案——通过外部工具（搜索、代码解释器、计算器、测试运行器）执行验证步骤。

这两篇论文共同定义了 2026 年迭代改进的默认模式：生成，通过外部工具验证（尽可能），改进，当验证器通过时停止。

## 核心概念

### Self-Refine（Madaan et al., NeurIPS 2023）

一个 LLM，三个角色：

```
generate(task)            -> output_0
feedback(task, output_0)  -> critique_0
refine(task, output_0, critique_0, history) -> output_1
feedback(task, output_1)  -> critique_1
refine(task, output_1, critique_1, history) -> output_2
...
当 feedback 说"没有问题"或预算耗尽时停止。
```

关键细节：`refine` 能看到完整的历史——所有之前的输出和批评——所以它不会重复之前的错误。论文做了消融实验：去掉历史记录，质量会急剧下降。

核心数据：在 7 个任务（数学、代码、缩写词、对话）上平均提升 +20 分，包括 GPT-4。无训练，无外部工具，单一模型。

### CRITIC（Gou et al., arXiv:2305.11738, v4 Feb 2024）

Self-Refine 的弱点：反馈步骤是 LLM 给自己打分。对于事实性声明这是不可靠的（幻觉往往对它自己产生的内容看起来很有说服力）。CRITIC 将 `feedback(task, output)` 替换为 `verify(task, output, tools)`，其中 `tools` 包括：

- 搜索引擎（用于事实性声明）。
- 代码解释器（用于代码正确性）。
- 计算器（用于算术运算）。
- 领域特定的验证器（单元测试、类型检查器、linter）。

验证器产生基于工具结果的、结构化的批评。然后改进器根据此批评进行调整。

核心数据：CRITIC 在事实性任务上优于 Self-Refine，因为批评是有基础的。在没有外部验证器的任务（创意写作、格式化）上，CRITIC 退化为 Self-Refine。

### 停止条件

两种常见形式：

1. **验证器通过。** 外部测试返回成功。有外部工具时首选（单元测试、类型检查器、护栏断言）。
2. **没有反馈。** 模型说"输出没问题"。更便宜但不可靠；需要配合最大迭代次数上限。

2026 年的默认做法：结合两者。"当验证器通过或模型说没问题，且迭代次数 >= 2 或迭代次数 >= 最大迭代次数时停止。"

### 评估器-优化器（Anthropic, 2024）

Anthropic 2024 年 12 月的文章将此列为五种工作流模式之一。两个角色：

- 评估器：对输出打分并产生批评。
- 优化器：根据批评修改输出。

循环直到评估器通过。这就是 Anthropic 版本的 Self-Refine/CRITIC。Anthropic 增加的关键工程细节：评估器和优化器的提示应该显著不同，这样模型不会只是橡皮图章式地通过。

### OpenAI Agents SDK 输出护栏

OpenAI Agents SDK 将此模式实现为"输出护栏"。护栏是一个在 Agent 产生最终输出后运行的验证器。如果护栏触发（抛出 `OutputGuardrailTripwireTriggered`），输出被拒绝，Agent 可以重试。护栏可以调用工具（CRITIC 风格）或作为纯函数（Self-Refine 风格）。

### 2026 年陷阱

- **橡皮图章循环。** 同一模型用相同风格的提示做生成和批评，会趋于"看起来不错"。使用结构不同的提示，或用更小更便宜的模型做批评。
- **过度改进。** 每次改进都增加延迟和 token。预算 1-3 次；之后升级到人工审核。
- **在trivial任务上用 CRITIC。** 如果没有外部验证器，CRITIC 退化为 Self-Refine；不要为stub验证器付出延迟代价。

## 动手实现

`code/main.py` 实现了一个自选任务的 Self-Refine 和 CRITIC：给定主题产生一个短bullet列表。验证器检查格式（3条，每条少于 60 字符）。CRITIC 添加了一个外部"事实验证器"，对已知幻觉进行惩罚。

组件：

- `generate`——脚本化生成器。
- `feedback`——LLM 式自我批评。
- `verify_external`——CRITIC 式有基础验证器。
- `refine`——根据历史重写输出。
- 停止条件——验证器通过或最多 4 次迭代。

运行：

```bash
python3 code/main.py
```

对比 Self-Refine 和 CRITIC 的运行结果。CRITIC 抓住了一个 Self-Refine 遗漏的事实错误，因为外部验证器有自我批评所没有的基础。

## 用现成库

Anthropic 的 evaluator-optimizer 就是这一模式的 Claude 友好版本。OpenAI Agents SDK 的输出护栏是 CRITIC 形状的（护栏可以调用工具）。LangGraph 提供了类似 Self-Refine 的反思节点。Google Gemini 2.5 Computer Use 在每一步添加了一个安全评估器，是 CRITIC 的变体：每个动作在提交前都要验证。

## 产出

`outputs/skill-refine-loop.md` 根据任务形状、验证器可用性和迭代预算配置一个评估器-优化器循环。输出生成器、评估器/验证器和优化器的提示，加上停止策略。

## 练习

1. 用 max_iterations=1 运行玩具任务。CRITIC 还有帮助吗？
2. 用一个有噪声的验证器替换外部验证器（随机 30% 误报）。循环会怎么做？这是 2026 年大多数护栏栈的现实情况。
3. 实现一个"不同模型的生成器-批评者"变体：大模型生成，小模型批评。它能打败同模型版本吗？
4. 阅读 CRITIC 第三节（arXiv:2305.11738 v4）。列出三个验证工具类别并各举一个例子。
5. 将 OpenAI Agents SDK 的 `output_guardrails` 映射到 CRITIC 的验证器角色。SDK 做错了什么，做对了什么？

## 术语表

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Self-Refine | "能自我修复的 LLM" | 单一模型中的生成→反馈→改进循环，带历史记录 |
| CRITIC | "工具接地的验证" | 用外部验证器（搜索、代码、计算、测试）替换反馈 |
| Evaluator-Optimizer | "Anthropic 工作流模式" | 两个角色——评估器打分，优化器修改——循环至收敛 |
| Output guardrail | "事后检查" | OpenAI Agents SDK 验证器，在 Agent 产生输出后运行 |
| Verify step | "批评阶段" | 关键决策点：基于外部工具还是自我评分 |
| Refine history | "模型已经尝试过的" | 先前的输出+批评前置到改进提示；去掉它质量会崩溃 |
| Rubber-stamp loop | "自我协议失败" | 同提示的批评返回"看起来不错"；用结构不同的提示修复 |
| Stop condition | "收敛测试" | 验证器通过或无反馈且达到迭代上限；永远不要单条件 |

## 扩展阅读

- [Madaan et al., Self-Refine (arXiv:2303.17651)](https://arxiv.org/abs/2303.17651) — 经典论文
- [Gou et al., CRITIC (arXiv:2305.11738)](https://arxiv.org/abs/2305.11738) — 工具接地验证
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — 评估器-优化器工作流模式
- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/) — 输出护栏作为 CRITIC 形状的验证器