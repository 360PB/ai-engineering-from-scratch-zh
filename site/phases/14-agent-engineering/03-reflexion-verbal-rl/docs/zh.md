# Reflexion：语言强化学习

> 基于梯度的 RL 需要数千次试验和 GPU 集群来修复一个失败模式。Reflexion（Shinn 等人，NeurIPS 2023）用自然语言做到这一点：每次失败试验后，Agent 写一段反思，存储在情景记忆中，并在下一次试验中以该记忆为条件。这是从 Letta 的睡眠时间计算、Claude Code 的 CLAUDE.md 学习，以及 pro-workflow 的 learn-rule 的模式。

**类型：** 构建
**语言：** Python（标准库）
**前置要求：** Phase 14 · 01（Agent 循环）、Phase 14 · 02（ReWOO）
**时间：** 约 60 分钟

## 学习目标

- 命名 Reflexion 的三个组件（Actor、Evaluator、Self-Reflector）和情景记忆的作用。
- 实现带二元评估器、反思缓冲区和新鲜重新尝试的标准库 Reflexion 循环。
- 为给定任务选择标量、启发式和自评估反馈源。
- 解释为什么语言强化能捕获梯度 RL 需要数千次试验才能修复的错误。

## 问题

Agent 任务失败。在标准 RL 中你会运行数千次更多试验，计算梯度，更新权重。昂贵、缓慢，大多数生产 Agent 没有每个失败的训练预算。

Reflexion（Shinn 等人，arXiv:2303.11366）问了一个不同的问题：如果 Agent 只是思考为什么失败并用那个思考重新尝试呢？无需权重更新。无需梯度。只有自然语言存储在试验之间。

结果：在 ALFWorld 上它击败 ReAct 和其他未微调基线。在 HotpotQA 上它比 ReAct 改进。在代码生成（HumanEval/MBPP）上它在当时创下最先进水平。全部无需一次梯度步骤。

## 概念

### 三个组件

```
Actor         : 生成轨迹（ReAct 风格循环）
Evaluator     : 评分轨迹——二元、启发式或自评估
Self-Reflector: 写一段关于失败的自然语言反思
```

加一个数据结构：

```
Episodic memory: 先前反思列表，在下一次试验的提示中前置
```

一次试验运行 Actor。Evaluator 评分。如果分数低，Self-Reflector 产生一段反思（"我选错了工具，因为我把问题误读为问 X 而实际上是问 Y"）。反思进入情景记忆。下一次试验全新开始但看到反思。

### 三种评估器类型

1. **标量** — 外部二元信号。ALFWorld 成功或失败。HumanEval 测试通过或失败。最简单，信号最强。
2. **启发式** — 预定义失败签名。"如果 Agent 连续产生相同动作两次，标记为卡住。""如果轨迹超过 50 步，标记为低效。"
3. **自评估** — LLM 给自己的轨迹评分。当没有真实标签时需要。信号较弱；与工具接地验证（第 05 课——CRITIC）配对效果良好。

2026 年默认是混合：可用时用标量，没有时用自评估，作为安全护栏用启发式。

### 为什么这能泛化

Reflexion 与其说是一个新算法，不如说是一个命名模式。几乎每个生产"自愈"Agent 都运行某种变体：

- Letta 的睡眠时间计算（第 08 课）：一个单独 Agent 反思过去的对话并写入内存块。
- Claude Code 的 `CLAUDE.md` / "保存记忆"模式：将反思捕获为学习内容，在未来会话中前置。
- pro-workflow 的 `/learn-rule` 命令：将纠正捕获为显式规则。
- LangGraph 的反思节点：一个对输出评分的节点，并在需要时路由到细化。

都来自同一洞察：自然语言是足够丰富的媒介，可以在运行之间携带"我从失败中学到了什么"。

### 何时有效何时无效

Reflexion 在以下情况下有效：

- 有清晰的失败信号（测试失败、工具错误、错误答案）。
- 任务类可重现（可以再次提出相同类型的问题）。
- 反思有空间改进轨迹（足够的行动预算）。

Reflexion 在以下情况下无效：

- Agent 第一次就成功了。
- 失败是外部的（网络宕机、工具损坏）——反思"网络宕机了"对未来运行没有帮助。
- 反思变成迷信——存储关于一次性不稳定运行的说法。

2026 年陷阱：记忆衰减。反思积累；有些已过时或错误；随着情景缓冲增长，重新运行变慢。缓解：定期压缩（第 06 课）、反思 TTL，或单独的睡眠时间清理 Agent（Letta）。

## 构建它

`code/main.py` 在一个玩具谜题上实现 Reflexion：产生一个总和为目标值的 3 元素列表。Actor 发出候选列表；Evaluator 检查总和；Self-Reflector 写一行关于什么出错了。反思进入情景记忆供下一次试验。

组件：

- `Actor` — 一个脚本策略，在看到反思时改进。
- `Evaluator.binary()` — 目标总和的通过/失败。
- `SelfReflector` — 生成一行关于失败的诊断。
- `EpisodicMemory` — 带 TTL 语义的有限列表。

运行它：

```
python3 code/main.py
```

追踪显示三次试验。第一次失败，存储一段反思，第二次看到反思并改进但仍失败，第三次成功。对比基线运行（无反思）——它在第一次试验的答案上卡住。

## 使用它

LangGraph 将反思作为节点模式发货。Claude Code 的 `/memory` 命令和 pro-workflow 的 `/learn-rule` 将情景缓冲外部化为 markdown 文件。Letta 的睡眠时间计算在 downtime 运行 Self-Reflector，以便主 Agent 保持延迟受限。OpenAI Agents SDK 不直接发货 Reflexion；你用自定义 Guardrail 构建它，评分拒绝轨迹，和跨运行持久化的内存 `Session`。

## 发布它

`outputs/skill-reflexion-buffer.md` 创建和维护带反思捕获、TTL 和去重的情景缓冲区。给定一个任务类和失败，它发出一段实际上帮助下一次试验的反思（不是泛泛的"更加小心"）。

## 练习

1. 从二元切换到标量评估器，返回距离度量（离目标多远）。它收敛得更快吗？

2. 给反思添加 10 次试验的 TTL。在那之后更旧的反思有帮助还是有害？

3. 实现启发式评估器：如果相同动作重复则将试验标记为卡住。这如何与 Self-Reflector 交互？

4. 用对抗性 Actor 运行 Reflexion，忽略反思。什么最小的反思提示工程强制 Actor 注意它们？

5. 阅读 Reflexion 论文第 4 节关于 AlfWorld。用概念重现 130% 成功率改进：与 vanilla ReAct 相比关键差异是什么？

## 关键术语

| 术语 | 人们通常说 | 实际含义 |
|------|-----------|----------|
| Reflexion | "自我纠正" | Shinn 等人 2023 年——Actor、Evaluator、Self-Reflector 加情景记忆 |
| Verbal reinforcement（语言强化） | "无梯度学习" | 自然语言反思，前置到下一次试验的提示 |
| Episodic memory（情景记忆） | "每任务反思" | 一个任务类的先前反思的有界缓冲区 |
| Scalar evaluator（标量评估器） | "二元成功信号" | 来自真实标签的通过/失败或数值分数 |
| Heuristic evaluator（启发式评估器） | "基于模式的检测器" | 预定义失败签名（如卡住循环、步数过多） |
| Self-evaluator（自评估器） | "LLM 作为自己轨迹的裁判" | 没有真实标签时的较低信号备选——与工具接地验证配对 |
| Memory rot（记忆衰减） | "过时反思" | 情景缓冲充满过时条目；用压缩/TTL 修复 |
| Sleep-time reflection（睡眠时间反思） | "异步自我反思" | 在热路径外运行 Self-Reflector，使主 Agent 保持快速 |

## 延伸阅读

- [Shinn 等人，Reflexion：带语言强化学习的语言 Agent（arXiv:2303.11366）](https://arxiv.org/abs/2303.11366) — 规范论文
- [Letta，睡眠时间计算](https://www.letta.com/blog/sleep-time-compute) — 生产中的异步反思
- [Anthropic，为 AI Agent 有效上下文工程](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) — 将情景缓冲作为上下文的一部分管理
- [LangGraph 概述](https://docs.langchain.com/oss/python/langgraph/overview) — 反思节点模式