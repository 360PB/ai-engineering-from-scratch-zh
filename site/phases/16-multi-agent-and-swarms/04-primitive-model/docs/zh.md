# 多智能体原语模型

> 2026 年发布的每个多智能体框架——AutoGen、LangGraph、CrewAI、OpenAI Agents SDK、Microsoft Agent Framework——都是四维设计空间中的一个点。四个原语，仅此而已：智能体、交接、共享状态、编排器。本课从零构建它们，在所有四个上运行一个玩具系统，然后映射每个主要框架到相同的轴上，这样你就能在一段话中阅读任何新版本。

**类型：** 学习
**语言：** Python（标准库）
**前置知识：** Phase 14（智能体工程），Phase 16 · 01（为什么要用多智能体）
**时间：** 约60分钟

## 问题

每六个月就有一个新的多智能体框架发布。AutoGen 2023 年。CrewAI 2024 年。LangGraph 和 OpenAI Swarm 2024 年。Google ADK 2025 年 4 月。Microsoft Agent Framework RC 2026 年 2 月。每次发布都声称是"正确的抽象"。

如果你一个一个学，你会筋疲力尽。API 看起来不同。文档对"智能体"是什么说法不一。一个框架称其共享内存为"黑板"，另一个称为"消息池"，第三个称为"StateGraph"。你开始怀疑这个领域只是在 churn。

不是的。在营销之下，四个原语是稳定的。学一次，在一段话中阅读每个新框架。

## 概念

### 四个原语

1. **智能体（Agent）**——系统提示加工具列表。无状态；每次运行从其系统提示和当前消息历史开始。
2. **交接（Handoff）**——从一智能体到另一智能体的结构化控制转移。机械地说，是一个返回新智能体的工具调用，或跟随条件的图边。
3. **共享状态（Shared state）**——多个智能体可以读取（有时写入）的任何数据结构。消息池、黑板、键值存储、向量内存。
4. **编排器（Orchestrator）**——决定谁下一个发言的任何人。选项：显式图（确定性）、LLM 说话者选择器（软）、最后发言者的交接调用（OpenAI Swarm）、队列上调度器（蜂群架构）。

这就是整个设计空间。每个框架在每个轴上选择默认值；其余都是表面语法。

### 每个 2026 框架如何映射到它

| 框架 | 智能体 | 交接 | 共享状态 | 编排器 |
|-----------|-------|---------|--------------|--------------|
| OpenAI Swarm / Agents SDK | `Agent(instructions, tools)` | 工具返回 Agent | 调用方的问题 | LLM 的下一个交接调用 |
| AutoGen v0.4 / AG2 | `ConversableAgent` | GroupChat 上的说话者选择器 | 消息池 | 选择器函数（LLM 或 round-robin） |
| CrewAI | `Agent(role, goal, backstory)` | `Process.Sequential / Hierarchical` | 链接的任务输出 | 管理者 LLM 或静态顺序 |
| LangGraph | 节点函数 | 图边 + 条件 | `StateGraph` reducer | 图，确定性 |
| Microsoft Agent Framework | 智能体 + 编排模式 | 模式特定 | 线程 / 上下文 | 模式特定 |
| Google ADK | 智能体 + A2A 卡片 | A2A 任务 | A2A 产物 | host 决定 |

表面差异看起来很大。在底层：相同的四个旋钮。

### 为什么这很重要

一旦你看到原语，框架比较就变成一个简短的清单：

- 编排器信任 LLM 路由（图）还是将路由固定在代码中（LangGraph）？
- 共享状态是完整历史（GroupChat）还是投影的（StateGraph reducer）？
- 智能体可以修改彼此的提示（CrewAI 管理者）还是只能交接（Swarm）？

这三个问题回答了哪个框架适合给定问题的 80%。你停止寻找"最佳多智能体框架"，开始为你真正关心的轴进行设计。

### 无状态洞察

除共享状态外每个原语都是无状态的。智能体是（提示、工具）的函数。交接是函数调用。编排器是调度器。**系统中唯一有状态的是共享状态。** 这就是所有有趣 bug 所在：内存中毒（Lesson 15）、消息排序、版本控制、写竞争。

隐藏共享状态的框架（Swarm）将问题推给调用方。集中化它的框架（LangGraph checkpoint、AutoGen pool）使其可检查，但将协调成本转移到共享状态实现上。

### 单个原语的解剖

#### 智能体

```
Agent = (system_prompt, tools, model, optional_name)
```

无内存。无状态。两个具有相同系统提示和工具的智能体是可互换的。任何看起来像每个智能体状态的东西实际上都在共享状态或交接协议中。

#### 交接

```
Handoff = (from_agent, to_agent, reason, payload)
```

三种主要实现：

- **函数返回**——工具返回下一个智能体。这是 OpenAI Swarm 模式。智能体在其工具模式中携带路由。
- **图边**——LangGraph。边是声明性的。LLM 产生一个值；条件选择下一个节点。
- **说话者选择**——AutoGen GroupChat。选择器函数（有时本身是 LLM 调用）读取池并选择下一个发言者。

#### 共享状态

```
SharedState = { messages: [], artifacts: {}, context: {} }
```

至少是一个消息列表。通常更多：结构化产物（CrewAI 任务输出）、类型化上下文（LangGraph reducer）、外部内存（MCP、向量 DB）。

两种拓扑：**完整池**（每个智能体看到每条消息）和**投影**（智能体看到角色作用域视图）。完整池简单但扩展差。投影池可扩展但需要前期模式设计。

#### 编排器

```
Orchestrator = ({state, last_speaker}) -> next_agent
```

四种风格：

- **静态**——图在构建时固定（LangGraph 确定性、CrewAI 顺序）。
- **LLM 选择**——LLM 读取池并选择下一个发言者（AutoGen、CrewAI 层级）。
- **交接驱动**——当前智能体通过调用交接工具决定（Swarm）。
- **队列驱动**——工作器从共享队列中拉取；无显式下一个说话者（蜂群架构、Matrix）。

### 框架之间什么变化

一旦原语固定，剩余的设计决策是：

- **内存策略**——临时 vs 持久 checkpointing（LangGraph checkpointer）。
- **安全边界**——谁可以批准交接（人工在环中）。
- **成本核算**——每个智能体 token 预算。
- **可观测性**——跟踪交接、持久化状态以供重放。

都可以在原语之上实现。都不是新的原语。

## 构建

`code/main.py` 用约 150 行标准库 Python 实现了四个原语。没有真实 LLM——每个智能体都是一个脚本化策略，这样焦点保持在协调结构上。

文件导出：

- `Agent`——名称、系统提示、工具、策略函数的 dataclass。
- `Handoff`——返回新智能体的函数。
- `SharedState`——线程安全消息池。
- `Orchestrator`——三种变体：`StaticOrchestrator`、`HandoffOrchestrator`、`LLMSelectorOrchestrator`（模拟）。

演示用所有三种编排器类型运行相同的三个智能体流水线（研究 → 编写 → 审查），并在最后打印消息池。你可以看到输出只在*谁选择下一个*上不同；智能体和共享状态在运行之间是相同的。

运行它：

```
python3 code/main.py
```

预期输出：三个编排器运行，每种模式一个。每种打印最终消息池。交接驱动的运行在研究者提前决定完成时达到更少的智能体——这就是 LLM 路由权衡的小型化。

## 使用

`outputs/skill-primitive-mapper.md` 是一个技能，读取任何多智能体代码库或框架文档并返回四原语映射。在深入阅读文档之前，在新框架发布上运行它以获得一段话的理解。

## 交付

在采用新框架之前，为它编写原语映射。如果你不能，文档不完整或框架在发明第五个原语（罕见——检查是否有你未见过的共享状态风格）。

将映射固定在你的架构文档中。当新团队成员加入时，在 API 文档之前发送映射。当框架版本变化时，diff 映射，不是 changelog。

## 练习

1. 运行 `code/main.py` 三次，使用不同的智能体策略。观察编排器选择如何改变运行的智能体。
2. 实现第四种编排器类型：队列驱动的，智能体轮询共享状态获取工作。什么死锁可能发生，如何检测？
3. 拿 LangGraph quickstart（https://docs.langchain.com/oss/python/langgraph/workflows-agents）并用四个原语重写它。LangGraph 的哪些抽象 1:1 映射，哪些是便利包装？
4. 阅读 OpenAI Swarm cookbook（https://developers.openai.com/cookbook/examples/orchestrating_agents）。识别 Swarm 最方便和最不方便的四个原语。
5. 在此表中找一个完全隐藏共享状态的框架。解释当智能体需要跨交接协调而无法重读历史时什么会崩溃。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| 智能体 | "带工具的 LLM" | `(system_prompt, tools, model)` 三元组。无状态。 |
| 交接 | "控制转移" | 命名下一个智能体并可选有效载荷的结构化调用。三种实现：函数返回、图边、说话者选择。 |
| 共享状态 | "内存 / 上下文" | 多智能体系统中唯一有状态的部分。消息池或黑板。 |
| 编排器 | "协调器" | 决定谁运行下一个的任何人。静态图、LLM 选择器、交接驱动或队列驱动。 |
| 原语 | "抽象" | 每个框架参数化的四个轴之一。不是框架特性。 |
| 消息池 | "共享聊天历史" | 完整历史共享状态。易于推理，扩展差。 |
| 投影状态 | "作用域视图" | 共享状态的角色特定视图。可扩展，需要模式设计。 |
| 说话者选择 | "下一个发言者" | 编排器模式，其中函数（通常是 LLM）从一组中选择下一个智能体。 |

## 延伸阅读

- [OpenAI cookbook：编排智能体——交接和例程](https://developers.openai.com/cookbook/examples/orchestrating_agents)——交接驱动编排的最清晰阐述
- [AutoGen 稳定文档](https://microsoft.github.io/autogen/stable/)——GroupChat + 说话者选择是 LLM 选择编排的参考
- [LangGraph 工作流和智能体](https://docs.langchain.com/oss/python/langgraph/workflows-agents)——图边编排和基于 reducer 的共享状态
- [CrewAI 介绍](https://docs.crewai.com/en/introduction)——角色-目标-背景故事智能体、顺序/层级流程
- [AG2（社区 AutoGen 延续）](https://github.com/ag2ai/ag2)——微软将 v0.4 放入维护后，v0.2 线的活跃上游
