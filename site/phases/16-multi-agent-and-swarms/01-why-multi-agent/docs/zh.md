# 为什么要用多智能体？

> 一个智能体撞墙了。聪明的做法不是做一个更大的智能体——而是更多智能体。

**类型：** 学习
**语言：** TypeScript
**前置知识：** Phase 14（智能体工程）
**时间：** 约60分钟

## 学习目标

- 识别单智能体的天花板（上下文溢出、专业能力混杂、串行瓶颈），并解释何时应该拆分为多个智能体
- 比较编排模式（流水线、并行扇出、主管、层级），并根据任务结构选择正确的模式
- 设计一个具有清晰角色边界、共享状态和通信契约的多智能体系统
- 分析多智能体复杂度（延迟、成本、调试难度）与单智能体简单性之间的权衡

## 问题

你在 Phase 14 构建了一个单智能体。它能工作。它可以读取文件、运行命令、调用 API，并对结果进行推理。然后你把它指向一个真实代码库：200 个文件、三种语言、依赖基础设施的测试，以及一个先研究外部 API 再写代码的需求。

智能体卡住了。不是因为 LLM 太笨，而是因为任务超出了单个智能体循环能处理的范围。上下文窗口被文件内容填满。智能体忘记了 40 次工具调用前读取的内容。它试图同时充当研究员、程序员和审查员，结果三个都做得很差。

这就是单智能体的天花板。每当任务需要以下条件时，你就会遇到它：

- **超出单个窗口的上下文**——读取 50 个文件就超过了 200k tokens
- **不同阶段需要不同专业能力**——研究需要的提示工程与代码生成不同
- **可以并行完成的工作**——为什么要依次读取三个文件而不是同时读取？

## 概念

### 单智能体的天花板

单个智能体是一个循环、一个上下文窗口、一个系统提示。想象一下：

```
┌─────────────────────────────────────────┐
│            单个智能体                    │
│                                         │
│  ┌───────────────────────────────────┐  │
│  │         上下文窗口                 │  │
│  │                                   │  │
│  │  研究笔记                           │  │
│  │  + 代码文件                       │  │
│  │  + 测试输出                       │  │
│  │  + 审查反馈                       │  │
│  │  + API 文档                       │  │
│  │  + ...                            │  │
│  │                                   │  │
│  │  ██████████████████████ 已满 ███  │  │
│  └───────────────────────────────────┘  │
│                                         │
│  一个系统提示试图覆盖                     │
│  研究 + 编码 + 审查 + 测试               │
│                                         │
│  结果：每件事都马马虎虎                  │
└─────────────────────────────────────────┘
```

三件事会崩溃：

1. **上下文饱和**——工具结果堆积。到第 30 轮时，智能体已经消耗了 150k tokens 的文件内容、命令输出和之前的推理。第 5 轮的关键细节丢失了。

2. **角色混淆**——系统提示说"你是一个研究员、程序员、审查员和测试员"，产生的是一个半研究、半编码、从未完成审查的智能体。

3. **串行瓶颈**——智能体依次读取文件 A，然后 B，然后 C。三个串行 LLM 调用。三个串行工具执行。没有并行。

### 多智能体解决方案

拆分工作。给每个智能体一个任务、一个上下文窗口和一个针对该任务调优的系统提示：

```
┌──────────────────────────────────────────────────────────┐
│                    编排器                                 │
│                                                          │
│  "构建一个用户管理的 REST API"                           │
│                                                          │
│         ┌──────────┬──────────┬──────────┐               │
│         │          │          │          │               │
│         ▼          ▼          ▼          ▼               │
│   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│   │  研究员  │ │  编码员  │ │  审查员  │ │  测试员  │  │
│   │          │ │          │ │          │ │          │  │
│   │  读取    │ │  根据    │ │  检查    │ │  运行    │  │
│   │  文档，  │ │  研究    │ │  代码    │ │  测试，  │  │
│   │  查找    │ │  + 规范  │ │  质量，  │ │  报告    │  │
│   │  模式    │ │  编写    │ │  发现    │ │  结果    │  │
│   │          │ │  代码    │ │  缺陷    │ │          │  │
│   └─────┬────┘ └────┬─────┘ └────┬────┘ └────┬────┘  │
│         │           │            │             │         │
│         └───────────┴────────────┴─────────────┘         │
│                          │                               │
│                     合并结果                            │
└──────────────────────────────────────────────────────────┘
```

每个智能体有：
- 一个专注的系统提示（"你是一个代码审查员。你的唯一任务是发现缺陷。"）
- 自己的上下文窗口（不受其他智能体工作污染）
- 清晰的输入/输出契约（接收研究笔记，输出代码）

### 这样做过的真实系统

**Claude Code 子智能体**——当 Claude Code 用 `Task` 生成子智能体时，它创建一个带有作用域任务的子智能体。父智能体保持上下文干净。子智能体执行专注的工作并返回摘要。

**Devin**——运行一个规划智能体、一个编码智能体和一个浏览器智能体。规划器将工作分解为步骤。编码器编写代码。浏览器研究文档。每个都有独立的上下文。

**多智能体编程团队（SWE-bench）**——SWE-bench 上表现最好的系统使用一个研究员阅读代码库、一个规划器设计修复方案、一个编码器实现修复。单智能体系统得分较低。

**ChatGPT Deep Research**——并行生成多个搜索智能体，每个探索不同角度，然后综合结果。

### 谱系

多智能体不是二元的。它是一个谱系：

```
简单 ──────────────────────────────────────────── 复杂

 单个       子智能体      流水线        团队         蜂群
 智能体

 ┌───┐    ┌───┐       ┌───┬───┐   ┌───┬───┐   ┌─┐┌─┐┌─┐
 │ A │    │ A │        │ A │ B │   │ A │ B │   │ ││ ││ │
 └───┘    └─┬─┘       └───┴─┬─┘   └─┬─┴─┬─┘   └┬┘└┬┘└┬┘
              │               │        │   │      ┌┴──┴──┴┐
            ┌─┴─┐         ┌───┴───┐  ┌─┴───┴─┐   │共享    │
            │ a │         │ C │ D │ │  msg   │   │ 状态   │
            └───┘         └───┴───┘  │  总线  │   └────────┘
                                      └────────┘
 1个循环    父+子任务    按阶段进行   │        │   N个对等体，
 1个上下文                             └────────┘   涌现行为
                                     显式角色
```

**单智能体**——一个循环，一个提示。适合简单任务。

**子智能体**——父智能体生成子任务执行专注的子任务。父智能体维护计划。子智能体报告返回。Claude Code 就是这样做的。

**流水线**——智能体按顺序运行。智能体 A 的输出成为智能体 B 的输入。适合阶段性工作流：研究 -> 编码 -> 审查 -> 测试。

**团队**——智能体通过共享消息总线并行运行。每个智能体有角色。编排器协调。适合需要同时使用不同技能的场景。

**蜂群**——许多相同或相近的智能体，共享状态。没有固定编排器。智能体从队列中获取工作。适合高吞吐量并行任务。

### 四种多智能体模式

#### 模式 1：流水线

```
输入 ──▶ 智能体 A ──▶ 智能体 B ──▶ 智能体 C ──▶ 输出
          （研究）    （编码）      （审查）
```

每个智能体转换数据并向前传递。简单易推理。一个阶段的失败会阻塞其余阶段。

#### 模式 2：扇出 / 扇入

```
                ┌──▶ 智能体 A ──┐
                │               │
输入 ──▶ 拆分 ├──▶ 智能体 B ──├──▶ 合并 ──▶ 输出
                │               │
                └──▶ 智能体 C ──┘
```

将工作拆分到并行智能体，然后合并结果。适合可分解为独立子任务的任务。

#### 模式 3：编排器-工作器

```
                    ┌──────────┐
                    │ 编排器   │
                    └──┬───┬───┘
                  任务 │   │ 任务
                 ┌─────┘   └─────┐
                 ▼               ▼
           ┌──────────┐   ┌──────────┐
           │ 工作器 A │   │ 工作器 B │
           └──────────┘   └──────────┘
```

智能编排器决定做什么，委托给工作器，并综合结果。编排器本身是一个具有工具的智能体，用于生成工作器。

#### 模式 4：对等蜂群

```
         ┌───┐ ◄──── 消息 ────▶ ┌───┐
         │ A │                  │ B │
         └─┬─┘                  └─┬─┘
           │                      │
      消息 │    ┌───────────┐     │ 消息
           └───▶│  共享     │◄────┘
                │  状态     │
           ┌───▶│  / 队列   │◄────┐
           │    └───────────┘      │
      消息 │                      │ 消息
         ┌─┴─┐                  ┌─┴─┐
         │ C │ ◄──── 消息 ────▶ │ D │
         └───┘                  └───┘
```

没有中央编排器。智能体点对点通信。决策从交互中涌现。更难调试，但可以扩展到许多智能体。

### 何时不用多智能体

多智能体增加了复杂性。每个智能体之间的消息都是潜在故障点。调试从"阅读一段对话"变成"追踪五个智能体之间的消息"。

**保持单智能体当：**
- 任务适合一个上下文窗口（工作数据在 ~100k tokens 以下）
- 不同阶段不需要不同的系统提示
- 串行执行速度足够快
- 任务足够简单，拆分它带来的开销超过价值

**复杂度成本：**
- 每个智能体边界都是一个有损压缩步骤：智能体 A 的完整上下文被总结成一条消息传给智能体 B
- 协调逻辑（谁做什么、何时做、按什么顺序）本身就是 bug 来源
- 延迟增加：N 个智能体意味着至少 N 次串行 LLM 调用，如果需要来回通信则更多
- 成本成倍增加：每个智能体独立消耗 tokens

经验法则：如果任务需要少于 20 次工具调用且适合 100k tokens，保持单智能体。

## 构建

### 步骤 1：过载的单智能体

这里是一个试图做所有事情的单个智能体。它有一个庞大的系统提示和一个上下文窗口，容纳研究、代码和审查：

```typescript
type AgentResult = {
  content: string;
  tokensUsed: number;
  toolCalls: number;
};

async function singleAgentApproach(task: string): Promise<AgentResult> {
  const systemPrompt = `You are a full-stack developer. You must:
1. Research the requirements
2. Write the code
3. Review the code for bugs
4. Write tests
Do ALL of these in a single conversation.`;

  const contextWindow: string[] = [];
  let totalTokens = 0;
  let totalToolCalls = 0;

  const research = await fakeLLMCall(systemPrompt, `Research: ${task}`);
  contextWindow.push(research.output);
  totalTokens += research.tokens;
  totalToolCalls += research.calls;

  const code = await fakeLLMCall(
    systemPrompt,
    `Given this research:\n${contextWindow.join("\n")}\n\nNow write code for: ${task}`
  );
  contextWindow.push(code.output);
  totalTokens += code.tokens;
  totalToolCalls += code.calls;

  const review = await fakeLLMCall(
    systemPrompt,
    `Given all previous context:\n${contextWindow.join("\n")}\n\nReview the code.`
  );
  contextWindow.push(review.output);
  totalTokens += review.tokens;
  totalToolCalls += review.calls;

  return {
    content: contextWindow.join("\n---\n"),
    tokensUsed: totalTokens,
    toolCalls: totalToolCalls,
  };
}
```

这种方法的问题：
- 上下文窗口随着每个阶段增长。到审查阶段，它包含研究笔记 AND 代码 AND 之前的推理。
- 系统提示是通用的。它无法针对每个阶段调优。
- 没有任何东西是并行运行的。

### 步骤 2：专业智能体

现在拆分它。每个智能体一个任务：

```typescript
type SpecialistAgent = {
  name: string;
  systemPrompt: string;
  run: (input: string) => Promise<AgentResult>;
};

function createSpecialist(name: string, systemPrompt: string): SpecialistAgent {
  return {
    name,
    systemPrompt,
    run: async (input: string) => {
      const result = await fakeLLMCall(systemPrompt, input);
      return {
        content: result.output,
        tokensUsed: result.tokens,
        toolCalls: result.calls,
      };
    },
  };
}

const researcher = createSpecialist(
  "researcher",
  "You are a technical researcher. Read documentation, find patterns, and summarize findings. Output only the facts needed for implementation."
);

const coder = createSpecialist(
  "coder",
  "You are a senior TypeScript developer. Given requirements and research notes, write clean, tested code. Nothing else."
);

const reviewer = createSpecialist(
  "reviewer",
  "You are a code reviewer. Find bugs, security issues, and logic errors. Be specific. Cite line numbers."
);
```

每个专业智能体有一个专注的提示。每个智能体获得一个干净的上下文窗口，只包含它需要的输入。

### 步骤 3：通过消息协调

用显式消息传递连接专业智能体：

```typescript
type AgentMessage = {
  from: string;
  to: string;
  content: string;
  timestamp: number;
};

async function multiAgentApproach(task: string): Promise<AgentResult> {
  const messages: AgentMessage[] = [];
  let totalTokens = 0;
  let totalToolCalls = 0;

  const researchResult = await researcher.run(task);
  messages.push({
    from: "researcher",
    to: "coder",
    content: researchResult.content,
    timestamp: Date.now(),
  });
  totalTokens += researchResult.tokensUsed;
  totalToolCalls += researchResult.toolCalls;

  const coderInput = messages
    .filter((m) => m.to === "coder")
    .map((m) => `[From ${m.from}]: ${m.content}`)
    .join("\n");

  const codeResult = await coder.run(coderInput);
  messages.push({
    from: "coder",
    to: "reviewer",
    content: codeResult.content,
    timestamp: Date.now(),
  });
  totalTokens += codeResult.tokensUsed;
  totalToolCalls += codeResult.toolCalls;

  const reviewerInput = messages
    .filter((m) => m.to === "reviewer")
    .map((m) => `[From ${m.from}]: ${m.content}`)
    .join("\n");

  const reviewResult = await reviewer.run(reviewerInput);
  messages.push({
    from: "reviewer",
    to: "orchestrator",
    content: reviewResult.content,
    timestamp: Date.now(),
  });
  totalTokens += reviewResult.tokensUsed;
  totalToolCalls += reviewResult.toolCalls;

  return {
    content: messages.map((m) => `[${m.from} -> ${m.to}]: ${m.content}`).join("\n\n"),
    tokensUsed: totalTokens,
    toolCalls: totalToolCalls,
  };
}
```

每个智能体只接收发给它的消息。没有上下文污染。研究员的 50k tokens 文档阅读永远不会进入审查员的上下文。

### 步骤 4：比较

```typescript
async function compare() {
  const task = "Build a rate limiter middleware for an Express.js API";

  console.log("=== 单智能体 ===");
  const single = await singleAgentApproach(task);
  console.log(`Tokens: ${single.tokensUsed}`);
  console.log(`Tool calls: ${single.toolCalls}`);

  console.log("\n=== 多智能体 ===");
  const multi = await multiAgentApproach(task);
  console.log(`Tokens: ${multi.tokensUsed}`);
  console.log(`Tool calls: ${multi.toolCalls}`);
}
```

多智能体版本使用更多总 tokens（三个智能体，三个独立的 LLM 调用），但每个智能体的上下文保持干净。每个阶段的输出质量提高，因为系统提示是专门化的。

## 使用

本课生成一个可重用的提示，用于决定何时使用多智能体。参见 `outputs/prompt-multi-agent-decision.md`。

## 练习

1. 添加第四个专业智能体：一个"测试员"智能体，接收来自编码员的代码和来自审查员的审查反馈，然后编写测试
2. 修改流水线，使审查员可以将反馈发回编码员进行修改循环（最多 2 轮）
3. 将顺序流水线转换为扇出：并行运行研究员和一个"需求分析员"智能体，然后在传递给编码员之前合并它们的输出

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|----------------------|
| 蜂群（Swarm） | "AI 智能体的蜂巢思维" | 具有共享状态和无固定领导者的对等智能体集合。行为从局部交互中涌现。 |
| 编排器（Orchestrator） | "老板智能体" | 其工具包含生成和管理其他智能体的智能体。它规划并委托但可能不做实际工作。 |
| 协调器（Coordinator） | "交通警察" | 非智能体组件（通常只是代码，不是 LLM），根据规则在智能体之间路由消息。 |
| 共识（Consensus） | "智能体们达成一致" | 多方智能体必须在继续之前达成一致的协议。用于需要解决冲突输出的场景。 |
| 涌现行为（Emergent behavior） | "智能体们自己想出来的" | 系统级模式，源自智能体交互但非显式编程。可以是有用的或有害的。 |
| 扇出/扇入（Fan-out/fan-in） | "智能体的 Map-Reduce" | 将任务拆分到并行智能体（扇出），然后组合它们的结果（扇入）。 |
| 消息传递（Message passing） | "智能体们互相交谈" | 智能体之间的通信机制：结构化数据从一个智能体发送到另一个智能体，替代共享上下文窗口。 |

## 延伸阅读

- [新兴 AI 智能体架构概览](https://arxiv.org/abs/2409.02977)——多智能体模式调查
- [AutoGen：实现下一代 LLM 应用](https://arxiv.org/abs/2308.08155)——微软的多智能体对话框架
- [Claude Code 子智能体文档](https://docs.anthropic.com/en/docs/claude-code)——Claude Code 如何用 Task 进行委托
- [CrewAI 文档](https://docs.crewai.com/)——基于角色的多智能体框架
