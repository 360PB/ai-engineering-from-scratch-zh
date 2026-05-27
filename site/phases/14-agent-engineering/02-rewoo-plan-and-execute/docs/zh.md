# ReWOO 和 Plan-and-Execute：解耦规划

> ReAct 在一个流中交错思考和行动。ReWOO 将它们分开：先做一个大计划，然后执行。Token 减少 5 倍，HotpotQA 上准确率提高 +4%，你可以将规划器蒸馏到 7B 模型。Plan-and-Execute 将其泛化；Plan-and-Act 将其扩展到 Web 导航。

**类型：** 构建
**语言：** Python（标准库）
**前置要求：** Phase 14 · 01（Agent 循环）
**时间：** 约 60 分钟

## 学习目标

- 解释为什么 ReWOO 的规划器 / 工作器 / 求解器分离比 ReAct 的交错循环节省 token 并提高鲁棒性。
- 实现计划 DAG、依赖顺序执行器和组合工作器输出的求解器——全部标准库。
- 使用 2026 年"五种工作流模式"框架（Anthropic）决定何时任务应作为计划然后执行 vs 交错的 ReAct。
- 识别何时 Plan-and-Act 的综合计划数据需要用于长期网络或移动任务。

## 问题

ReAct 的交错思考-行动-观察循环简单灵活，但每次工具调用必须携带完整先前上下文——包括每个先前思考。Token 使用随深度二次增长。更糟的是：当工具在循环中途失败时，模型必须从错误观察重新推导整个计划。

ReWOO（Xu 等人，arXiv:2305.18323，2023 年 5 月）注意到这一点并下了赌注：先制定整个计划，并行获取证据，最后组合答案。一次 LLM 调用做计划，N 次工具调用获取证据（可并行），一次 LLM 调用求解。权衡是更少的灵活性（计划是静态的）换取更好的 token 效率和更清晰的失败模式。

## 概念

### 三个角色

```
Planner:  user_question -> [plan_dag]
Workers:  [plan_dag]     -> [evidence]        (工具调用，可能并行)
Solver:   user_question, plan_dag, evidence -> final_answer
```

Planner 产生一个 DAG。每个节点命名一个工具、其参数以及它依赖的先前节点（引用如 `#E1`、`#E2`）。Workers 按拓扑顺序执行节点。Solver 将所有内容缝合在一起。

### 为什么 token 减少 5 倍

ReAct 的提示长度随步数线性增长。在第 10 步时，提示包含思考 1 加行动 1 加观察 1 加思考 2 加行动 2 加观察 2，等等。每个中间步骤还冗余地包含原始提示。

ReWOO 支付一个大型规划器提示、N 个小型工作器提示（每个只是工具调用，无链）和一个求解器提示。在 HotpotQA 上论文测量 token 减少约 5 倍，同时准确率提高 +4。

### 为什么它更鲁棒

如果工作器 3 在 ReAct 中失败，循环必须在流中推理出错误。在 ReWOO 中，工作器 3 返回错误字符串；求解器在上下文中看到它带有原始计划，可以优雅降级。失败本地化是按节点，而非按步。

### 规划器蒸馏

论文的第二个结果：因为规划器看不到观察，你可以用 175B 教师模型的规划器输出微调一个 7B 模型。小模型处理规划；大模型在推理时不需要。这现在是标准——2026 年许多生产 Agent 使用小型规划器和大型执行器，反之亦然。

### Plan-and-Execute（LangChain，2023）

LangChain 团队 2023 年 8 月的帖子将此泛化为模式名称：Plan-and-Execute。前置规划器发出步骤列表，执行器运行每步，可选重新规划器在观察结果后可以修订。这更接近 ReAct 而非 ReWOO（重新规划器将观察带回规划），但保留了 token 节省。

### Plan-and-Act（Erdogan 等人，arXiv:2503.09572，ICML 2025）

Plan-and-Act 将该模式扩展到长期网络和移动 Agent。关键贡献是综合计划数据：一个带标签的轨迹生成器产生训练数据，其中计划是明确的。用于微调规划器模型，使它们在 WebArena 类任务上超过 30-50 步仍能继续工作，而单一 ReAct 轨迹会失去连贯性。

### 何时选择哪个

| 模式 | 何时 |
|------|------|
| ReAct | 短任务、未知环境、需要反应式异常处理 |
| ReWOO | 结构化任务、已知工具、token 敏感、可并行化的证据 |
| Plan-and-Execute | 类似 ReWOO 但在部分执行后重新规划 |
| Plan-and-Act | 长期（>30 步）、网络/移动/computer-use |
| 思维树 | 搜索值得付费（第 04 课） |

Anthropic 2024 年 12 月指导：从最简单的开始。如果任务是一个工具调用加摘要，不要构建 ReWOO。如果任务是 40 步研究任务，不要单独使用 ReAct。

## 构建它

`code/main.py` 实现一个玩具 ReWOO：

- `Planner` — 一个脚本策略，从提示发出计划 DAG。
- `Worker` — 通过注册表调度每个节点的工具调用。
- `Solver` — 脚本组合，读取证据并产生最终答案。
- 依赖解析——引用如 `#E1` 在调度时用先前工作器输出替换。

演示回答"法国首都的人口，以百万为单位四舍五入是多少？"使用两步计划：（1）查找首都，（2）查找人口，然后求解。

运行它：

```
python3 code/main.py
```

追踪先显示完整计划，然后工作器结果，然后求解器组合。比较 token 计数（我们打印一个粗略字符数）与 ReAct 风格的交错运行——ReWOO 在这类结构化任务上胜出。

## 使用它

LangGraph 将 Plan-and-Execute 作为配方发货（`create_react_agent` 用于 ReAct，自定义图用于计划-执行）。CrewAI 的 Flows 直接编码模式：你预先定义任务，Flow DAG 执行它们。Plan-and-Act 的综合数据方法仍然主要是研究；运行时模式（显式计划 DAG）通过 LangGraph 和 CrewAI Flows 在生产中发货。

## 发布它

`outputs/skill-rewoo-planner.md` 给定工具目录，从用户请求生成 ReWOO 计划 DAG。它在交给执行器之前验证计划（无环、每个引用解析、每个工具存在）。

## 练习

1. 并行化独立计划节点的工作器执行。在有 2 个并行组的 6 节点 DAG 上这能带来什么？

2. 添加一个在任意工作器返回错误时触发的重新规划器节点。对 ReWOO 最小的改变是什么使其成为 Plan-and-Execute？

3. 用小型模型（7B 类）替换 `Planner`，保持 `Solver` 在前沿模型上。比较端到端质量——分割在哪里失败？

4. 阅读 ReWOO 论文第 4 节关于规划器蒸馏的内容。从概念上重现 175B → 7B 结果：你需要什么训练数据，如何评分计划质量？

5. 将玩具移植到 Plan-and-Act 的轨迹形状：计划是一个序列，不是 DAG。什么权衡改变了？

## 关键术语

| 术语 | 人们通常说 | 实际含义 |
|------|-----------|----------|
| ReWOO | "无观察推理" | 先计划，然后并行获取证据，然后求解——规划提示中无观察 |
| Plan-and-Execute | "LangChain 的计划-执行模式" | 带可选重新规划节点的 ReWOO（在执行后） |
| Plan-and-Act | "扩展的计划-执行" | 带综合计划训练数据的显式规划器/执行器分离，用于长期任务 |
| Evidence reference（证据引用） | "#E1, #E2, ..." | 计划节点占位符，在调度时用先前工作器输出替换 |
| Planner distillation（规划器蒸馏） | "小型规划器、大型执行器" | 用大型教师模型的规划器轨迹微调小型模型 |
| Token efficiency（Token 效率） | "更少往返" | 论文中 HotpotQA 上比 ReAct 少 5 倍 token |
| DAG executor（DAG 执行器） | "拓扑调度器" | 按依赖顺序运行计划节点；在每层并行 |

## 延伸阅读

- [Xu 等人，ReWOO：解耦推理与观察（arXiv:2305.18323）](https://arxiv.org/abs/2305.18323) — 规范论文
- [Erdogan 等人，Plan-and-Act（arXiv:2503.09572）](https://arxiv.org/abs/2503.09572) — 带综合计划的规模化规划器-执行器
- [LangGraph Plan-and-Execute 教程](https://docs.langchain.com/oss/python/langgraph/overview) — 框架配方
- [Anthropic，构建有效 Agent](https://www.anthropic.com/research/building-effective-agents) — 选择最简单的可行模式