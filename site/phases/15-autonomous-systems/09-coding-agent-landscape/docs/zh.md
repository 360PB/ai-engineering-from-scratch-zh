# 自主编码智能体图景（2026）

> SWE-bench Verified 在不到三年内从 4% 提升到 80.9%。同一 Claude Sonnet 4.5 在 SWE-agent v1 上得分 43.2%，在 Cline 自主模式下达 59.8%——模型周围的脚手架现在与模型本身一样重要。OpenHands（前 OpenDevin）是最活跃的 MIT 许可平台，其 CodeAct 循环直接在沙箱中执行 Python 操作，而不是 JSON 工具调用。标题数字掩盖了一个方法论问题：500 个 SWE-bench Verified 任务中有 161 个只需要 1-2 行修改，而 SWE-bench Pro（10+ 行修改任务）对于相同的前沿模型仅为 23-59%。

**类型：** Learn
**语言：** Python（标准库，CodeAct 与 JSON 工具调用比较）
**前置要求：** Phase 14 · 07（工具使用），Phase 15 · 01（长时域智能体）
**时间：** 约 45 分钟

## 问题

"哪个编码智能体最好"是错误的问题。正确的问题是：在我工作的任务分布上，在我将在生产中运行的脚手架下，我能得到怎样的端到端可靠性？

在 2022 年到 2026 年之间，该领域认识到脚手架——检索层、规划器、沙箱、编辑验证循环、反馈格式——是承重构件。Claude Sonnet 4.5 在 SWE-agent v1 上得分 43.2%；同一模型在 Cline 的自主脚手架中得分 59.8%。16.6 个绝对百分点的差异，相同的权重。基础模型是一个组件；循环是产品。

伴随问题是基准饱和掩盖了回归。SWE-bench Verified 接近饱和，简单任务尾（500 个任务中有 161 个需要 ≤2 行）拉高了顶级分数。真实世界质量在像 SWE-bench Pro（10+ 行修改）这样的分布上测量更好，在相同的领先者上仍处于 23-59%。

## 概念

### SWE-bench，一段话

SWE-bench（Jimenez 等人）以带真实补丁的 GitHub issues 为输入，要求智能体产生使测试套件通过的补丁。SWE-bench Verified（OpenAI，2024 年）是人类策划的 500 任务子集，移除了模糊和损坏的任务。SWE-bench Pro 是更难的后继——需要 10+ 行修改的任务，当前前沿智能体处于 23-59%。

### 2022 → 2026 曲线实际显示的内容

- **2022**：研究模型在原始 SWE-bench 上约 4%。
- **2024**：GPT-4 + Devin 风格脚手架约 14%；SWE-agent 约 12%。
- **2025**：Claude 3.5/3.7 Sonnet 在 Aider 和 SWE-agent 中推入 40-55% 范围。
- **2026**：Claude Sonnet 4.5 和前沿竞争者在 SWE-bench Verified 上达到 70-80%+。Epoch AI 的排行榜实时追踪这一趋势。

斜率来自三个复合来源：更好的基础模型、更好的脚手架（CodeAct、反思、验证器循环）和更好的基准（Verified 移除噪声）。

### CodeAct vs JSON 工具调用

OpenHands（All-Hands-AI，arXiv:2407.16741，前 OpenDevin）采用了一个特定的架构赌注：模型不是发出 JSON 工具调用由主机解码和执行，而是发出 Python 代码和 Jupyter 风格的kernel在沙箱中运行它。智能体可以在一个操作中循环遍历文件、链接工具并捕获自己的异常。

权衡：

- **JSON 工具调用**：每个操作是一轮；易于审计；组合性有限；默认安全是因为每次调用都通过显式验证器。
- **CodeAct**：一个操作可以是一个完整程序；组合性强；需要加固的沙箱（OpenHands 使用 Docker 隔离）；失败模式包括沙箱运行时允许的任何内容。

两种架构都在生产中。CodeAct 在开放平台（OpenHands、smolagents）中占主导。JSON 工具调用在托管服务（Anthropic Managed Agents、OpenAI Assistants）中占主导，在那里提供商控制执行器。

### 2026 图景中的脚手架

| 脚手架 | 许可 | 执行模型 | 显著特性 |
|---|---|---|---|
| OpenHands (OpenDevin) | MIT | Docker 中的 CodeAct | 最活跃的开放平台；事件流可回放 |
| SWE-agent | MIT | 智能体-计算机接口 (ACI) | 第一个端到端 SWE-bench 脚手架 |
| Aider | Apache-2 | 在本地仓库中通过 diff 编辑 | 最小脚手架，强回归稳定性 |
| Cline | Apache-2 | 带工具策略的 VS Code 智能体 | Sonnet 4.5 上最高评分的开放脚手架 |
| Devin (Cognition) | 专有 | 托管 VM + 规划器 | 第一个"AI 软件工程师"产品类别 |
| Claude Code | 专有 | 权限模式 + 例程 | 第 10 课详细涵盖智能体循环 |

### 为什么脚手架占主导地位

编码运行是一条长时域轨迹（第 1 课）。可靠性跨步骤复合。脚手架在三个地方获得分数：

1. **检索**：找到要读取的正确文件是静默的瓶颈。SWE-agent 的 ACI、OpenHands 的文件索引和 Aider 的 repo-map 都针对这一点。
2. **验证器循环**：运行测试、读取堆栈跟踪并重新尝试，在 SWE-bench 上是 10+ 点的增量。
3. **故障containment**：出错时回滚的沙箱防止复合损害。有无验证器循环的同一模型看起来像两个不同的产品。

### 基准饱和与真实分布

OpenHands 作者和 Epoch AI 都指出，SWE-bench Verified 有一个简单的尾：500 个任务中有 161 个只需要 1-2 行修改。高分数部分由此尾驱动。SWE-bench Pro 限制为 10+ 行修改，在前沿系统的分数范围为 23-59%。你的生产分布几乎肯定比 Pro 更接近 Verified。

选择智能体的启示：在你的 bug 待办列表中运行类似 Pro 的子集。重要的分数是你在代表性任务上的分数。

## 使用它

`code/main.py` 在固定的迷你任务分布上比较两个玩具智能体脚手架：

1. 一个每次操作一轮的 **JSON 工具调用** 脚手架。
2. 一个每次操作可以发出小 Python 代码片段的 **CodeAct** 脚手架。

两者都使用一个存根"模型"（确定性规则），因此比较将脚手架与模型质量隔离。输出显示 CodeAct 脚手架以更少的轮次解决更多任务，但代价是每个操作更大的爆炸半径。

## 交付它

`outputs/skill-scaffold-audit.md` 帮助你在采用前审计提议的编码智能体脚手架：检索质量、验证器存在、沙箱隔离和基准到分布的契合度。

## 练习

1. 运行 `code/main.py`。每个脚手架在相同任务集上花费多少轮？每个的每操作爆炸半径是多少？

2. 阅读 OpenHands 论文（arXiv:2407.16741）。论文认为 CodeAct 在复杂任务上优于 JSON 工具调用。找出论文承认的一个失败模式，并写一句话说明该模式何时在生产中占主导。

3. 从你的 bug 待办列表中挑选一个需要跨两个文件修改 10+ 行的任务。估计前沿模型在 (a) JSON 工具调用和 (b) CodeAct 下的端到端成功概率。说明差距。

4. SWE-bench Verified 有 161 个单文件、1-2 行任务。构建一个排除它们的分数。排行榜如何重新排序？

5. 阅读"Introducing SWE-bench Verified"（OpenAI）。解释用于移除模糊任务的具体方法，并命名审查会遗漏的一类。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|---|---|---|
| SWE-bench | "编码基准" | 带真实补丁和测试套件的 GitHub issues |
| SWE-bench Verified | "清理子集" | 500 个人类策划任务，存在简单尾 |
| SWE-bench Pro | "更难子集" | 10+ 行修改；前沿处于 23-59% |
| CodeAct | "代码即操作" | 智能体发出 Python； Jupyter 风格kernel在沙箱中执行 |
| JSON 工具调用 | "函数调用" | 每个操作是一个结构化 JSON 有效载荷，在执行前验证 |
| 脚手架 | "智能体框架" | 基础模型周围的检索 + 规划器 + 执行器 + 验证器循环 |
| ACI（智能体-计算机接口） | "SWE-agent 的格式" | 为 LLM 人体工程学设计而非人类 shell 的命令集 |
| 验证器循环 | "测试和重试" | 运行测试、读取输出、修改补丁；非模型可靠性的最大增益 |

## 进一步阅读

- [Jimenez et al. — SWE-bench](https://www.swebench.com/) — 原始基准和方法论。
- [OpenAI — Introducing SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) — 如何构建策划子集。
- [Wang et al. — OpenHands: An Open Platform for AI Software Developers](https://arxiv.org/abs/2407.16741) — CodeAct 架构和事件流设计。
- [Epoch AI — SWE-bench leaderboard](https://epoch.ai/benchmarks) — 实时追踪分数。
- [Anthropic — Measuring agent autonomy](https://www.anthropic.com/research/measuring-agent-autonomy) — 长时域编码智能体可靠性框架。