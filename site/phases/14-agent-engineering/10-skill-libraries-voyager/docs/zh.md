# 技能库与终身学习（Voyager）

> Voyager（Wang 等，TMLR 2024）将可执行代码视为技能。技能有名称、可检索、可组合，并通过环境反馈持续精调。这是 2026 年 Claude Agent SDK skills、skillkit 以及技能库模式的参考架构。

**类型：** 动手实现
**语言：** Python（标准库）
**前置知识：** 第 14 阶段 · 07（MemGPT），第 14 阶段 · 08（Letta Blocks）
**学习时长：** 约 75 分钟

## 学习目标

- 说出 Voyager 的三个组件——自动课程、技能库、迭代提示——以及各自的作用。
- 解释为什么 Voyager 将动作空间设为代码，而非原始命令。
- 用标准库实现一个技能库，包含注册、检索、组合和失败驱动的精调。
- 将 Voyager 的模式映射到 2026 年 Claude Agent SDK skills 和 skillkit 生态。

## 问题背景

每个会话都从零重建能力的 Agent 会犯三个错误：

1. **浪费 token。** 每个任务都重新生成同样的推理过程。
2. **丢失进度。** 在会话 A 中学到的纠正无法传递到会话 B。
3. **长时序组合失败。** 复杂任务需要能力层次；一次性提示无法表达。

Voyager 的答案是：将每个可复用能力视为一段有名称的代码块，存入库中，可按相似度检索，可与其他技能组合，并可通过执行反馈持续精调。

## 核心概念

### 三个组件

Voyager（arXiv:2305.16291）围绕以下三点构建 Agent：

1. **自动课程。** 由好奇心驱动的提议器根据 Agent 当前技能集和环境状态挑选下一个任务。探索是自底向上的。
2. **技能库。** 每个技能都是可执行代码。新技能在任务成功时添加。技能通过查询与描述的相似度检索。
3. **迭代提示机制。** 失败时，Agent 收到执行错误、环境反馈和自我验证输出，然后精调该技能。

Minecraft 评估（Wang 等，2024）：与基线相比，独特物品数量提升 3.3 倍，石器快 8.5 倍，铁器快 6.4 倍，地图探索距离长 2.3 倍。这些数字是 Minecraft 特定的，但模式可迁移。

### 动作空间 = 代码

大多数 Agent 输出原始命令。Voyager 输出 JavaScript 函数。一个技能如下：

```
async function craftIronPickaxe(bot) {
  await mineIron(bot, 3);
  await mineStick(bot, 2);
  await placeCraftingTable(bot);
  await craft(bot, 'iron_pickaxe');
}
```

由子技能组合而成。以描述和嵌入为键存储。检出不只是一段提示，而是一个程序。

这就是 2026 年 Claude Agent SDK skill：一段有名称、可检索的代码，加载时由 Agent 按需调用。

### 技能检索

新任务"制作钻石镐"。Agent：

1. 嵌入任务描述。
2. 在技能库中查询 top-k 相似的技能。
3. 检出 `craftIronPickaxe`、`mineDiamond`、`placeCraftingTable` 等。
4. 由检出的原语和新逻辑组合成新技能。

这就是 MCP resources（第 13 阶段）和 Agent SDK skills 实现的模式：在知识/代码表面上检索，作用域限定在当前任务。

### 迭代精调

Voyager 的反馈循环：

1. Agent 编写一个技能。
2. 技能在环境中运行。
3. 返回三种信号之一：`success`、`error`（含堆栈跟踪）、`self-verification failure`。
4. Agent 依据信号重写该技能。
5. 循环直到成功或达到最大轮数。

这是 Self-Refine（第 5 课）应用于代码生成并结合环境验证。CRITIC（第 5 课）是同一模式，以外部工具作为验证器。

### 课程与探索

Voyager 的课程模块根据 Agent 已有的和尚未完成的事项，提议"在湖边建一个庇护所"这样的任务。提议器利用环境状态 + 技能清单选择一项略高于当前能力的任务——即探索的甜蜜点。

对于生产级 Agent，这意味着"还缺什么"算子：给定当前技能库和某个领域，我们还没有覆盖哪些技能？团队通常将其手工实现为课程评审。

### 该模式的常见失效

- **技能库腐化。** 同一个技能以略有不同的描述被添加了 10 次。在写入时去重；检索只返回一个。
- **组合技能漂移。** 父技能依赖的子技能被精调了。给技能加版本号；父技能固定在 v1 就不会悄无声息地升级到 v3。
- **检索质量。** 技能描述的向量检索在库超过几百个后质量下降。补充标签过滤和硬约束（"只看 `category=tooling` 的技能"）。

## 动手实现

`code/main.py` 用标准库实现技能库：

- `Skill` — 名称、描述、代码（字符串）、版本、标签、依赖。
- `SkillLibrary` — 注册、搜索（token 重叠）、组合（依赖的拓扑排序）和精调（更新时版本号递增）。
- 一个脚本化 Agent，注册三个原语技能，组合第四个，遇到失败后精调。

运行：

```
python3 code/main.py
```

跟踪输出展示库写入、检索、组合、一次失败的执行和 v2 精调——Voyager 循环的端到端全过程。

## 用现成库

- **Claude Agent SDK skills**（Anthropic）— 2026 年参考实现：每个 skill 有描述、代码和指令；在 Agent 会话中按需加载。
- **skillkit**（npm: skillkit）— 跨 Agent 技能管理，支持 32+ 款 AI 编程 Agent。
- **自建技能库** — 领域特定（数据 Agent 的 SQL 技能，基础架构 Agent 的 Terraform 技能）。Voyager 模式可缩减规模。
- **OpenAI Agents SDK `tools`** — 低配方案；每个 tool 是一个轻量技能。

## 产出

`outputs/skill-skill-library.md` 生成一个 Voyager 形态的技能库，包含注册、检索、版本管理和精调，为任意目标运行时而准备。

## 练习

1. 给 `compose()` 增加循环依赖检测器。当技能 A 依赖 B，B 又依赖 A 时会发生什么？报错还是警告？
2. 实现技能级版本固定。当父技能组合了子技能 `crafting@1` 时，对 `crafting@2` 的精调不得静默升级父技能。
3. 用 sentence-transformers 嵌入（或标准库 BM25 实现）替换 token 重叠检索。在 50 个技能的玩具库上测量 retrieval@5。
4. 增加一个"课程"Agent：给定当前库和领域描述，提出 5 个缺失的技能。每周调用一次。
5. 阅读 Anthropic 的 Claude Agent SDK skill 文档。将玩具库迁移到 SDK 的 skill schema。发现力（discoverability）有什么变化？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| Skill（技能） | "可复用能力" | 有名称的代码块，含描述，按相似度可检索 |
| Skill library（技能库） | "Agent 的方法论记忆" | 技能的持久化存储，可搜索、可组合 |
| Curriculum（课程） | "任务提议器" | 由当前能力缺口驱动的自底向上目标生成器 |
| Composition（组合） | "技能 DAG" | 技能调用技能；执行时按拓扑序排列 |
| Iterative refinement（迭代精调） | "自修正循环" | 环境反馈 + 错误 + 自我验证注入下一版本 |
| Action-space-as-code（动作空间即代码） | "程序化动作" | 输出函数而非原始命令，以支持时序扩展行为 |
| Dedup on write（写入时去重） | "技能归并" | 相近描述的技能合并为一个规范技能 |

## 延伸阅读

- [Wang 等，Voyager（arXiv:2305.16291）](https://arxiv.org/abs/2305.16291) — 技能库原始论文
- [Claude Agent SDK 概述](https://platform.claude.com/docs/en/agent-sdk/overview) — skills 作为 2026 年产品化形态
- [Anthropic，Building agents with the Claude Agent SDK](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk) — 实践中的 skills 和子 Agent
- [Madaan 等，Self-Refine（arXiv:2303.17651）](https://arxiv.org/abs/2303.17651) — Voyager 精调循环的底层逻辑