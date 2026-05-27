# 记忆分块与睡眠时计算（Letta）

> MemGPT 于 2024 年更名为 Letta。2026 年的演进加入了两个理念：模型可直接编辑的离散功能记忆分块，以及在主体 agent 空闲时异步整合记忆的睡眠时 agent。以下是超越单次对话扩展记忆的方式。

**类型：** 动手实现
**语言：** Python（标准库）
**前置知识：** Phase 14 · 07（MemGPT）
**时长：** 约 75 分钟

## 学习目标

- 说出荷刻（Letta）使用的三个记忆层（core、recall、archival）及其各自作用。
- 解释记忆分块模式：Human 块、Persona 块和用户自定义分块作为一等类型化对象。
- 描述什么是睡眠时计算，为何它位于关键路径之外，以及为何它可以运行比主体 agent 更强的模型。
- 实现脚本化的双 agent 循环：主体 agent 提供响应，睡眠时 agent 在轮次之间整合分块。

## 问题

MemGPT（课程 07）解决了虚拟内存控制流。但三个生产问题浮现出来：

1. **延迟。** 每个记忆操作都落在关键路径上。如果 agent 在用户等待时需要剪枝、摘要或调和，尾部延迟就会飙升。
2. **记忆腐烂。** 写入不断累积。矛盾的事实留存。检索被陈旧内容淹没。
3. **结构丢失。** 扁平的归档存储无法表达"Human 块始终在提示中；Persona 块始终在提示中；Task 块每次会话切换"。

Letta（letta.com）是 2026 年的重写。记忆分块使结构显式化；睡眠时计算将整合移出关键路径。

## 核心概念

### 三层架构

| 层级 | 范围 | 存放位置 | 写入方 |
|------|-------|----------|---------|
| Core | 始终可见 | 主提示内部 | Agent 工具调用 + 睡眠时重写 |
| Recall | 对话历史 | 可检索 | 自动轮次记录 |
| Archival | 任意事实 | 向量 + KV + 图 | Agent 工具调用 + 睡眠时摄入 |

Core 就是 MemGPT 的核心。Recall 是对话缓冲区及其被逐出的尾部。Archival 是外部存储。三层拆分理清了 MemGPT 两层混用的设计。

### 记忆分块

分块（block）是核心层中类型化、持久化、可编辑的片段。原始 MemGPT 论文定义了两个：

- **Human 块** — 关于用户的事实（姓名、角色、偏好、目标）。
- **Persona 块** — agent 的自我概念（身份、语气、约束）。

Letta 泛化到任意用户自定义分块：当前目标的 `Task` 块、代码库事实的 `Project` 块、硬约束的 `Safety` 块。每个分块有 `id`、`label`、`value`、`limit`（字符上限）和 `description`（供模型判断何时编辑）。

分块通过工具层编辑：

- `block_append(label, text)` — 追加内容
- `block_replace(label, old, new)` — 替换内容
- `block_read(label)` — 读取分块
- `block_summarize(label)` — 压缩接近上限的分块

### 睡眠时计算

这是 Letta 2025 年的新增功能：在后台运行第二个 agent，位于关键路径之外。睡眠时 agent 处理对话记录和代码库上下文，将 `learned_context` 写入共享分块，整合或失效归档记录。

由此引出的特性：

- **零延迟代价。** 主体响应不等待记忆操作。
- **允许更强的模型。** 睡眠时 agent 可以使用更昂贵、更慢的模型，因为它不受延迟约束。
- **自然的整合窗口。** 在用户不等待时去重、摘要、失效矛盾事实。

这与人类的工作方式相符：做任务、睡一觉，长期记忆隔夜沉淀。

### Letta V1 与原生推理

Letta V1（`letta_v1_agent`，2026）弃用了 `send_message`/心跳和内联 `Thought:` 标记，转而采用原生推理。Responses API（OpenAI）和带扩展思考的 Messages API（Anthropic）在独立通道上发出推理内容，并在轮次间传递（在生产环境中跨提供商加密传输）。控制循环仍是 ReAct。思维轨迹是结构化的，而非提示形态的。

### 此模式的常见误区

- **分块膨胀。** 无限 `block_append` 会快速触及上限。在写入即将超过上限之前，串联一个分块摘要器。
- **静默漂移。** 睡眠时 agent 重写了分块，但主体 agent 从未察觉。对分块进行版本化管理，在轨迹中展示差异。
- **污染整合。** 睡眠时 agent 处理了攻击者可触达的内容并写入 core。课程 27 也适用于睡眠时表面。

## 动手实现

`code/main.py` 实现了：

- `Block` — id、label、value、limit、description。
- `BlockStore` — CRUD + `near_limit(label)` 辅助方法。
- 两个脚本化 agent — `PrimaryAgent` 处理一轮，`SleepTimeAgent` 在轮次之间整合。
- 一条轨迹，展示三轮对话中的分块写入，以及一个睡眠时过程：摘要化一个分块并失效一条过时事实。

运行：

```
python3 code/main.py
```

轨迹展示了分离效果：主体轮次快速且产生原始写入；睡眠过程则压缩和清理。

## 用现成库

- **Letta**（letta.com）提供参考实现。可自托管或使用托管云服务。
- **Claude Agent SDK skills** 作为分块形态的知识 — skill 是命名的、带版本的、可检索的一组指令，agent 按需加载。
- **自定义构建** 适用于想控制存储后端的团队。使用 Letta API 合约，以便后续迁移。

## 产出

`outputs/skill-memory-blocks.md` 生成一个带睡眠时钩子的 Letta 形态分块系统，可用于任何运行时，含安全规则和引用接线。

## 练习

1. 添加一个 `block_summarize` 工具：当 `near_limit` 返回 true 时，用模型生成的摘要替换分块值。哪个触发阈值能最小化摘要调用和分块溢出？
2. 在归档上实现睡眠时去重：两个文本 token 重叠超过 90% 的记录合并为一条。仅在睡眠过程中进行，不在关键路径上。
3. 对分块进行版本化管理。每次写入时记录旧值和差异。暴露 `block_history(label)` 让运维人员调试"agent 为何忘记 X"。
4. 将睡眠时 agent 视为不受信任的写入方。当它们触碰 Persona 或 Safety 块时，需要第二个 agent 审查后再提交。
5. 将示例移植到 Letta API（`letta_v1_agent`）。分块 schema 有何变化？原生推理如何改变轨迹形态？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Memory block（记忆分块） | "可编辑的提示段落" | 核心记忆中类型化、持久化、可由 LLM 编辑的片段 |
| Human block（人类块） | "用户记忆" | 关于用户的事实，固定在 core 中 |
| Persona block（人格块） | "Agent 身份" | 自我概念、语气、约束，固定在 core 中 |
| Sleep-time compute（睡眠时计算） | "异步记忆工作" | 第二个 agent 在关键路径外进行整合 |
| Core / Recall / Archival | "三层架构" | 三层记忆划分：始终可见 / 对话 / 外部 |
| Block limit（分块上限） | "上限" | 每个分块的字符限制；触发摘要 |
| Native reasoning（原生推理） | "思考通道" | 提供商级别的推理输出，而非提示级 `Thought:` |
| Learned context（习得上下文） | "睡眠输出" | 睡眠时 agent 写入共享分块的事实 |

## 延伸阅读

- [Letta, Memory Blocks blog](https://www.letta.com/blog/memory-blocks) — 分块模式
- [Letta, Sleep-time Compute blog](https://www.letta.com/blog/sleep-time-compute) — 异步整合
- [Letta, Rearchitecting the Agent Loop](https://www.letta.com/blog/letta-v1-agent) — 原生推理重写
- [Packer et al., MemGPT (arXiv:2310.08560)](https://arxiv.org/abs/2310.08560) — 起源论文