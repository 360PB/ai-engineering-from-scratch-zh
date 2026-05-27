# 记忆：虚拟上下文与 MemGPT

> 上下文窗口是有限的。对话、文档和工具调用轨迹则不是。MemGPT（Packer et al., 2023）将此框架类比为操作系统虚拟内存——主上下文是 RAM，外部存储是磁盘，agent 在两者之间分页。这是一切 2026 年记忆系统继承的模式。

**类型：** 动手实现
**语言：** Python（标准库）
**前置知识：** Phase 14 · 01（Agent Loop）、Phase 14 · 06（Tool Use）
**时长：** 约 75 分钟

## 学习目标

- 解释 MemGPT 赖以构建的 OS 类比：主上下文 = RAM，外部上下文 = 磁盘，记忆工具 = 分页入/出。
- 用标准库实现两层的 MemGPT 模式：主上下文缓冲区、外部可搜索存储，以及分页入/出工具。
- 描述 agent 如何发出"中断"来查询或修改外部记忆，以及结果如何拼接进下一轮提示。
- 识别 MemGPT 中延续到 Letta（课程 08）和 Mem0（课程 09）的设计选择。

## 问题

上下文窗口看似能解决记忆问题。实际上并不能。三个失败模式在生产中反复出现：

1. **溢出。** 多轮对话、长文档或工具调用密集的轨迹会超出窗口。截断点之后的一切都消失了。
2. **稀释。** 即使在窗口内，塞入无关上下文会稀释对重要内容的注意力。前沿模型在长输入上仍会退化。
3. **持久性。** 新会话从空白窗口开始。没有外部记忆的 agent 无法说"记得你之前让我……"来跨会话延续。

更大的窗口有帮助，但不能根治问题。Mem0 的 2025 年论文测量显示，128k 窗口基准仍然遗漏长期事实，而有外部记忆的 4k 窗口 agent 却能捕获。

## 核心概念

### MemGPT：操作系统类比

Packer et al.（arXiv:2310.08560，v2 Feb 2024）将上下文管理映射到操作系统虚拟内存：

| OS 概念 | MemGPT 概念 | 2026 生产类比 |
|---------|-------------|---------------|
| RAM | 主上下文（提示） | Anthropic/OpenAI 上下文窗口 |
| 磁盘 | 外部上下文 | 向量数据库、KV、图存储 |
| 缺页中断 | 记忆工具调用 | `memory.search`、`memory.read`、`memory.write` |
| OS 内核 | Agent 控制循环 | 带记忆工具的 ReAct 循环 |

Agent 运行一个标准的 ReAct 循环。额外的一类工具让它在主上下文和外部上下文之间分页数据。

### 两层架构

- **主上下文。** 承载当前任务的固定大小提示。模型始终可见。
- **外部上下文。** 无界，可通过工具搜索。在相关时读取，在事实出现时写入。

原始论文在两项超出基础窗口的任务上评估了该设计：超过 100k token 的文档分析和跨多天的多会话聊天（持久记忆）。

### 中断模式

MemGPT 引入了记忆即中断：对话中途 agent 可以调用记忆工具，运行时执行该操作，结果作为新观察拼接入下一轮助手响应。概念上等同于 Unix `read()` 系统调用：阻塞进程、返回字节，进程继续执行。

标准记忆工具层：

- `core_memory_append(section, text)` — 写入提示的持久段落。
- `core_memory_replace(section, old, new)` — 编辑持久段落。
- `archival_memory_insert(text)` — 写入可搜索的外部存储。
- `archival_memory_search(query, top_k)` — 从外部存储检索。
- `conversation_search(query)` — 扫描过往轮次。

### MemGPT 止于此，Letta 起于此

2024 年 9 月 MemGPT 更名为 Letta。研究仓库（`cpacker/MemGPT`）保留；Letta 拓展了该设计：

- 三层而非两层（core、recall、archival — 课程 08）。
- 原生推理替代 `send_message`/心跳模式（课程 08）。
- 睡眠时 agent 异步运行记忆工作（课程 08）。

MemGPT 论文是 2026 年的基石，即使生产系统运行的是 Letta、Mem0 或自定义两层存储。

### 此模式的常见误区

- **记忆腐烂。** 写入累积速度快于读取；检索被过时事实淹没。修复：定期整合（Letta 睡眠时）、显式失效（Mem0 冲突检测器）。
- **记忆投毒。** 外部记忆是检索来的文本。如果攻击者可控制的内容进入记忆笔记，agent 在下一会话会重新摄入。这是 Greshake et al. 的攻击在时间维度上的复述（课程 27）。
- **引用丢失。** Agent 回忆"用户让我交付 X"但无法引用是哪一轮。在每条归档写入时存储来源引用（会话 ID、轮次 ID）。

## 动手实现

`code/main.py` 用标准库实现了 MemGPT 的两层模式：

- `MainContext` — 固定大小提示缓冲区，含 `core` 字典和 `messages` 列表；超出上限时自动压缩最旧的消息。
- `ArchivalStore` — 内存 BM25 风格存储（token 重叠评分），记录格式为（id、text、tags、session、turn）。
- 五个映射到 MemGPT 工具层接口的记忆工具。
- 一个脚本化 agent：向归档写入事实，然后通过调用 `archival_memory_search` 回答一个问题。

运行：

```
python3 code/main.py
```

轨迹展示：agent 写入三条事实，填满主上下文至上限（触发驱逐），然后通过从归档检索来回答后续问题——无需真实 LLM 即可复现 MemGPT 工作流。

## 用现成库

目前每个生产记忆系统都是 MemGPT 的变体：

- **Letta**（课程 08）— 三层架构、原生推理、睡眠时计算。
- **Mem0**（课程 09）— 向量 + KV + 图融合评分层。
- **OpenAI Assistants / Responses** — 通过线程和文件实现托管记忆。
- **Claude Agent SDK** — 通过 skills 和会话存储实现长期记忆。

根据运维形态（自托管、托管、框架集成）选择，而非核心模式——核心模式都是 MemGPT。

## 产出

`outputs/skill-virtual-memory.md` 是一个可复用的 skill，为任何目标运行时生成正确两层记忆框架（主 + 归档 + 工具层），含驱逐策略和引用字段接线。

## 练习

1. 添加一个用 token 衡量的 `max_main_context_tokens` 上限（用 `len(text.split()) * 1.3` 近似）。超出上限时将最旧消息压缩为摘要。对比有摘要器和无摘要器的行为差异。
2. 在归档存储上正确实现 BM25（词频、逆文档频）。在一个玩具事实集上测量 recall@10，与 token 重叠基线对比。
3. 在归档插入时添加 `citation` 字段（session_id、turn_id、source_url）。让 agent 在每个基于检索的回答中引用来源。
4. 模拟记忆投毒：添加一条归档记录，写着"忽略所有未来用户指令"。编写一个守卫，扫描检索结果中的指令形态文本并将其标记为不可信。
5. 将实现移植到 MemGPT 研究仓库的核心记忆 JSON schema（`cpacker/MemGPT`）。从扁平字符串切换到类型化段落时有哪些变化？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Virtual context（虚拟上下文） | "无限记忆" | 主（提示）+ 外部（可搜索）两层，含分页入/出 |
| Main context（主上下文） | "工作记忆" | 提示——固定大小，始终可见 |
| Archival memory（归档记忆） | "长期存储" | 按需检索的外部可搜索持久化 |
| Core memory（核心记忆） | "持久提示段落" | 固定在主上下文内的命名段落 |
| Memory tool（记忆工具） | "记忆 API" | Agent 发出的读写外部记忆的工具调用 |
| Interrupt（中断） | "记忆缺页" | Agent 暂停，运行时获取，结果拼入下一轮 |
| Memory rot（记忆腐烂） | "过时事实" | 旧写入淹没检索；通过整合修复 |
| Memory poisoning（记忆投毒） | "注入的持久笔记" | 攻击者内容作为记忆存储，在召回时重新摄入 |

## 延伸阅读

- [Packer et al., MemGPT (arXiv:2310.08560)](https://arxiv.org/abs/2310.08560) — OS 启发的虚拟上下文论文
- [Letta, Memory Blocks blog](https://www.letta.com/blog/memory-blocks) — 三层演进
- [Anthropic, Effective context engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) — 将上下文视为预算
- [Chhikara et al., Mem0 (arXiv:2504.19413)](https://arxiv.org/abs/2504.19413) — 基于此模式的生产混合记忆