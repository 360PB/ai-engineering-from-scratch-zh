# Agno 与 Mastra：生产运行时

> Agno（Python）和 Mastra（TypeScript）是 2026 年生产运行时的配对组合。Agno 瞄准微秒级 Agent 实例化和无状态 FastAPI 后端。Mastra 在 Vercel AI SDK 基底上提供 Agent、工具、工作流、统一模型路由和组合存储。

**类型：** 学习
**语言：** Python、TypeScript
**前置知识：** Phase 14 · 01（Agent 循环）、Phase 14 · 13（LangGraph）
**时间：** 约 45 分钟

## 学习目标

- 识别 Agno 的性能目标以及何时它们有意义。
- 说出 Mastra 的三个原语——Agents、Tools、Workflows——以及支持的服务器适配器。
- 解释为何无状态会话作用域的 FastAPI 后端是 Agno 推荐的生产路径。
- 根据技术栈选择 Agno vs Mastra（Python 优先 vs TypeScript 优先）。

## 问题背景

LangGraph、AutoGen、CrewAI 都是重量级框架。想要"只要 Agent 循环，快速，嵌入我的运行时"的团队会选用 Agno（Python）或 Mastra（TypeScript）。两者都放弃了一些框架owned 的原语，换取原始速度和与周围技术栈更紧密的契合。

## 核心概念

### Agno

- Python 运行时，前身 Phi-data。
- "无图、无链、无复杂模式——只有纯 Python。"
- 文档中的性能目标：Agent 实例化约 2μs，每个 Agent 约 3.75 KiB 内存，约 23 个模型提供商。
- 生产路径：无状态会话作用域的 FastAPI 后端。每次请求启动一个全新的 Agent；会话状态存放在数据库。
- 原生多模态（文本、图像、音频、视频、文件）和 Agentic RAG。

速度目标在每秒有数以千计的短生命周期 Agent（聊天扇入、评估流水线）时才有意义。当单个 Agent 运行 10 分钟时，意义就小多了。

### Mastra

- TypeScript，构建在 Vercel AI SDK 之上。
- 三个原语：**Agents**、**Tools**（Zod 类型化）、**Workflows**。
- 统一模型路由器——3,300+ 模型，覆盖 94 个提供商（2026 年 3 月）。
- 组合存储：memory、workflows、可观测性分别接入不同后端；ClickHouse 推荐用于大规模可观测性。
- Apache 2.0，`ee/` 目录采用源代码可用企业许可证。
- 服务器适配器支持 Express、Hono、Fastify、Koa；与 Next.js 和 Astro 一等整合。
- 附带 Mastra Studio（localhost:4111）用于调试。
- 22k+ GitHub stars，1.0（2026 年 1 月）每周 npm 下载 300k+。

### 定位

两者都不是要成为 LangGraph。它们的竞争点在：

- **语言契合度。** Python 优先团队用 Agno；TypeScript 优先团队用 Mastra。
- **运行时人体工程学。** Agno = 近零开销；Mastra = 与 Vercel 生态深度整合。
- **可观测性。** 两者都与 Langfuse/Phoenix/Opik 整合（第 24 课），但 Mastra Studio 是第一方的。

### 何时选择哪个

- **Agno** — Python 后端、许多短生命周期 Agent、强烈性能要求、FastAPI 技术栈。
- **Mastra** — TypeScript 后端、Next.js / Vercel 部署、统一多提供商模型路由、Zod 类型化工具。
- **LangGraph**（第 13 课）— 当持久化状态和显式图推理比原始速度更重要时。
- **OpenAI / Claude Agent SDK** — 当你需要提供商的产品化形态时（第 16–17 课）。

### 这个模式会出问题的地方

- **为性能而性能。** 工作负载是每个请求一次慢 Agent 调用，却选了 Agno 因为"2μs"听起来不错。开销不是瓶颈。
- **生态锁定。** Mastra 与 Vercel 的整合在 Vercel 上是加分项，在别处是减分项。
- **企业许可证混淆。** Mastra 的 `ee/` 目录是源代码可用，不是 Apache 2.0。计划 fork 前先读许可证。

## 动手实现

本课主要是比较——没有单个代码产物能同时公平对待两个框架。见 `code/main.py`：并排toy实现，一个最小"运行 Agent、流式输出、持久化会话"的流程实现两次（一次 Agno 形态，一次 Mastra 形态）。

运行：

```
python3 code/main.py
```

两个功能等效但结构不同的执行跟踪。

## 用现成库

- **Agno** — Python 后端，需要速度和 FastAPI 形态。
- **Mastra** — TypeScript 后端，多提供商和工作流原语。
- 两者都附带第一方可观测性钩子。都与 Langfuse 整合。

## 产出

`outputs/skill-runtime-picker.md` 根据技术栈、延迟预算和运维形态选择 Agno、Mastra、LangGraph 或提供商 SDK。

## 练习

1. 读取 Agno 文档。将标准库 ReAct 循环（第 01 课）迁移到 Agno。什么消失了？什么保留了？
2. 读取 Mastra 文档。将同一循环迁移到 Mastra。工具类型化发生了什么变化（Zod vs 无）？
3. 基准测试：在你的技术栈上测量 Agent 实例化延迟。Agno 的 2μs 对你的工作负载重要吗？
4. 设计迁移：如果你的 Python 技术栈一直在跑 CrewAI，迁移到 Agno 会损失什么？
5. 读取 Mastra 的 `ee/` 许可证条款。什么限制会影响开源 fork？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Agno | "快速 Python Agent" | 无状态会话作用域 Agent 运行时 |
| Mastra | "Vercel AI SDK 上的 TypeScript Agent" | Agents + Tools + Workflows + Model Router |
| Unified Model Router | "多提供商访问" | 单一客户端访问 3,300+ 模型，覆盖 94 个提供商 |
| Composite storage | "多后端" | memory/workflows/可观测性各自接入不同存储 |
| Mastra Studio | "本地调试器" | localhost:4111 UI，用于内省 Agent |
| Source-available | "非开源" | 许可证允许阅读源码但限制商业使用 |

## 延伸阅读

- [Agno Agent Framework docs](https://www.agno.com/agent-framework)：性能目标、FastAPI 整合
- [Mastra docs](https://mastra.ai/docs)：原语、服务器适配器、模型路由器
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview)：有状态图替代方案
- [Comet Opik](https://www.comet.com/site/products/opik/)：Mastra 整合引用的可观测性对比