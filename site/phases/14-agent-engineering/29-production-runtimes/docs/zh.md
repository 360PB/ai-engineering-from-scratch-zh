# 生产环境运行时：队列、事件、定时任务

> 生产 Agent 存在六种运行时形态：请求-响应、流式输出、持久执行、队列后台处理、事件驱动、定时任务。先选形态，再选框架。每种形态下可观测性都是核心依赖。

**类型：** 学习
**语言：** Python（标准库）
**前置要求：** Phase 14 · 13（LangGraph）、Phase 14 · 22（语音）
**时长：** 约 60 分钟

## 学习目标

- 说出六种生产环境运行时形态，分别对应哪些框架/产品形态。
- 解释为什么持久执行（LangGraph）对长时任务至关重要。
- 描述事件驱动运行时的特点，以及 Claude Managed Agents 在何时适用。
- 解释"可观测性是核心依赖"这一主张的含义。

## 问题

生产 Agent 会在 Jupyter 笔记本里根本不会出现的地方失败：第 37 步时网络超时、用户中途挂断语音电话、Cron 任务随机器重启而终止、后台 worker 内存耗尽。运行时形态决定了哪些故障是可存活的。

## 概念

### 请求-响应

- 同步 HTTP。用户等待完成。
- 只适用于短任务（<30 秒）。
- 技术栈：Agno（Python + FastAPI）、Mastra（TypeScript + Express/Hono/Fastify/Koa）。
- 可观测性：标准 HTTP 访问日志 + OTel 链路追踪。

### 流式输出

- SSE 或 WebSocket 渐进输出。
- LiveKit 扩展到 WebRTC 用于语音/视频（Lesson 22）。
- 技术栈：任意支持流式输出的框架 + 处理 SSE/WS 的前端。
- 可观测性：每块时序、首 token 延迟、尾延迟。

### 持久执行

- 每步后状态自动存档；故障后自动恢复。
- AutoGen v0.4 角色模型将故障隔离到单个 Agent（Lesson 14）。
- LangGraph 的核心差异化能力（Lesson 13）。
- 当步数未知、恢复成本高时必不可少。

### 队列/后台处理

- 任务进入队列，Worker 抢取，结果通过 Webhook 或发布/订阅返回。
- 长时 Agent 的必备模式（Anthropic 的计算机使用公告中提到每任务数十到数百步）。
- 技术栈：Celery（Python）、BullMQ（Node）、SQS + Lambda（AWS）、自建。
- 可观测性：队列深度、单任务延迟分布、DLQ 大小。

### 事件驱动

- Agent 订阅触发器：新邮件、PR 打开、Cron 触发等。
- Claude Managed Agents 原生覆盖此场景（Lesson 17）。
- CrewAI Flows（Lesson 15）构建事件驱动的确定性工作流。
- 可观测性：触发源、事件到启动延迟、Agent 延迟。

### 定时任务

- Cron 形态的 Agent 周期性运行。
- 配合持久执行使用，失败的夜间任务在下一个周期自动恢复。
- 技术栈：Kubernetes CronJob + 持久框架；托管方案（Render cron、Vercel cron）。

### 2026 部署模式

- **CrewAI Flows** 适用于事件驱动的生产环境。
- **Agno** 无状态 FastAPI 适用于 Python 微服务。
- **Mastra** 服务器适配器（Express、Hono、Fastify、Koa）用于嵌入。
- **Pipecat Cloud / LiveKit Cloud** 用于托管语音（Lesson 22）。
- **Claude Managed Agents** 用于托管的长时异步场景。

### 可观测性是核心依赖

没有 OpenTelemetry GenAI 链路（Lesson 23）和 Langfuse/Phoenix/Opik 后端（Lesson 24），无法调试在第 40 步失败的多步 Agent。这在生产环境中不是可选项。它决定了"我们能快速定位问题"还是"我们只能从头回放并加日志"。

### 生产运行时失败场景

- **形态选错。** 为一个 5 分钟的任务选择请求-响应。用户挂断；Worker 堆积；重试叠加。
- **没有 DLQ。** 队列 Worker 没有死信队列。失败的任务从此消失。
- **后台工作不透明。** Agent 后台运行不导出链路。失败静默发生，直到用户报告才发现。
- **跳过持久状态。** 任何超过 30 秒且无法接受重启的任务都需要持久执行。

## 动手实现

`code/main.py` 是一个标准库的多形态演示：

- 请求-响应端点（普通函数）。
- 流式处理器（生成器）。
- 队列 Worker 含 DLQ。
- 事件触发注册表。
- Cron 形态调度器。

运行：

```bash
python3 code/main.py
```

输出：五条链路，分别展示同一任务在不同形态下的行为。同样的 Agent 逻辑，不同的外层封装。持久执行（第六种形态）在 Lesson 13 配合 LangGraph 检查点已有完整覆盖，此处不再重复。

## 用现成库

- **请求-响应** 用于聊天风格交互界面。
- **流式输出** 用于渐进式响应。
- **持久执行** 用于长时任务。
- **队列** 用于批处理/异步/长时任务。
- **事件驱动** 用于 Agent 响应式交互。
- **定时任务** 用于日常维护（记忆整合、评估、成本报告）。

## 产出

`outputs/skill-runtime-shape.md` 根据任务特征选择运行时形态，并明确可观测性需求。

## 练习

1. 将 Lesson 01 的 ReAct 循环移植到你的技术栈中的全部六种形态。哪种形态适合哪种产品形态？
2. 给队列演示添加 DLQ。模拟 10% 任务失败，观察 DLQ 大小。
3. 编写一个定时触发的评估 Agent，夜间对当天 Top 20 条链路运行评估。
4. 实现带背压的流式输出：如果客户端响应慢，暂停 Agent。这与轮次预算如何相互作用？
5. 阅读 Claude Managed Agents 文档。什么时候应该将自托管的长时 Agent 迁移到托管服务？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 请求-响应 | 同步 | 用户等待；仅适用于短任务 |
| 流式输出 | SSE / WS | 渐进输出；更好的用户体验；每块延迟可观测 |
| 持久执行 | 故障恢复 | 状态存档；从头步恢复 |
| 队列处理 | 后台任务 | 生产者 / Worker 池 / DLQ |
| 事件驱动 | 触发式 | Agent 响应外部事件 |
| DLQ | 死信队列 | 失败任务的停车场 |
| Claude Managed Agents | 托管运行时 | Anthropic 托管的长时异步，含缓存和压缩 |

## 延伸阅读

- [LangGraph 概述](https://docs.langchain.com/oss/python/langgraph/overview) — 持久执行详解
- [Claude Managed Agents 概述](https://platform.claude.com/docs/en/managed-agents/overview) — 托管长时异步
- [Anthropic，引入计算机使用](https://www.anthropic.com/news/3-5-models-and-computer-use) — "每任务数十到数百步"
- [AutoGen v0.4（微软研究院）](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — 角色模型故障隔离