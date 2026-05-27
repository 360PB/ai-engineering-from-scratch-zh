# A2A——智能体对智能体协议

> Google 2025 年 4 月宣布 A2A；2026 年 4 月规范在 https://a2a-protocol.org/latest/specification/，150+ 组织支持。A2A 是 MCP 的水平补充（Lesson 13）：MCP 是垂直的（智能体 ↔ 工具），A2A 是对等的（智能体 ↔ 智能体）。它定义智能体卡片（发现）、带产物的任务（文本、结构化数据、视频）、不透明任务生命周期和认证。生产系统越来越多地将 MCP 与 A2A 配对。Google Cloud 在 2025-2026 年将 A2A 支持纳入 Vertex AI Agent Builder。

**类型：** 学习 + 构建
**语言：** Python（标准库，`http.server`，`json`）
**前置知识：** Phase 16 · 04（原语模型）
**时间：** 约75分钟

## 问题

你的智能体需要调用另一系统上的另一智能体。怎么做到？你可以暴露 HTTP 端点，定义自定义 JSON 模式，并希望另一方说它。每个智能体对变成自定义集成。

A2A 是该调用的通用有线协议。标准发现、标准任务模型、标准传输、标准产物。像 HTTP+REST 但智能体是一等公民。

## 概念

### 四个元素

**智能体卡片。** 在 `/.well-known/agent.json` 的 JSON 文档，描述智能体：名称、技能、端点、支持模态、认证要求。发现通过读取卡片发生。

```
GET https://agent.example.com/.well-known/agent.json
→ {
    "name": "code-review-agent",
    "skills": ["review-python", "review-typescript"],
    "endpoints": {
      "tasks": "https://agent.example.com/tasks"
    },
    "auth": {"type": "bearer"},
    "modalities": ["text", "structured"]
  }
```

**任务。** 工作单元。异步、有状态对象，带生命周期：`submitted → working → completed / failed / canceled`。客户端发送任务，轮询或订阅更新。

**产物。** 任务产生的结果类型。文本、结构化 JSON、图像、视频、音频。产物有类型所以不同模态是第一类。

**不透明生命周期。** A2A 不规定*远程智能体如何*解决任务。客户端看到状态转换和产物；实现自由选择框架/工具。

### MCP/A2A 分裂

- **MCP**（Lesson 13）：智能体 ↔ 工具。智能体通过 JSON-RPC 读/写工具服务器。默认无状态。
- **A2A**：智能体 ↔ 智能体。对等协议；双方都是带自己推理的智能体。

生产多智能体系统两者都用。 A2A 对等方调用 MCP 工具在端。分裂保持两个关注点干净。

### 发现流程

```
客户端                    智能体服务器
  ├──GET /.well-known/agent.json──>
  <──智能体卡片 JSON────────────
  ├──POST /tasks {skill, input}──>
  <──201 task_id, state=submitted
  ├──GET /tasks/{id}──────────────>
  <──state=working, 42% 完成────
  ├──GET /tasks/{id}──────────────>
  <──state=completed, 产物──
```

或用流式：`/tasks/{id}/events` SSE 订阅推送更新。

### 认证

A2A 支持三种常见模式：

- **Bearer token** — OAuth2 或 opaque。
- **mTLS** — 互相 TLS；组织互相证明身份。
- **签名请求** — 有效载荷上的 HMAC。

认证在智能体卡片中声明；客户端发现并遵守。

### 2026 年 4 月 150+ 组织

企业 adoption 推动了 A2A 规模。头条：A2A 成了企业智能体系统跨信任边界的方式。Google Cloud 运输了 Vertex AI Agent Builder A2A 支持；Microsoft Agent Framework 支持它；大多数主要框架（LangGraph、CrewAI、AutoGen）发布 A2A 适配器。

### A2A 胜出场景

- **跨组织调用。** 公司 A 的智能体调用公司 B 的智能体。没有 A2A，每对是自定义契约。
- **异构框架。** LangGraph 智能体调用 CrewAI 智能体调用自定义 Python 智能体。A2A 规范化。
- **类型化产物。** 视频结果、结构化 JSON、音频——都是第一类。
- **长运行任务。** 不透明生命周期 + 轮询使小时级任务直接。

### A2A 困难场景

- **延迟敏感微调用。** A2A 生命周期是异步的。亚毫秒智能体对智能体不适合；用直接 RPC。
- **紧耦合进程内智能体。** 如果两个智能体在同一 Python 进程中运行，A2A HTTP 往返是杀鸡用牛刀。
- **小团队。** 规范开销是真实的；内部-only 智能体可能不需要形式化。

### A2A vs ACP、ANP、NLIP

2024-2026 年出现了几个相关规范：

- **ACP**（IBM/Linux Foundation）— A2A 前身，范围更窄。
- **ANP**（智能体网络协议）— 对等发现为主，去中心化优先。
- **NLIP**（Ecma 自然语言交互协议，2025 年 12 月标准化）— 自然语言内容类型。

A2A 是 2026 年 4 月采用最多的对等协议。见 arXiv:2505.02279（Liu 等，"智能体互操作性协议调查"）比较。

## 构建

`code/main.py` 使用 `http.server` 和 JSON 实现 A2A 最小服务器和客户端。服务器：

- 暴露 `/.well-known/agent.json`，
- 接受 `POST /tasks`，
- 管理任务状态，
- 在 `GET /tasks/{id}` 返回产物。

客户端：
- 获取智能体卡片，
- 提交任务，
- 轮询直到完成，
- 读取产物。

运行：

```
python3 code/main.py
```

脚本在后台线程启动服务器，然后客户端针对它运行。你看到完整流程：发现、提交、轮询、产物。

## 使用

`outputs/skill-a2a-integrator.md` 设计 A2A 集成：智能体卡片内容、任务模式、认证选择、流式 vs 轮询。

## 交付

清单：

- **固定规范版本。** A2A 仍在演进；智能体卡片声明协议版本。
- **幂等任务创建。** 重复提交（网络重试）应产生一个任务。
- **产物模式。** 声明返回形状；消费者应验证。
- **速率限制 + 认证。** A2A 是公开面对的；应用标准 Web 安全。
- **失败任务死信。** 检查模式随时间识别重复失败类型。

## 练习

1. 运行 `code/main.py`。确认客户端发现服务器并接收正确产物。
2. 添加第二技能到服务器（例如"summarize"）。更新智能体卡片。写一个基于任务类型选择技能的客户端。
3. 实现 SSE 流式端点：`/tasks/{id}/events` 发出状态变化。客户端需要做什么不同？
4. 阅读 A2A 规范（https://a2a-protocol.org/latest/specification/）。识别本演示未实现的三个规范要求。
5. 比较 A2A（智能体卡片发现）与 MCP（服务器端能力列表通过 `listTools`）。自描述智能体和能力探测之间什么权衡？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| A2A | "智能体对智能体" | 跨系统调用智能体的对等协议。Google 2025。 |
| 智能体卡片 | "智能体名片" | `/.well-known/agent.json` 的 JSON，描述技能、端点、认证。 |
| 任务 | "工作单元" | 带生命周期的异步有状态对象；产物在完成时产生。 |
| 产物 | "结果" | 类型化输出：文本、结构化 JSON、图像、视频、音频。第一类媒体。 |
| 不透明生命周期 | "如何解决是智能体的事" | 客户端看到状态转换；服务器自由选择框架/工具。 |
| 发现 | "找到智能体" | `GET /.well-known/agent.json` 返回卡片。 |
| MCP vs A2A | "工具 vs 对等" | MCP：垂直智能体 ↔ 工具。A2A：水平智能体 ↔ 智能体。 |
| ACP / ANP / NLIP | "兄弟协议" | 相邻规范；A2A 是 2026 年最多采用。 |

## 延伸阅读

- [A2A 规范](https://a2a-protocol.org/latest/specification/)——规范
- [Google Developers 博客——A2A 公告](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/)——2025 年 4 月发布帖
- [A2A GitHub 仓库](https://github.com/a2aproject/A2A)——参考实现和 SDK
- [Liu 等——智能体互操作性协议调查](https://arxiv.org/html/2505.02279v1)——MCP、ACP、A2A、ANP 比较
