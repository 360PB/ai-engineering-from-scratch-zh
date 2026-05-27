# A2A —— Agent 间协议

> MCP 是 Agent 到工具。A2A（Agent2Agent）是 Agent 到 Agent——一个开放协议，让建立在不同框架上的不透明 Agent 协作。Google 于 2025 年 4 月发布，2025 年 6 月捐赠给 Linux 基金会，2026 年 4 月达到 v1.0，有 150+ 支持者，包括 AWS、Cisco、Microsoft、Salesforce、SAP 和 ServiceNow。它吸收了 IBM 的 ACP 并添加了 AP2 支付扩展。本课走查 Agent Card、任务生命周期和两个传输绑定。

**类型：** 构建
**语言：** Python（标准库、Agent Card + 任务 harness）
**前置要求：** Phase 13 · 06（MCP 基础）、Phase 13 · 08（MCP 客户端）
**时间：** 约 75 分钟

## 学习目标

- 区分 Agent 到工具（MCP）和 Agent 到 Agent（A2A）用例。
- 在 `/.well-known/agent.json` 发布 Agent Card，带技能和端点元数据。
- 走查任务生命周期（submitted → working → input-required → completed / failed / canceled / rejected）。
- 使用带 Parts（文本、文件、数据）的消息和 Artifacts 作为输出。

## 问题

客户服务 Agent 需要将报告撰写委托给专业写作 Agent。A2A 前的选项：

- 自定义 REST API。可以工作，但每对配对都是定制。
- 共享代码库。要求两个 Agent 运行相同框架。
- MCP。不合适：MCP 用于调用工具，而非两个 Agent 在保留各自不透明内部推理的情况下协作。

A2A 填补了这个空白。它将交互建模为一个 Agent 向另一个发送任务，带生命周期、消息和 artifacts。被调用 Agent 的内部状态保持不透明——调用方只看到任务状态转换和最终输出。

A2A 是"让跨框架 Agent 相互对话"的协议。它不替换 MCP；两者是互补的。

## 概念

### Agent Card

每个符合 A2A 的 Agent 在 `/.well-known/agent.json` 发布一张卡：

```json
{
  "schemaVersion": "1.0",
  "name": "research-agent",
  "description": "Summarizes academic papers and drafts citations.",
  "url": "https://research.example.com/a2a",
  "version": "1.2.0",
  "skills": [
    {
      "id": "summarize_paper",
      "name": "Summarize a paper",
      "description": "Read a paper PDF and produce a 3-paragraph summary.",
      "inputModes": ["text", "file"],
      "outputModes": ["text", "artifact"]
    }
  ],
  "capabilities": {"streaming": true, "pushNotifications": true}
}
```

发现是基于 URL 的：获取卡，了解 A2A 端点的 URL，列举技能。

### 签名 Agent Card（AP2）

AP2 扩展（2025 年 9 月）向 Agent Card 添加加密签名。发布者用 JWT 签署自己的卡；消费者验证。防止冒充。

### 任务生命周期

```
submitted -> working -> completed | failed | canceled | rejected
             -> input_required -> working (通过消息循环)
```

客户端用 `tasks/send` 发起。被调用 Agent 转换状态；客户端通过 SSE 订阅状态更新或轮询。

### 消息和 Parts

消息携带一个或多个 Parts：

- `text` — 纯内容。
- `file` — 带 mimeType 的 base64 blob。
- `data` — 类型化 JSON 载荷（被调用 Agent 的结构化输入）。

示例：

```json
{
  "role": "user",
  "parts": [
    {"type": "text", "text": "Summarize this paper."},
    {"type": "file", "file": {"name": "paper.pdf", "mimeType": "application/pdf", "bytes": "..."}},
    {"type": "data", "data": {"targetLength": "3 paragraphs"}}
  ]
}
```

### Artifacts

输出是 Artifacts，而非原始字符串。Artifact 是一个命名的类型化输出：

```json
{
  "name": "summary",
  "parts": [{"type": "text", "text": "..."}],
  "mimeType": "text/markdown"
}
```

Artifacts 可以作为块流式传输。调用方积累。

### 两个传输绑定

1. **JSON-RPC over HTTP。** `/a2a` 端点，POST 发请求，可选 SSE 用于流式传输。默认绑定。
2. **gRPC。** 用于 gRPC 本地的企业环境。

两种绑定携带相同的逻辑消息形状。

### 不透明保持

关键设计原则：被调用 Agent 的内部状态是不透明的。调用方看到任务状态和 artifacts。被调用 Agent 的思维链、其工具调用、其子 Agent 委托——全部不可见。这与 MCP 不同，MCP 中工具调用是透明的。

理由：A2A 使竞争对手能够在不暴露内部的情况下协作。A2A 可以是"调用此客户服务 Agent"，而调用方无需了解该 Agent 如何实现服务。

### 时间线

- **2025-04-09。** Google 宣布 A2A。
- **2025-06-23。** 捐赠给 Linux 基金会。
- **2025-08。** 吸收 IBM 的 ACP。
- **2025-09。** AP2 扩展（Agent 支付）发布。
- **2026-04。** v1.0 发布，150+ 支持组织。

### 与 MCP 的关系

| 维度 | MCP | A2A |
|------|-----|-----|
| 用例 | Agent 到工具 | Agent 到 Agent |
| 不透明 | 透明工具调用 | 不透明内部推理 |
| 典型调用方 | Agent 运行时 | 另一个 Agent |
| 状态 | 工具调用结果 | 带生命周期的任务 |
| 授权 | OAuth 2.1（Phase 13 · 16） | JWT 签名 Agent Card（AP2） |
| 传输 | Stdio / Streamable HTTP | JSON-RPC over HTTP / gRPC |

当你想要调用特定工具时使用 MCP。当你想将整个任务委托给另一个 Agent 时使用 A2A。许多生产系统同时使用两者：Agent 使用 MCP 作为其工具层，使用 A2A 作为其协作层。

## 使用它

`code/main.py` 实现一个最小化 A2A harness：研究 Agent 发布其卡，写作者 Agent 接收带部分（包括 PDF 和文本指令）的 `tasks/send`，转换 through working → input_required → working → completed，并返回文本 artifact。全部标准库；使用内存传输以专注于消息形状。

要注意的点：

- Agent Card JSON 形状。
- 任务 id 分配和状态转换。
- 混合类型部分的消息。
- 任务中期的 input-required 分支。
- 完成时的 Artifact 返回。

## 发布它

本课生成 `outputs/skill-a2a-agent-spec.md`。给定一个应可被其他 Agent 调用的新 Agent，该 Skill 生成 Agent Card JSON、技能 schema 和端点蓝图。

## 练习

1. 运行 `code/main.py`。追踪完整任务生命周期，包括被调用 Agent 要求澄清时 input-required 暂停。

2. 添加签名 Agent Card。用 HMAC 对卡的标准 JSON 进行签名。写验证器并确认变异卡失败。

3. 实现任务流式传输：写作者 Agent 通过 SSE 发出三个增量 artifact 块，调用方积累它们。

4. 设计一个包装 MCP 服务器的 A2A Agent。将每个 MCP 工具映射到 A2A 技能。注意权衡——损失了什么不透明性？

5. 阅读 A2A v1.0 公告并找出截至 2026 年 4 月任何框架尚未实现的一个功能。（提示：它涉及多跳任务委托。）

## 关键术语

| 术语 | 人们通常说 | 实际含义 |
|------|-----------|----------|
| A2A | "Agent 间协议" | 不透明 Agent 协作的开放协议 |
| Agent Card | "`.well-known/agent.json`" | 发布的描述 Agent 技能和端点的元数据 |
| Skill | "可调用单元" | Agent 支持的命名操作（类似于 MCP 工具） |
| Task | "委托单元" | 带生命周期和最终 artifact 的工作项 |
| Message | "任务输入" | 携带 Parts（文本、文件、数据） |
| Part | "类型化块" | 消息的 `text` / `file` / `data` 元素 |
| Artifact | "任务输出" | 完成时返回的命名、类型化输出 |
| AP2 | "Agent 支付协议" | 用于信任和支付的签名 Agent Card 扩展 |
| Opacity（不透明） | "黑盒协作" | 被调用 Agent 的内部对调用方隐藏 |
| Input-required | "任务暂停" | 当 Agent 需要更多信息时的生命周期状态 |

## 延伸阅读

- [a2a-protocol.org](https://a2a-protocol.org/latest/) — 规范 A2A 规范
- [a2aproject/A2A — GitHub](https://github.com/a2aproject/A2A) — 参考实现和 SDK
- [Linux 基金会 — A2A 启动新闻稿](https://www.linuxfoundation.org/press/linux-foundation-launches-the-agent2agent-protocol-project-to-enable-secure-intelligent-communication-between-ai-agents) — 2025 年 6 月治理转移
- [Google Cloud — A2A 协议升级](https://cloud.google.com/blog/products/ai-machine-learning/agent2agent-protocol-is-getting-an-upgrade) — 路线图和合作伙伴势头
- [Google Dev — A2A 1.0 里程碑](https://discuss.google.dev/t/the-a2a-1-0-milestone-ensuring-and-testing-backward-compatibility/352258) — v1.0 发布说明和向后兼容指南