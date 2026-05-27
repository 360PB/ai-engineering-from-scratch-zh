# MCP Apps —— 通过 `ui://` 的交互式 UI Resources

> 纯文本工具输出限制了 Agent 可以展示的内容。MCP Apps（SEP-1724，2026 年 1 月 26 日正式发布）让工具返回沙盒交互式 HTML，内联渲染在 Claude Desktop、ChatGPT、Cursor、Goose 和 VS Code 中。仪表板、表单、地图、3D 场景，全部通过一个扩展。本课走查 `ui://` 资源方案、`text/html;profile=mcp-app` MIME 类型、iframe 沙盒 postMessage 协议，以及允许服务器渲染 HTML 所带来的安全表面。

**类型：** 构建
**语言：** Python（标准库、UI 资源发出器）、HTML（示例应用）
**前置要求：** Phase 13 · 07（MCP 服务器）、Phase 13 · 10（resources）
**时间：** 约 75 分钟

## 学习目标

- 从工具调用返回 `ui://` resource 并设置正确的 MIME 和元数据。
- 用 `_meta.ui.resourceUri`、`_meta.ui.csp` 和 `_meta.ui.permissions` 声明工具的关联 UI。
- 实现 UI 到宿主通信的 iframe 沙盒 postMessage JSON-RPC。
- 应用 CSP 和权限策略默认设置，防御来自 UI 的攻击。

## 问题

2025 年代的 `visualize_timeline` 工具可以返回"这是按时间顺序组织的 14 个笔记：..."。那是一段话。用户实际想要交互式时间线。在 MCP Apps 之前，选项是：客户端特定的小部件 API（Claude artifacts、OpenAI Custom GPT HTML），或根本没有 UI。

MCP Apps（SEP-1724，2026 年 1 月 26 日发布）标准化了合约。工具结果包含一个 URI 为 `ui://...` 且 MIME 为 `text/html;profile=mcp-app` 的 `resource`。宿主在带有限制性 CSP 且除非明确授予否则无网络访问的沙盒 iframe 中渲染它。iframe 内的 UI 通过微小的 postMessage JSON-RPC 方言向宿主发布消息。

每个兼容客户端（Claude Desktop、ChatGPT、Goose、VS Code）以相同方式渲染相同的 `ui://` resource。一个服务器，一个 HTML 包，通用 UI。

## 概念

### `ui://` 资源方案

工具返回：

```json
{
  "content": [
    {"type": "text", "text": "Here is your notes timeline:"},
    {"type": "ui_resource", "uri": "ui://notes/timeline"}
  ],
  "_meta": {
    "ui": {
      "resourceUri": "ui://notes/timeline",
      "csp": {
        "defaultSrc": "'self'",
        "scriptSrc": "'self' 'unsafe-inline'",
        "connectSrc": "'self'"
      },
      "permissions": []
    }
  }
}
```

宿主然后对 `ui://notes/timeline` URI 调用 `resources/read` 并获取：

```json
{
  "contents": [{
    "uri": "ui://notes/timeline",
    "mimeType": "text/html;profile=mcp-app",
    "text": "<!doctype html>..."
  }]
}
```

### iframe 沙盒

宿主在沙盒 `<iframe>` 中渲染 HTML：

- `sandbox="allow-scripts allow-same-origin"`（或按服务器声明更严格）
- 通过响应头应用服务器声明的 CSP。
- 无来自宿主源的 cookies、localStorage。
- 网络访问限制为 CSP 中的 `connectSrc`。

### postMessage 协议

iframe 通过 `window.postMessage` 与宿主通信。一个微小的 JSON-RPC 2.0 方言：

始终将 `targetOrigin` 固定到对端的精确来源，并在接收端在处理任何载荷之前根据允许列表验证 `event.origin`。永远不要对此通道的任何一方使用 `"*"` —— 主体携带工具调用和资源读取。

```js
// iframe 到宿主  (固定到宿主来源)
window.parent.postMessage({
  jsonrpc: "2.0",
  id: 1,
  method: "host.callTool",
  params: { name: "notes_update", arguments: { id: "note-14", title: "..." } }
}, "https://host.example.com");

// 宿主到 iframe  (固定到 iframe 来源)
iframe.contentWindow.postMessage({
  jsonrpc: "2.0",
  id: 1,
  result: { content: [...] }
}, "https://iframe.example.com");

// 两侧的接收器
window.addEventListener("message", (event) => {
  if (event.origin !== "https://expected-peer.example.com") return;
  // 可以安全处理 event.data
});
```

UI 可以调用的可用宿主端方法：

- `host.callTool(name, arguments)` — 调用服务器工具。
- `host.readResource(uri)` — 读取 MCP resource。
- `host.getPrompt(name, arguments)` — 获取 prompt 模板。
- `host.close()` — 关闭 UI。

每个调用仍然通过 MCP 协议并继承服务器权限。

### 权限

`_meta.ui.permissions` 列表请求额外功能：

- `camera` — 访问用户相机（用于扫描文档 UI）。
- `microphone` — 语音输入。
- `geolocation` — 位置。
- `network:*` — 比单独 `connectSrc` 更广泛的网络访问。

每个权限是用户在 UI 渲染前看到的提示。

### 安全风险

iframe 中的 HTML 仍然是 HTML。新的攻击面：

- **通过 UI 的提示注入。** 恶意服务器 UI 可以显示看起来像系统消息的文本并欺骗用户。宿主渲染应明显区分服务器 UI 和宿主 UI。
- **通过 `connectSrc` 窃取。** 如果 CSP 允许 `connect-src: *`，UI 可以将数据发送到任何地方。默认应严格。
- **点击劫持。** UI 覆盖宿主 chrome。宿主必须防止 z-index 操作并强制执行不透明度规则。
- **窃取焦点。** UI 获取键盘焦点并捕获下一条消息。宿主必须拦截。

Phase 13 · 15 在 MCP 安全部分深入涵盖这些；本课介绍它们。

### `ui/initialize` 握手

iframe 加载后，通过 postMessage 发送 `ui/initialize`：

```json
{"jsonrpc": "2.0", "id": 0, "method": "ui/initialize",
 "params": {"theme": "dark", "locale": "en-US", "sessionId": "..."}}
```

宿主用能力和会话令牌响应。UI 在每个后续宿主调用上使用会话令牌。

### AppRenderer / AppFrame SDK 原语

ext-apps SDK 暴露两个便利原语：

- `AppRenderer`（服务器端）— 包装 React/Vue/Solid 组件并发出带正确 MIME 和元数据的 `ui://` resource。
- `AppFrame`（客户端）— 接收 resource，挂载 iframe，并调解 postMessage。

你可以使用这些或手写 HTML 和 JSON-RPC。

### 生态系统状态

MCP Apps 于 2026 年 1 月 26 日发布。截至 2026 年 4 月的客户端支持：

- **Claude Desktop。** 自 2026 年 1 月起完全支持。
- **ChatGPT。** 通过 Apps SDK 完全支持（相同的底层 MCP Apps 协议）。
- **Cursor。** Beta 版；通过设置启用。
- **VS Code。** 仅内部版本。
- **Goose。** 完全支持。
- **Zed、Windsurf。** 路线图。

生产中的服务器：仪表板、地图可视化、数据表、图表构建器、沙盒 IDE 预览。

## 使用它

`code/main.py` 用 `visualize_timeline` 工具扩展笔记服务器，该工具返回 `ui://notes/timeline` resource，加上对那个 URI 的 `resources/read` 处理程序，返回带有 SVG 时间线的完整 HTML 包。HTML 是标准库模板化的——无构建系统。postMessage 在 JS 注释中勾勒，因为标准库无法驱动浏览器。

要注意的点：

- 工具响应上的 `_meta.ui` 携带 resourceUri、CSP、permissions。
- HTML 渲染无需网络访问；所有数据都是内联的。
- JS 通过 `window.parent.postMessage` 调用 `host.callTool`（记录但在此标准库演示中为惰性）。

## 发布它

本课生成 `outputs/skill-mcp-apps-spec.md`。给定一个可能受益于交互式 UI 的工具，该 Skill 生成完整 MCP Apps 合约：`ui://` URI、CSP、permissions、postMessage 入口点，和安全检查清单。

## 练习

1. 运行 `code/main.py` 并检查发出的 HTML。直接在浏览器中打开 HTML；验证 SVG 渲染。然后勾勒 UI 将用来调用 `host.callTool("notes_update", ...)` 的 postMessage 合约。

2. 收紧 CSP：移除 `'unsafe-inline'` 并使用基于 nonce 的脚本策略。HTML 生成代码中什么改变了？

3. 添加第二个 UI resource `ui://notes/editor`，带一个用于就地编辑笔记的表单。当用户提交时，iframe 调用 `host.callTool("notes_update", ...)`。

4. 审计 UI 的攻击面。恶意服务器可以在哪里注入内容？iframe 沙盒防御什么而不防御什么？

5. 阅读 SEP-1724 规范并找出 MCP Apps SDK 中的一个能力，这个玩具实现没有使用。（提示：组件级状态同步。）

## 关键术语

| 术语 | 人们通常说 | 实际含义 |
|------|-----------|----------|
| MCP Apps | "交互式 UI Resources" | SEP-1724 扩展，2026-01-26 发布 |
| `ui://` | "应用 URI 方案" | UI 包的资源方案 |
| `text/html;profile=mcp-app` | "MIME 类型" | MCP App HTML 的内容类型 |
| Iframe sandbox（iframe 沙盒） | "渲染容器" | 带 CSP 和权限的浏览器沙盒 |
| postMessage JSON-RPC | "UI 到宿主线路" | 用于宿主调用的微小 postMessage 之上的 JSON-RPC 方言 |
| `_meta.ui` | "工具-UI 绑定" | 将工具结果链接到 UI resource 的元数据 |
| CSP | "内容安全策略" | 声明脚本、网络、样式的允许来源 |
| AppRenderer | "服务器 SDK 原语" | 将框架组件转换为 `ui://` resource |
| AppFrame | "客户端 SDK 原语" | 调解 postMessage 的 iframe 挂载辅助工具 |
| `ui/initialize` | "握手" | UI 到宿主的第一个 postMessage |

## 延伸阅读

- [MCP ext-apps — GitHub](https://github.com/modelcontextprotocol/ext-apps) — 参考实现和 SDK
- [MCP Apps 规范 2026-01-26](https://github.com/modelcontextprotocol/ext-apps/blob/main/specification/2026-01-26/apps.mdx) — 正式规范文档
- [MCP — Apps 扩展概述](https://modelcontextprotocol.io/extensions/apps/overview) — 高级文档
- [MCP 博客 — MCP Apps 发布](https://blog.modelcontextprotocol.io/posts/2026-01-26-mcp-apps/) — 2026 年 1 月发布帖
- [MCP Apps API 参考](https://apps.extensions.modelcontextprotocol.io/api/) — JSDoc 风格 SDK 参考