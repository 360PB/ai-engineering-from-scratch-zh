# Roots 和 Elicitation —— 作用域和飞行中用户输入

> 硬编码路径在用户打开不同项目时失败。预填充的工具参数在用户规格不足时失败。Roots 将服务器作用域限制为用户控制的一组 URI；elicitation 在工具调用中途暂停，通过表单或 URL 请求用户提供结构化输入。两个客户端原语，修复两个常见 MCP 失败模式。SEP-1036（URL 模式 elicitation，2025-11-25）在 2026 年 H1 期间是实验性的——依赖前检查 SDK 版本。

**类型：** 构建
**语言：** Python（标准库、roots + elicitation 演示）
**前置要求：** Phase 13 · 07（MCP 服务器）
**时间：** 约 45 分钟

## 学习目标

- 声明 `roots` 并响应 `notifications/roots/list_changed`。
- 将服务器文件操作限制在声明根集内的 URI。
- 使用 `elicitation/create` 在工具调用中途请求用户确认或结构化输入。
- 在表单模式和 URL 模式 elicitation 之间选择（后者是实验性的；已注明漂移风险）。

## 问题

笔记 MCP 服务器在生产中会遇到的两种具体失败。

**路径假设失效。** 服务器针对 `~/notes` 编写。不同机器上笔记在 `~/Documents/Notes` 的用户收到一个静默失败（未找到文件）或更糟的、写入错误位置的工具调用。

**用户会知道的缺失参数。** 用户要求"删除旧的 TPS 报告笔记"。模型调用 `notes_delete(title: "TPS report")` 但有三个匹配的笔记（2023、2024、2025）。工具无法猜测。返回"模糊"很烦人；在所有三个上运行是灾难性的。

Roots 修复第一个：客户端在 `initialize` 时声明服务器可访问的 URI 集合。Elicitation 修复第二个：服务器暂停工具调用并发送 `elicitation/create` 请求用户选择哪个。

## 概念

### Roots

客户端在 `initialize` 时声明根列表：

```json
{
  "capabilities": {"roots": {"listChanged": true}}
}
```

服务器然后可以调用 `roots/list`：

```json
{"roots": [{"uri": "file:///Users/alice/Documents/Notes", "name": "Notes"}]}
```

服务器必须将 roots 视为边界：任何对根集之外文件的读取或写入都会被拒绝。这不是由客户端强制执行的（服务器仍然是用户信任的代码），但符合规范的服务器会遵守。

当用户添加或删除根时，客户端发送 `notifications/roots/list_changed`。服务器重新调用 `roots/list` 并更新其边界。

### 为什么 roots 是客户端原语

Roots 由客户端声明，因为它们代表用户的同意模型。用户告诉 Claude Desktop"给这个笔记服务器访问这两个目录的权限"。服务器无法扩大该作用域。

### Elicitation：表单模式默认

`elicitation/create` 接受表单 schema 加自然语言提示：

```json
{
  "method": "elicitation/create",
  "params": {
    "message": "Delete 'TPS report'? Multiple notes match; pick one.",
    "requestedSchema": {
      "type": "object",
      "properties": {
        "note_id": {
          "type": "string",
          "enum": ["note-3", "note-7", "note-14"]
        },
        "confirm": {"type": "boolean"}
      },
      "required": ["note_id", "confirm"]
    }
  }
}
```

客户端渲染表单，收集用户答案，返回：

```json
{
  "action": "accept",
  "content": {"note_id": "note-14", "confirm": true}
}
```

三种可能的操作：`accept`（用户填写）、`decline`（用户关闭）、`cancel`（用户中止整个工具调用）。

表单 schema 是扁平的——嵌套对象在 v1 中不支持。SDK 通常拒绝比单层更复杂的内容。

### Elicitation：URL 模式（SEP-1036，实验性）

2025-11-25 新增。不是 schema，服务器发送 URL：

```json
{
  "method": "elicitation/create",
  "params": {
    "message": "Sign in to GitHub",
    "url": "https://github.com/login/oauth/authorize?client_id=..."
  }
}
```

客户端在浏览器中打开 URL，等待完成，当用户返回时返回。对于 OAuth 流程、支付授权和文档签名（表单不足时）有用。

漂移风险提示：SEP-1036 响应形状仍在调整；一些 SDK 返回回调 URL，其他返回完成 token。在生产中使用 URL 模式前阅读 SDK 发布说明。

### 何时使用 elicitation

- 破坏性操作前的用户确认（destructive hint + elicitation）。
- 消歧（N 个匹配中选一个）。
- 首次运行设置（API 密钥、目录、首选项）。
- OAuth 风格流程（URL 模式）。

### 何时不用 elicitation

- 填充模型可以用散文询问的工具的必需参数。使用正常重新提示，而非 elicitation 对话框。
- 高频调用。Elicitation 中断对话；不要在循环内触发它。
- 服务器可以在事后验证的任何内容。验证，返回错误，让模型在文本中询问用户。

### 人在回路桥

Elicitation 加上采样共同启用 MCP 的"人在回路"模型。服务器的 Agent 循环可以暂停等待用户输入（elicitation）或模型推理（采样）。Phase 13 · 11 涵盖采样；本课涵盖 elicitation。将它们放在一起获得完整的中环控制。

## 使用它

`code/main.py` 用以下内容扩展笔记服务器：

- `roots/list` 响应，服务器在根列表变更通知后重新查询。
- 一个 `notes_delete` 工具，在多个笔记匹配时使用 `elicitation/create` 消歧。
- 一个 `notes_setup` 工具，使用 URL 模式 elicitation 打开首次运行配置页面（模拟）。
- 一个边界检查，拒绝对声明根之外 URI 的操作。

演示运行三个场景：快乐路径（一个匹配）、消歧（三个匹配，elicitation 触发）、根外写入（拒绝）。

## 发布它

本课生成 `outputs/skill-elicitation-form-designer.md`。给定一个可能需要用户确认或消歧的工具，该 Skill 设计 elicitation 表单 schema 和消息模板。

## 练习

1. 运行 `code/main.py`。触发消歧路径；确认模拟用户答案被路由回工具。

2. 添加一个新工具 `notes_archive`，每次都需要 elicitation 确认（destructive hint）。检查 UX：这与模型在文本中重新询问相比如何？

3. 为首次运行 OAuth 流程实现 URL 模式 elicitation。注意漂移风险并添加 SDK 版本保护。

4. 扩展 `roots/list` 处理：当通知到达时，服务器应原子性地重新读取并重新扫描可能已超出范围的打开文件句柄。

5. 阅读 GitHub 上 SEP-1036 问题讨论线程。找出一个影响服务器应如何处理 URL 模式回调的开放问题。

## 关键术语

| 术语 | 人们通常说 | 实际含义 |
|------|-----------|----------|
| Root | "同意边界" | 客户端允许服务器访问的 URI |
| `roots/list` | "服务器请求作用域" | 客户端返回当前根集 |
| `notifications/roots/list_changed` | "用户变更了作用域" | 客户端发出根集已变更的信号 |
| Elicitation | "在调用中询问用户" | 服务器发起的结构化用户输入请求 |
| `elicitation/create` | "该方法" | elicitation 请求的 JSON-RPC 方法 |
| Form mode（表单模式） | "Schema 驱动表单" | 扁平 JSON Schema 在客户端 UI 中渲染为表单 |
| URL mode（URL 模式） | "浏览器重定向" | SEP-1036 实验性；在浏览器中打开 URL 并等待 |
| `accept` / `decline` / `cancel` | "用户响应结果" | 服务器处理的三种分支 |
| Disambiguation（消歧） | "选一个" | 当工具有 N 个候选时常见的 elicitation 用例 |
| Flat form（扁平表单） | "仅顶层属性" | Elicitation schema 不能嵌套 |

## 延伸阅读

- [MCP — 客户端 roots 规范](https://modelcontextprotocol.io/specification/draft/client/roots) — 规范 roots 参考
- [MCP — 客户端 elicitation 规范](https://modelcontextprotocol.io/specification/draft/client/elicitation) — 规范 elicitation 参考
- [Cisco — MCP elicitation、结构化内容、OAuth 增强中的新内容](https://blogs.cisco.com/developer/whats-new-in-mcp-elicitation-structured-content-and-oauth-enhancements) — 2025-11-25 附加走查
- [MCP — GitHub SEP-1036](https://github.com/modelcontextprotocol/modelcontextprotocol) — URL 模式 elicitation 提案（实验性，漂移风险）
- [The New Stack — elicitation 如何将人在回路引入 AI 工具](https://thenewstack.io/how-elicitation-in-mcp-brings-human-in-the-loop-to-ai-tools/) — UX 走查