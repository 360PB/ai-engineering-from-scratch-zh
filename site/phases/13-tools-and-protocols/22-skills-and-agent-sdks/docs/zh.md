# Skills 和 Agent SDK —— Anthropic Skills、AGENTS.md、OpenAI Apps SDK

> MCP 说"存在哪些工具"。Skills 说"如何完成任务"。2026 年技术栈分层两者。Anthropic 的 Agent Skills（开放标准，2025 年 12 月）作为 SKILL.md 发货，带渐进披露。OpenAI 的 Apps SDK 是 MCP 加上小部件元数据。AGENTS.md（在 60,000+ 仓库中）在仓库根目录作为项目级 Agent 上下文。本课命名每个涵盖的内容并构建一个可在 Agent 间旅行的最小化 SKILL.md + AGENTS.md 包。

**类型：** 学习
**语言：** Python（标准库、SKILL.md 解析器和加载器）
**前置要求：** Phase 13 · 07（MCP 服务器）
**时间：** 约 45 分钟

## 学习目标

- 区分三层：AGENTS.md（项目上下文）、SKILL.md（可复用know-how）、MCP（工具）。
- 用 YAML frontmatter 和渐进披露编写 SKILL.md。
- 将技能以文件系统风格加载到 Agent 运行时。
- 用 MCP 服务器和 AGENTS.md 组合技能，使一个包在 Claude Code、Cursor 和 Codex 中工作。

## 问题

工程师将发布说明写作工作流提炼为多步提示："读取最新合并的 PR。按区域分组。总结每个。按团队的样式写变更日志条目。发布到 Slack 草稿。"他们把它放在 Notion 文档中供团队使用。

现在他们想从 Claude Code、Cursor 和 Codex CLI 中使用这个工作流。每个 Agent 加载指令的方式不同：Claude Code 斜杠命令、Cursor 规则、Codex `.codex.md`。工程师复制工作流三次并维护三份副本。

AGENTS.md 和 SKILL.md 一起修复了这个：

- **AGENTS.md** 放在仓库根目录。每个兼容 Agent 在会话开始时读取它。"这个项目如何运作？有哪些约定？哪些命令运行测试？"
- **SKILL.md** 是一个可移植包：YAML frontmatter（名称、描述）+ markdown 正文 + 可选资源。支持技能的 Agent 按需按名称加载它们。
- **MCP**（Phase 13 · 06-14）处理技能需要调用的工具。

三层，一个可移植 artifacts。

## 概念

### AGENTS.md

2025 年底发布，截至 2026 年 4 月被 60,000+ 仓库采用。仓库根目录的一个文件。格式：

```markdown
# Project: my-service

## Conventions
- TypeScript with strict mode.
- Use Pydantic for models on the Python side.
- Tests run with `pnpm test`.

## Build and run
- `pnpm dev` for local dev server.
- `pnpm build` for production bundle.
```

Agent 在会话开始时读取这个，并用它来校准其在该项目中的行为。2026 年的每个编码 Agent 都支持 AGENTS.md：Claude Code、Cursor、Codex、Copilot Workspace、opencode、Windsurf、Zed。

### SKILL.md 格式

Anthropic 的 Agent Skills（2025 年 12 月作为开放标准发布）：

```markdown
---
name: release-notes-writer
description: Write a changelog entry for the latest merged PRs following this project's style.
---

# Release notes writer

When invoked, run these steps:

1. List PRs merged since the last tag. Use `gh pr list --base main --state merged`.
2. Group by label: feature, fix, chore, docs.
3. For each PR in each group, write one line: `- <title> (#<num>)`.
4. Draft the release notes and stage them in CHANGELOG.md.

If the user says "ship", run `git tag vX.Y.Z` and `gh release create`.

## Notes

- Never include commits without a PR.
- Skip "chore" entries from the public changelog.
```

Frontmatter 声明技能的标识。体是技能加载时展示给模型的提示。

### 渐进披露

技能可以引用子资源，Agent 仅在需要时获取。示例：

```
skills/
  release-notes-writer/
    SKILL.md
    style-guide.md
    template.md
    scripts/
      generate.sh
```

SKILL.md 说"见 style-guide.md 获取样式规则。"Agent 仅在技能运行时拉取 style-guide.md。这避免了用模型可能不需要的细节膨胀提示。

### 文件系统发现

Agent 运行时扫描已知目录中的 SKILL.md 文件：

- `~/.anthropic/skills/*/SKILL.md`
- 项目 `./skills/*/SKILL.md`
- `~/.claude/skills/*/SKILL.md`

按文件夹名称和 frontmatter `name` 加载。Claude Code、Anthropic Claude Agent SDK 和 SkillKit（跨 Agent）都遵循此模式。

### Anthropic Claude Agent SDK

`@anthropic-ai/claude-agent-sdk`（TypeScript）和 `claude-agent-sdk`（Python）在会话开始时加载技能，在运行时中作为可调用"Agent"暴露。Agent 循环在用户调用时调度到技能。

### OpenAI Apps SDK

2025 年 10 月发布；直接在 MCP 上构建。统一了 OpenAI 之前的 Connectors 和 Custom GPT Actions 在单一开发者表面。Apps SDK 应用是：

- 一个 MCP 服务器（工具、资源、prompts）。
- 加上 ChatGPT UI 的小部件元数据。
- 加上可选的 MCP Apps `ui://` resource 用于交互表面。

相同协议，更丰富 UX。

### 通过 SkillKit 跨 Agent 可移植性

SkillKit 和类似跨 Agent 分发层工具将单一 SKILL.md 翻译为 32+ AI Agent（Claude Code、Cursor、Codex、Gemini CLI、OpenCode 等）的原生格式。单一真相来源；多个消费者。

### 三层栈

| 层 | 文件 | 加载时机 | 目的 |
|----|------|---------|------|
| AGENTS.md | 仓库根目录 | 会话开始 | 项目级约定 |
| SKILL.md | skills 目录 | 技能调用时 | 可复用工作流 |
| MCP 服务器 | 外部进程 | 需要工具时 | 可调用操作 |

全部组合：Agent 在会话开始时读取 AGENTS.md，用户调用技能，技能的指令包含 MCP 工具调用，Agent 经由 MCP 客户端调度。

## 使用它

`code/main.py` 发货一个标准库 SKILL.md 解析器和加载器。它在 `./skills/` 下发现技能，解析 YAML frontmatter 加 markdown 正文，并生成按技能名称键控的 dict。然后它模拟一个 Agent 循环，按名称调用 `release-notes-writer`。

要注意的点：

- YAML frontmatter 用最小化标准库解析器解析（无 `pyyaml` 依赖）。
- 技能体原样存储；Agent 在调用时将其 prepend 到系统提示。
- 通过 `read_subresource` 函数演示渐进披露，该函数按需拉取引用文件。

## 发布它

本课生成 `outputs/skill-agent-bundle.md`。给定一个工作流，该 Skill 生成组合 SKILL.md + AGENTS.md + MCP 服务器蓝图包，可在 Agent 间移植。

## 练习

1. 运行 `code/main.py`。在 `skills/` 下添加第二个技能并确认加载器选取它。

2. 为本课程仓库写一个 AGENTS.md。包括测试命令、样式约定和 Phase 13 心智模型。

3. 将团队内部文档中的多步工作流移植到 SKILL.md。验证它在 Claude Code 中加载。

4. 手动将技能翻译为 Cursor 和 Codex 的原生规则格式。计算格式之间的差异——这是 SkillKit 自动化的翻译表面。

5. 阅读 Anthropic Agent Skills 博客帖子。找出 Claude Agent SDK 中本课加载器未涵盖的一个功能。（提示：Agent 子调用。）

## 关键术语

| 术语 | 人们通常说 | 实际含义 |
|------|-----------|----------|
| SKILL.md | "技能文件" | YAML frontmatter 加 markdown 正文，由 Agent 运行时加载 |
| AGENTS.md | "仓库根 Agent 上下文" | 项目级约定文件，在会话开始时读取 |
| Progressive disclosure（渐进披露） | "惰性加载子资源" | 技能体引用仅在需要时拉取的文件 |
| Frontmatter | "顶部 YAML 块" | `---` 分隔符中的元数据（名称、描述） |
| Claude Agent SDK | "Anthropic 的技能运行时" | `@anthropic-ai/claude-agent-sdk`，加载技能并路由 |
| OpenAI Apps SDK | "MCP + 小部件元数据" | 在 MCP 之上加 ChatGPT UI 钩子的 OpenAI 开发者表面 |
| Skill discovery（技能发现） | "文件系统扫描" | 走查已知目录找 SKILL.md，按名称键控 |
| Cross-agent portability（跨 Agent 可移植性） | "一个技能多个 Agent" | 通过 SkillKit 风格工具将一个 SKILL.md 翻译为 32+ Agent |
| Agent Skill | "可移植 know-how" | MCP 工具概念之外的可复用任务模板 |
| Apps SDK | "MCP 加 ChatGPT UI" | 统一在 MCP 上的 Connectors 和 Custom GPT |

## 延伸阅读

- [Anthropic — Agent Skills 公告](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills) — 2025 年 12 月发布
- [Anthropic — Agent Skills 文档](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview) — SKILL.md 格式参考
- [OpenAI — Apps SDK](https://developers.openai.com/apps-sdk) — 基于 MCP 的 ChatGPT 开发者平台
- [agents.md](https://agents.md/) — AGENTS.md 格式和采用列表
- [Anthropic — anthropics/skills GitHub](https://github.com/anthropics/skills) — 官方技能示例