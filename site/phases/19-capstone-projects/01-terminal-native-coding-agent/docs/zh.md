# 顶点课 01 — 终端原生编码 Agent

> 到 2026 年，编码 Agent 的形态已经定型。一个 TUI 承载体、一个有状态的计划、一个沙盒工具集、一个"计划→行动→观察→恢复"的循环。Claude Code、Cursor 3、OpenCode 远看都一模一样。本顶点课要求你从头构建一个——CLI 输入，PR 输出——并在 SWE-bench Pro 上对照 mini-swe-agent 和 Live-SWE-agent 做评测。你会明白为什么最难的部分不是模型调用，而是工具循环、沙盒和 50 轮运行的成本天花板。

**类型：** 顶点课
**语言：** TypeScript / Bun（承载体），Python（评测脚本）
**前置要求：** Phase 11（LLM 工程）、Phase 13（工具与协议）、Phase 14（Agent）、Phase 15（自主系统）、Phase 17（基础设施）
**涉及的 Phase：** P0 · P5 · P7 · P10 · P11 · P13 · P14 · P15 · P17 · P18
**时间：** 35 小时

## 问题

编码 Agent 在 2026 年成为主导的 AI 应用类别。Claude Code（Anthropic）、Cursor 3 搭配 Composer 2 和 Agent Tabs（Cursor）、Amp（Sourcegraph）、OpenCode（112k stars）、Factory Droids、Google Jules 都交付了同一套架构的变体：一个终端承载体、一个授权工具集、一个沙盒、一个围绕前沿模型构建的"计划-执行-观察"循环。前沿很窄——Live-SWE-agent 在 SWE-bench Verified 上达到 79.2%（使用 Opus 4.5）——但工程技艺很广。大多数失败模式不是模型犯错，而是工具循环不稳定、上下文污染、Token 成本失控、破坏性文件系统操作。

你无法从外部推断这些 Agent。你必须亲手构建一个，看着它在第 47 轮时崩溃——ripgrep 返回了 8MB 的匹配结果——然后重建截断层。这才是本顶点课的意义。

## 概念

承载体有四个面。**计划（Plan）** 维护一个 TodoWrite 风格的状态对象，模型每轮重写它。**行动（Act）** 分派工具调用（read、edit、run、search、git）。**观察（Observe）** 捕获 stdout/stderr/exit code，截断后把摘要反馈回去。**恢复（Recover）** 处理工具错误，不撑爆上下文窗口，也不无限循环。2026 年的形态多了一个要素：**钩子（Hooks）**。`PreToolUse`、`PostToolUse`、`SessionStart`、`SessionEnd`、`UserPromptSubmit`、`Notification`、`Stop`、`PreCompact`——可配置的扩展点，运营者在这些位置注入策略、遥测和护栏。

沙盒使用 E2B 或 Daytona。每个任务在一个全新的 devcontainer 中运行，git worktree 以读写方式挂载。承载体从不触碰主机文件系统。工作树在成功或失败后销毁。成本控制在三层执行：每轮 Token 上限、会话美元预算、硬性轮次上限（通常 50 轮）。可观测层是 OpenTelemetry span，带 GenAI 语义约定，发送到自托管的 Langfuse。

## 架构

```
  user CLI  ->  harness (Bun + Ink TUI)
                  |
                  v
           plan / act / observe loop  <--->  Claude Sonnet 4.7 / GPT-5.4-Codex / Gemini 3 Pro
                  |                          (via OpenRouter, model-agnostic)
                  v
           tool dispatcher (MCP StreamableHTTP client)
                  |
     +------------+------------+----------+
     v            v            v          v
  read/edit    ripgrep     tree-sitter   git/run
     |            |            |          |
     +------------+------------+----------+
                  |
                  v
           E2B / Daytona sandbox  (worktree isolated)
                  |
                  v
           hooks: Pre/Post, Session, Prompt, Compact
                  |
                  v
           OpenTelemetry -> Langfuse (spans, tokens, $)
                  |
                  v
           PR via GitHub app
```

## 技术栈

- 承载体运行时：Bun 1.2 + Ink 5（React in terminal）
- 模型接入：OpenRouter 统一 API，支持 Claude Sonnet 4.7、GPT-5.4-Codex、Gemini 3 Pro、Opus 4.5（用于最难任务）
- 工具传输：Model Context Protocol StreamableHTTP（MCP 2026 修订版）
- 沙盒：E2B sandboxes（JS SDK）或 Daytona devcontainers
- 代码搜索：ripgrep 子进程，tree-sitter 解析器支持 17 种语言（预编译）
- 隔离：`git worktree add` per task，成功/失败时清理
- 评测承载体：SWE-bench Pro（验证子集）+ Terminal-Bench 2.0 + 自定义 30 题 holdout
- 可观测：OpenTelemetry SDK 配合 `gen_ai.*` semconv → 自托管 Langfuse
- PR 推送：GitHub App 配合细粒度 Token，范围限定在目标仓库

## 构建步骤

1. **TUI 和命令循环。** 用 Ink 脚手架一个 Bun 项目。接受 `agent run <repo> "<task>"`。打印分栏视图：计划窗格（顶部）、工具调用流（中）、Token 预算（底部）。Ctrl-C 取消时在退出前触发 `SessionEnd` 钩子。

2. **计划状态。** 定义类型化的 TodoWrite schema（pending/in_progress/done 项目含注释）。模型每轮把完整状态作为工具调用重写——不允许增量变更。将计划持久化到 `.agent/state.json`，以便崩溃后恢复。

3. **工具面。** 定义六个工具：`read_file`、`edit_file`（含 diff 预览）、`ripgrep`、`tree_sitter_symbols`、`run_shell`（含超时）、`git`（status/diff/commit/push）。通过 MCP StreamableHTTP 暴露，使承载体与传输层解耦。每个工具返回截断后的输出（每次调用上限 4k Token）。

4. **沙盒包装。** 每个任务启动一个 E2B 沙盒。`git worktree add -b agent/$TASK_ID` 一个新分支。所有工具调用在沙盒内执行。主机文件系统不可达。

5. **钩子。** 实现全部 8 种 2026 钩子类型。至少接入四个用户编写的钩子：(a) `PreToolUse` 破坏性命令守卫，阻止 worktree 外的 `rm -rf`；(b) `PostToolUse` Token 计数；(c) `SessionStart` 预算初始化；(d) `Stop` 写入最终 trace bundle。

6. **评测循环。** 克隆 SWE-bench Pro Python 子集（30 题）。在每个题目上运行你的承载体。与 mini-swe-agent（最小基线）对比 pass@1、每任务轮数、每任务美元成本。将结果写入 `eval/results.jsonl`。

7. **成本控制。** 硬性截断：50 轮、200k 上下文、每任务 $5。在 150k 标记处触发 `PreCompact` 钩子，用较小模型（Haiku 4.5）将早期轮次压缩成先验状态块，释放空间给新观察而不丢失计划。

8. **PR 推送。** 成功后，最后一步是 `git push` + GitHub API 调用，以计划内容和 diff 摘要为正文打开 PR。

## 使用示例

```
$ agent run ./my-repo "Fix the race condition in worker.rs"
[plan]  1 locate worker.rs and enumerate mutex uses
        2 identify shared state under contention
        3 propose fix, verify tests
[tool]  ripgrep mutex.*lock -t rust           (44 matches, truncated)
[tool]  read_file src/worker.rs 120..180
[tool]  edit_file src/worker.rs (+8 -3)
[tool]  run_shell cargo test worker::          (passed)
[plan]  1 done · 2 done · 3 done
[done]  PR opened: #482   turns=9   tokens=38k   cost=$0.41
```

## 交付标准

交付物位于 `outputs/skill-terminal-coding-agent.md`。给定一个仓库路径和任务描述，它在沙盒中运行完整的计划-行动-观察循环，返回一个 PR URL 和 trace bundle。评分标准如下：

| 权重 | 指标 | 测量方式 |
|:-:|---|---|
| 25 | SWE-bench Pro pass@1 vs 基线 | 你的承载体 vs mini-swe-agent 在 30 道匹配的 Python 题上 |
| 20 | 架构清晰度 | Plan/Act/Observe 分离、钩子面、工具 schema——对照 Live-SWE-agent 布局审查 |
| 20 | 安全性 | 沙盒逃逸测试、授权提示、破坏性命令守卫通过红队测试 |
| 20 | 可观测性 | Trace 完整性（100% 工具调用有 span）、每轮 Token 计数 |
| 15 | 开发者体验 | 冷启动 < 2s、崩溃后恢复计划、Ctrl-C 在工具中途干净取消 |
| **100** | | |

## 练习

1. 将底模从 Claude Sonnet 4.7 换成 Qwen3-Coder-30B（vLLM 自托管）。对比 pass@1 和每任务美元成本。报告开放模型在哪方面表现不足。

2. 增加一个 `reviewer` 子 Agent，在 PR 推送前读取 diff 并请求修改循环。评测误报审查是否使 SWE-bench 通过率低于单 Agent 基线（提示：通常是的）。

3. 压力测试沙盒：写一个尝试 `curl` 外部 URL 的任务和一个尝试写入 worktree 之外的任务。确认两者都被 PreToolUse 钩子拦截。记录尝试。

4. 用较小模型（Haiku 4.5）实现 `PreCompact` 压缩。评测 3 倍压缩下计划保真度损失了多少。

5. 将 MCP StreamableHTTP 传输换成 stdio。评测冷启动和单次调用延迟。对于仅本地使用的场景选择更优方案。

## 关键术语

| 术语 | 别人怎么称呼 | 实际含义 |
|------|-----------------|------------------------|
| Harness | "Agent 循环" | 包裹模型的代码，负责分发工具、维护计划状态、执行预算限制 |
| Hook | "Agent 事件监听器" | 用户编写的脚本，在承载体八个生命周期事件之一触发时运行 |
| Worktree | "Git 沙盒" | 链接到独立路径的 git checkout；可丢弃而不影响主克隆 |
| TodoWrite | "计划状态" | 类型化的 pending/in_progress/done 项目列表，模型每轮重写 |
| StreamableHTTP | "MCP 传输层" | 2026 MCP 修订版：持久 HTTP 连接 + 双向流；取代 SSE |
| Token 天花板 | "上下文预算" | 每轮或每会话的输入+输出 Token 上限；触发压缩或终止 |
| pass@1 | "单次尝试通过率" | 第一次运行（不重试、不偷看测试集）解决 SWE-bench 任务的比例 |

## 延伸阅读

- [Claude Code 文档](https://docs.anthropic.com/en/docs/claude-code) — Anthropic 提供的参考承载体
- [Cursor 3 更新日志](https://cursor.com/changelog) — Agent Tabs 和 Composer 2 产品说明
- [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) — SWE-bench 承载体对比的最小基线
- [Live-SWE-agent](https://github.com/OpenAutoCoder/live-swe-agent) — 使用 Opus 4.5 在 SWE-bench Verified 上达 79.2%
- [OpenCode](https://opencode.ai) — 开放承载体，112k stars
- [SWE-bench Pro 排行榜](https://www.swebench.com) — 本顶点课瞄准的评测基准
- [Model Context Protocol 2026 路线图](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — StreamableHTTP、能力元数据
- [OpenTelemetry GenAI 语义约定](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 工具调用和 Token 使用的 span schema