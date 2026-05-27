# Capstone 16 — GitHub Issue 到 PR 的自主 Agent

> AWS Remote SWE Agents、Cursor Background Agents、OpenAI Codex cloud 和 Google Jules 都交付了相同的2026年产品形态：标注 issue，得到 PR。在云沙箱中运行 Agent，验证测试通过，发布带理由的 review-ready PR。难点是自动复现仓库的构建环境、防止凭据泄露、强制每仓库预算，以及确保 Agent 不能 force-push。毕业项目是构建自托管版本，并在成本和通过率上与托管替代品对比。

**类型：** 毕业项目
**语言：** Python（Agent），TypeScript（GitHub App），YAML（Actions）
**前置知识：** Phase 11（LLM工程）、Phase 13（工具）、Phase 14（Agent工程）、Phase 15（自主系统）、Phase 17（基础设施）
**涉及阶段：** P11 · P13 · P14 · P15 · P17
**时长：** 30小时

## 问题

异步云编码 Agent 是与交互式编码 Agent（capstone 01）不同的产品类别。UX 是一个 GitHub 标签。你给 issue 打标签 `@agent fix this`，一个 worker 在云沙箱中启动，clone 仓库，运行测试，编辑文件，验证，通过 Agent 理由作为 body 发布 PR。无交互循环，无终端。AWS Remote SWE Agents、Cursor Background Agents、OpenAI Codex cloud、Google Jules 和 Factory Droids 都收敛到这一点。

工程挑战是具体的：环境复现（Agent 必须从零构建仓库，没有缓存的 dev 镜像）、不稳定测试（必须重跑或隔离）、凭据范围（最小细粒度权限的 GitHub App）、每仓库每天预算强制，以及 no-force-push 策略。毕业项目测量通过率、成本和安全性 vs 托管替代品。

## 核心概念

触发是 GitHub webhook（issue 标签或 PR 评论）。调度器将工作排队到 ECS Fargate 或 Lambda。Worker 将仓库拉入 Daytona 或 E2B 沙箱，带从仓库推断的通用 Dockerfile。Agent 对 Claude Opus 4.7 或 GPT-5.4-Codex 运行 mini-swe-agent 或 SWE-agent v2 循环。它迭代：读代码、提案修复、应用 patch、运行测试。

验证是门控步骤。在 PR 打开前，完整 CI 必须在沙箱中通过。计算覆盖率变化；如果负超过阈值，PR 打开但标记 `needs-review`。Agent 将理由作为 PR 描述发布，外加一个 `@agent` thread，reviewer 可以 ping 它进行跟进。

安全通过两个不同的 GitHub 表面限定范围：App 提供带 `workflows: read` 和窄 repo contents/PR 作用域的短期安装 token；分支保护（唯一能做到这一点的表面）强制"不直接写 `main`"和"no force-push"——App 不在 bypass 列表中。`.github/workflows` 的路径范围只读访问不是真实的 GitHub App 原语，所以 Agent 的文件编辑白名单必须在 worker 层强制执行。每仓库每天的预算上限在调度器层强制（如，每仓库每天最多5个 PR，每个 PR $20）。

## 架构

```
GitHub issue 标记 `@agent fix` 或 PR 评论
            |
            v
    GitHub App webhook -> AWS Lambda 调度器
            |
            v
    ECS Fargate 任务（或 GitHub Actions 自托管 runner）
       - 拉取仓库
       - 推断 Dockerfile（语言、包管理器）
       - Daytona / E2B 沙箱带目标运行时
       - clone -> git worktree -> Agent 分支
            |
            v
    mini-swe-agent / SWE-agent v2 循环
       Claude Opus 4.7 或 GPT-5.4-Codex
       工具：ripgrep、tree-sitter、read/edit、run_tests、git
            |
            v
    在沙箱中验证 CI 通过 + 覆盖率变化检查
            |
            v（已验证）
    git push + 通过 GitHub App 打开 PR
       PR body = 理由 + diff 摘要 + trace URL
       标签：needs-review
            |
            v
    运维者 review；可以 @-mention Agent 跟进
```

## 技术栈

- 触发器：带细粒度 token 的 GitHub App；通过 Lambda 或 Fly.io 的 webhook 接收器
- Worker：ECS Fargate 任务（或 GitHub Actions 自托管 runner）
- 沙箱：每个任务一个 Daytona devcontainer 或 E2B 沙箱
- Agent 循环：基于 Claude Opus 4.7 / GPT-5.4-Codex 的 mini-swe-agent 基线或 SWE-agent v2
- 检索：tree-sitter repo-map + ripgrep
- 验证：沙箱内全量 CI + 覆盖率变化门控
- 可观测性：Langfuse，带从 PR body 链接的每 PR trace 归档
- 预算：每仓库每日美元上限；每仓库每天最多 PR 数

## 动手实现

1. **GitHub App。** 细粒度安装 token：issues read+write、pull_requests write、contents read+write、workflows read。分支保护（唯一能做到这点的表面）强制"不直接 push 到 `main`"和"no force-push"；App 不在 bypass 列表。Worker 在 proposed diff 上强制"不在 `.github/workflows` 下写入"作为白名单检查，因为 GitHub App 权限不是路径范围的。

2. **Webhook 接收器。** Lambda 函数接受 issue 标签/PR 评论 webhook。按标签 `@agent fix this` 过滤。入队到 SQS。

3. **调度器。** 从 SQS 取任务。强制每仓库每天预算。启动带仓库 URL、issue body 和全新 Daytona 沙箱的 ECS Fargate 任务。

4. **环境推断。** 检测语言（Python、Node、Go、Rust）和包管理器（uv、pnpm、go mod、cargo）。如果没有则动态生成 Dockerfile。

5. **Agent 循环。** Claude Opus 4.7 上的 mini-swe-agent 或 SWE-agent v2。工具：ripgrep、tree-sitter repo-map、read_file、edit_file、run_tests、git。硬限制：$20 成本，30分钟墙上时钟，30个 Agent 轮次。

6. **验证。** 循环结束后，在沙箱中运行完整测试套件。通过 jacoco / coverage.py 计算覆盖率变化。如果 CI 红：停止，不打开 PR。如果覆盖率下降超过 2%：打开 PR 标记 `needs-review`。

7. **发布 PR。** 推送 Agent 分支。通过 GitHub API 打开 PR：标题、理由、diff 摘要、trace URL、成本、轮次。

8. **凭据卫生。** Worker 用短期 GitHub App 安装 token 运行。日志在归档前清除 secrets。

9. **评估。** 30个不同难度的内部 seeded issue。测量通过率、PR 质量（diff 大小、风格、覆盖率）、成本、延迟。在相同 issue 上与 Cursor Background Agents 和 AWS Remote SWE Agents 对比。

## 用现成库

```bash
# on github.com
  - 用户给 issue #842 标记 `@agent fix this`
  - 14分钟后出现 PR #1903
  - body:
    > 修复了 widget.dedupe() 中的 NPE，由空比较器条目引起。
    > 添加了回归测试 widget_test.go::TestDedupeNullComparator。
    > 覆盖率变化：+0.12%
    > 轮次：7  成本：$1.80  Trace：langfuse:...
    > 标签：needs-review
```

## 产出

`outputs/skill-issue-to-pr.md` 是交付物。一个 GitHub App + 异步云 worker，将标记的 issue 变成带受限成本和范围凭据的 review-ready PR。

| 权重 | 指标 | 衡量方式 |
|:-:|---|---|
| 25 | 30个 issue 上通过率 | 端到端成功（CI 绿 + 覆盖率 OK） |
| 20 | PR 质量 | Diff 大小、覆盖率变化、风格合规 |
| 20 | 每个已解决 issue 的成本和延迟 | 每 PR $ 和墙上时钟 |
| 20 | 安全性 | 范围 token、每仓库预算、no force-push、凭据卫生 |
| 15 | 运维 UX | 理由注释、重试便利、@-mention 跟进 |
| **100** | | |

## 练习

1. 添加"修复不稳定测试"模式：标签 `@agent stabilize-flake TestX` 在沙箱中运行测试50次，提案稳定它的最小更改。

2. 在三个共享 issue 上对比成本 vs Cursor Background Agents。报告哪些工具在哪胜出。

3. 实现预算仪表盘：每仓库每天成本、每用户成本。异常时告警。

4. 构建"试运行"模式：不运行 CI 就打开 draft PR，让 reviewer 廉价检查计划。

5. 添加保留策略：超过7天未合并的 PR 分支自动删除。

## 关键术语

| 术语 | 行话 | 实际含义 |
|------|-----------------|------------------------|
| GitHub App | "范围 bot 身份" | 带细粒度权限 + 短期安装 token 的 App |
| Async cloud agent | "后台 Agent" | 在云沙箱中运行而非终端的非交互式 worker |
| Environment inference | "Dockerfile 合成" | 检测语言 + 包管理器，缺失时生成 Dockerfile |
| Verification | "沙箱内 CI" | 在打开 PR 前在 worker 内运行完整测试套件 |
| Coverage delta | "覆盖率保持" | 从基线到 Agent 分支的测试覆盖率变化 % |
| Per-repo budget | "每日上限" | 在调度器强制的美元和 PR 计数上限 |
| Rationale | "PR body 说明" | Agent 变更内容和原因摘要；PR body 必需 |

## 扩展阅读

- [AWS Remote SWE Agents](https://github.com/aws-samples/remote-swe-agents) — 权威异步云 Agent 参考
- [SWE-agent](https://github.com/SWE-agent/SWE-agent) — CLI 参考
- [Cursor Background Agents](https://docs.cursor.com/background-agent) — 商业替代
- [OpenAI Codex（cloud）](https://openai.com/codex) — 托管竞品
- [Google Jules](https://jules.google) — Google 托管版本
- [Factory Droids](https://www.factory.ai) — 备选商业参考
- [GitHub App 文档](https://docs.github.com/en/apps) — 范围 bot 身份
- [Daytona 云沙箱](https://daytona.io) — 参考沙箱