# Capstone 09 — 代码迁移 Agent（仓库级语言/运行时升级）

> Amazon 的 MigrationBench（Java 8 到 17）和 Google 的 App Engine Py2 到 Py3 迁移器设定了2026年的标准。Moderne 的 OpenRewrite 做大规模确定性 AST 重写。Grit 用 codemod 风格 DSL 处理同一问题。生产形态结合两者：确定性底层做安全重写，加 Agent 层处理歧义情况，分支独立沙箱构建，在 PR 打开前让测试变绿。毕业项目是迁移50个真实仓库，发布通过率并附失败分类。

**类型：** 毕业项目
**语言：** Python（Agent），Java / Python（目标），TypeScript（仪表盘）
**前置知识：** Phase 5（NLP）、Phase 7（Transformers）、Phase 11（LLM工程）、Phase 13（工具）、Phase 14（Agent工程）、Phase 15（自主系统）、Phase 17（基础设施）
**涉及阶段：** P5 · P7 · P11 · P13 · P14 · P15 · P17
**时长：** 30小时

## 问题

大规模代码迁移是2026年编程 Agent 最干净的生产应用之一。基准真相明确（迁移后测试套件通过吗？），回报是真实的（Java-8 集群迁移是一项人力规模的工程），基准是公开的（MigrationBench 50仓库子集）。Moderne 的 OpenRewrite 处理确定性侧。Agent 层处理所有 OpenRewrite 配方无法处理的情况：歧义重写、构建系统漂移、长尾语法、可迁继依赖破坏。

你将构建一个 Agent，输入一个 Java 8 仓库（或 Python 2 仓库），输出一个 CI 变绿的迁移分支。你将测量通过率、测试覆盖率保持率、每仓库成本，并构建失败分类。与纯确定性基线的并排对比告诉你 Agent 的价值实际在哪里。

## 核心概念

流水线分两层。**确定性底层**（Java 用 OpenRewrite，Python 用 libcst）安全快速地运行大量机械重写：import、方法签名、空安全编辑、try-with-resources、废弃 API 替换。它快速且产生可审计的 diff。**Agent 层**（OpenAI Agents SDK 或 LangGraph over Claude Opus 4.7 和 GPT-5.4-Codex）处理配方无法处理的情况：构建文件升级（Maven/Gradle/pyproject）、可迁继依赖冲突、测试抖动、自定义注解。

每个仓库在 Daytona 沙箱中运行，预装了目标运行时。Agent 迭代：运行构建，分类失败，应用修复，重新运行。硬限制：每仓库30分钟，$8，20个 Agent 轮次。如果所有测试通过且覆盖率变化非负，打开 PR。否则，仓库归入失败类并附证据。

失败分类是交付物。跨50个仓库，什么坏了？可迁继依赖？自定义注解？构建工具版本？不相关的测试抖动？每类得到计数和一个样例 diff。未来的配方作者可以针对前三类。

## 架构

```
目标仓库
      |
      v
OpenRewrite / libcst 确定性配方
   （安全、快速、可审计，约 70-80% 的修复）
      |
      v
每个分支一个 Daytona 沙箱
      |
      v
agent 循环（Claude Opus 4.7 / GPT-5.4-Codex）:
   - 运行构建 -> 捕获失败
   - 分类失败（构建、测试、lint）
   - 应用修复（patch 或重试配方）
   - 重新运行
   - 预算：30分钟，$8，20轮
      |
      v
测试 + 覆盖率变化门控
      |
      v（通过）
打开 PR
      |
      v（失败）
归入失败类 + 附复现
```

## 技术栈

- 确定性底层：OpenRewrite（Java）或 libcst（Python）
- Agent：OpenAI Agents SDK 或 LangGraph over Claude Opus 4.7 + GPT-5.4-Codex
- 沙箱：每个分支一个 Daytona devcontainer，预装目标运行时（Java 17 / Python 3.12）
- 构建系统：Maven、Gradle、uv（Python）
- 基准：Amazon MigrationBench 50仓库子集（Java 8 到 17），Google App Engine Py2 到 Py3 仓库
- 测试工具：并行运行器，Jacoco（Java）或 coverage.py（Python）测覆盖率
- 可观测性：Langfuse + 每个仓库的 trace bundle，含每个 diff 块
- 仪表盘：失败分类仪表盘，每类计数和样例 diff

## 动手实现

1. **配方通过。** 先运行 OpenRewrite（Java）或 libcst（Python）配方。覆盖 70-80% 的机械迁移。提交为"配方"commit。

2. **构建试运行。** Daytona 沙箱：安装目标运行时，运行构建。如果绿，跳到测试。如果红，交由 Agent。

3. **Agent 循环。** LangGraph 含工具：`run_build`、`read_file`、`edit_file`、`run_test`、`git_diff`。Agent 分类失败（dep、syntax、test、build-tool）并应用针对性修复。重新运行。

4. **预算上限。** 每仓库30分钟墙上时钟，$8 成本，20个 Agent 轮次。任何超限则停止并归入"budget_exhausted"，附当前 diff。

5. **测试 + 覆盖率门控。** 构建变绿后，运行测试套件。与基线仓库比较覆盖率。如果覆盖率下降超过 2%，归入"coverage_regression"。

6. **打开 PR。** 成功后，推送分支，用 diff 和哪些配方适用、哪些 commit 是 Agent 编写的摘要打开 PR。

7. **失败分类。** 对每个失败仓库，标注类别：`dep_upgrade_required`、`build_tool_drift`、`custom_annotation`、`test_flake`、`syntax_edge_case`、`budget_exhausted`。构建仪表盘。

8. **50仓库运行。** 跨 MigrationBench 子集执行。报告每类通过率、每仓库成本、覆盖率保持率和 vs 纯确定性基线对比。

## 用现成库

```bash
$ migrate legacy-java-service --target java17
[配方]    应用了27个重写（JUnit 4->5，HashMap 初始化器，try-with-resources）
[构建]    失败：找不到符号 sun.misc.BASE64Encoder
[Agent]   轮次1 分类：removed_jdk_api
[Agent]   轮次2 应用：sun.misc.BASE64Encoder -> java.util.Base64
[构建]    OK
[测试]    412/412 通过；覆盖率 84.1% -> 84.3%
[PR]      打开了 #1841  成本=$3.20  轮次=4
```

## 产出

`outputs/skill-migration-agent.md` 是交付物。给定一个仓库，执行确定性配方然后 Agent 循环产出绿色的迁移分支，或者将仓库归入分类类。

| 权重 | 指标 | 衡量方式 |
|:-:|---|---|
| 25 | MigrationBench 通过率 | 50仓库子集 pass@1 |
| 20 | 测试覆盖率保持率 | vs 基线的平均覆盖率差值 |
| 20 | 每迁移仓库成本 | 通过运行上的 $/仓库 |
| 20 | Agent / 确定性工具集成 | OpenRewrite 处理 vs Agent 撰写的修复比例 |
| 15 | 失败分析报告 | 带样例的分类完整性 |
| **100** | | |

## 练习

1. 用纯 OpenRewrite（无 Agent）运行迁移流水线。与完整流水线对比通过率。识别 Agent 单独发挥作用的情况。

2. 实现"lint清理"检查：迁移后运行风格 linter（Java 用 spotless，Python 用 ruff）。新 lint 错误出现则 PR 失败。测量覆盖率保持但风格退化率。

3. 添加"最小diff"优化器：Agent 分支测试通过后，用第二轮修整不必要的更改。报告 diff 大小缩减。

4. 扩展到第三个迁移：Node 18 到 Node 22。复用沙箱封装；换自定义 codemod 的配方层。

5. 测量首次绿色构建时间（TTFGB）作为 UX 指标。目标：p50 在10分钟以内。

## 关键术语

| 术语 | 行话 | 实际含义 |
|------|-----------------|------------------------|
| Deterministic substrate | "配方引擎" | OpenRewrite / libcst：有安全保证的声明式 AST 重写 |
| Codemod | "代码修改程序" | 机械改变源代码的重写规则 |
| Build drift | "工具版本偏差" | 主版本间 Maven / Gradle / uv 行为的细微变化 |
| Failure class | "分类桶" | 仓库未迁移的标注原因：dep、syntax、test、build-tool、budget |
| Coverage delta | "覆盖率保持" | 从基线到迁移分支的测试覆盖率变化 % |
| Agent turn | "工具调用轮次" | Agent 循环中一个 plan -> act -> observe 周期 |
| Budget exhaustion | "触顶" | 仓库消耗了 30分钟/$8/20轮 限制仍未通过 |

## 扩展阅读

- [Amazon MigrationBench](https://aws.amazon.com/blogs/devops/amazon-introduces-two-benchmark-datasets-for-evaluating-ai-agents-ability-on-code-migration/) — 2026年权威基准
- [Moderne.io OpenRewrite 平台](https://www.moderne.io) — 确定性底层参考
- [OpenRewrite 文档](https://docs.openrewrite.org) — 配方编写
- [Grit.io](https://www.grit.io) — 备选 codemod DSL
- [OpenAI 沙箱迁移 cookbook](https://developers.openai.com/cookbook/examples/agents_sdk/sandboxed-code-migration/sandboxed_code_migration_agent) — Agents SDK 参考
- [Google App Engine Py2 到 Py3 迁移器](https://cloud.google.com/appengine) — 备选迁移基准
- [libcst](https://github.com/Instagram/LibCST) — Python 确定性底层
- [Daytona 沙箱](https://daytona.io) — 每分支沙箱参考