# Capstone 10 — 多 Agent 软件工程团队

> SWE-AF 的工厂架构、MetaGPT 的角色化提示、AutoGen 0.4 的类型化 Actor 图、Cognition 的 Devin 和 Factory 的 Droids 在2026年收敛到了同一形态：架构师规划，N 个程序员在并行 worktree 中工作，评审员门控，测试员验证。并行 worktree 将墙上时钟转化为吞吐量。共享状态和交接协议成为失败面。毕业项目是构建团队，在 SWE-bench Pro 上评估，报告哪些交接坏了以及频率。

**类型：** 毕业项目
**语言：** Python / TypeScript（Agent），Shell（worktree 脚本）
**前置知识：** Phase 11（LLM工程）、Phase 13（工具）、Phase 14（Agent工程）、Phase 15（自主系统）、Phase 16（多Agent）、Phase 17（基础设施）
**涉及阶段：** P11 · P13 · P14 · P15 · P16 · P17
**时长：** 40小时

## 问题

单 Agent 编码框架在大任务上触到天花板。不是因为单个 Agent 弱，而是因为 200k-token 的上下文无法容纳架构计划加四个并行代码库切片加评审员评论加测试输出。多 Agent 工厂分解了问题：架构师拥有计划，程序员在并行 worktree 中独立实现，评审员门控，测试员验证。SWE-AF 的"工厂"架构、MetaGPT 的角色、AutoGen 的类型化 Actor 图——三种表述描述的是同一形态。

失败面在交接。架构师规划了程序员无法实现的东西。程序员产出冲突的 diff。评审员批准了幻觉修复。测试员与仍在写的程序员赛跑。你将构建这样一个团队，在50个 SWE-bench Pro issue 上运行它，跟踪每次交接，并发布事后分析。

## 核心概念

角色是类型化 Agent。**架构师**（Claude Opus 4.7）阅读 issue，写计划，拆分为带显式接口的子任务。**程序员**（Claude Sonnet 4.7，N 个并行实例，每个在 `git worktree` + Daytona 沙箱中）独立实现子任务。**评审员**（GPT-5.4）阅读合并后的 diff，批准或请求具体修改。**测试员**（Gemini 2.5 Pro）在隔离环境中运行测试套件，报告通过/失败并附产物。

通信通过共享任务板（文件后端或 Redis）。每个角色消费它被允许处理的任务。交接是 A2A 协议类型消息。协调关注点：合并冲突解决（协调者角色或自动三方合并）、共享状态同步（程序员开始时计划冻结；重新规划是独立事件）、评审员门控（评审员不能批准自己写的或自己提议的修改）。

Token 放大是隐藏成本。每个角色边界添加摘要提示和交接上下文。一个40轮单 Agent 运行变成四个角色共160轮。评分标准特别权衡 Token 效率 vs 单 Agent 基线，因为问题不是"多 Agent 有效吗"而是"每美元它能赢吗"。

## 架构

```
GitHub issue URL
      |
      v
架构师（Opus 4.7）
   阅读 issue，产出带子任务 + 接口的计划
      |
      v
任务板（文件 / Redis）
      |
   +-- 子任务 1 ---+-- 子任务 2 ---+-- 子任务 3 ---+-- 子任务 4 ---+
   v                v                v                v                v
程序员 A          程序员 B          程序员 C          程序员 D          （4个并行）
 (Sonnet)         (Sonnet)         (Sonnet)         (Sonnet)
 worktree A       worktree B       worktree C       worktree D
 Daytona          Daytona          Daytona          Daytona
      |                |                |                |
      +--------+-------+-------+--------+
               v
           合并协调器 （三方合并 + 冲突解决）
               |
               v
           评审员（GPT-5.4）
               |
               v
           测试员（Gemini 2.5 Pro）  -> 通过？ -> 打开 PR
                                     -> 失败？  -> 打回程序员
```

## 技术栈

- 编排：LangGraph 带共享状态 + 每个 Agent 子图
- 消息：A2A 协议（Google 2025）用于类型化 Agent 间消息
- 模型：Opus 4.7（架构师）、Sonnet 4.7（程序员）、GPT-5.4（评审员）、Gemini 2.5 Pro（测试员）
- Worktree 隔离：每个程序员一个 `git worktree add` 分支 + Daytona 沙箱
- 合并协调器：自定义三方合并 + LLM 调解的冲突解决
- 评估：50个 issue 的 SWE-bench Pro、SWE-AF 场景、HumanEval++ 用于单元测试
- 可观测性：Langfuse 带角色标签 span，每 Agent Token 核算
- 部署：K8s，每个角色作为独立 Deployment + backlog 上 HPA

## 动手实现

1. **任务板。** 文件后端 JSONL，带类型消息：`plan_request`、`subtask`、`diff_ready`、`review_needed`、`test_needed`、`approved`、`rejected`、`replan_needed`。Agent 订阅标签。

2. **架构师。** 读取 GitHub issue，用要求显式子任务接口的计划模板运行 Opus 4.7（触碰的文件、公共函数、测试影响）。发出一个 `plan_request` 带子任务 DAG。

3. **程序员。** N 个并行 worker，每个从板上认领一个子任务。每个生成一个全新的 `git worktree add` 分支加 Daytona 沙箱。实现子任务。发出 `diff_ready` 带 patch + 测试增量。

4. **合并协调器。** 所有人完成后，将 N 个分支三方合并到 staging 分支。只有文件级重叠存在时才用 LLM 调解冲突解决。

5. **评审员。** GPT-5.4 读取合并的 diff。不能批准自己写的 diff。发出 `approved`（无操作）或 `review_feedback` 带具体修改请求，路由给相关程序员。

6. **测试员。** Gemini 2.5 Pro 在干净沙箱中运行测试套件。捕获产物。发出 `test_passed` 或 `test_failed` 带堆栈。失败的测试循环回拥有失败子任务的程序员。

7. **交接核算。** 每个跨角色边界的消息在 Langfuse 中获得一个 span，含 payload 大小和使用的模型。计算每个子任务的 Token 放大（coder_tokens + reviewer_tokens + tester_tokens + architect_share / coder_tokens）。

8. **评估。** 在50个 SWE-bench Pro issue 上运行。对比 pass@1 和 每解决 issue $成本与单 Agent 基线（一个 Sonnet 4.7 在单个 worktree 中）。

9. **事后分析。** 对每个失败的 issue，识别哪个交接坏了（计划太模糊、合并冲突、评审员误批准、测试员抖动）。产出交接失败直方图。

## 用现成库

```bash
$ team run --issue https://github.com/acme/widget/issues/842
[架构师] 计划：4个子任务（parser, cache, api, migration）
[看板]    分派到4个程序员并行 worktree
[程序员A] 子任务 parser  -> 42行，本地测试通过
[程序员B] 子任务 cache   -> 88行，本地测试通过
[程序员C] 子任务 api     -> 31行，本地测试通过
[程序员D] 子任务 migration -> 19行，本地测试通过
[合并]    三方合并：0冲突
[评审员]  评论 cache（线程池大小）；路由给程序员B
[程序员B] 修改：92行；提交
[评审员]  批准
[测试员]  全部412个测试通过
[PR]      打开了 #3382   4个程序员，1次修改，$4.90，18分钟
```

## 产出

`outputs/skill-multi-agent-team.md` 是交付物。给定 issue URL 和并行级别，团队产出带每角色 Token 核算的可合并 PR。

| 权重 | 指标 | 衡量方式 |
|:-:|---|---|
| 25 | SWE-bench Pro pass@1 | 匹配的50-issue 子集，pass@1 |
| 20 | 并行加速 | 墙上时钟 vs 单 Agent 基线 |
| 20 | 评审质量 | 注入 bug 探测上的误批准率 |
| 20 | Token 效率 | 每解决 issue 总 Token vs 单 Agent |
| 15 | 协调工程 | 合并冲突解决、交接失败直方图 |
| **100** | | |

## 练习

1. 在运行中向 diff 注入一个明显的 bug（主函数体前多加 `return None`）。测量评审员的误批准率。调整评审员提示直到误批准率低于 5%。

2. 缩减为两个程序员（架构师 + 程序员 + 评审员 + 测试员，程序员顺序跑两个子任务）。对比墙上时钟和通过率。

3. 将合并协调器替换为单写约束（子任务触碰不相交的文件集）。测量架构师的规划负担。

4. 将评审员从 GPT-5.4 换成 Claude Opus 4.7。测量误批准率和 Token 成本差值。

5. 添加第五个角色：文档员（Haiku 4.5）。评审后，它产出变更日志条目。测量文档质量是否值得额外 Token 支出。

## 关键术语

| 术语 | 行话 | 实际含义 |
|------|-----------------|------------------------|
| Parallel worktree | "隔离分支" | 每个程序员一个 `git worktree add` 产生的新工作树 |
| Task board | "共享消息总线" | Agent 订阅的类型化消息的文件或 Redis 存储 |
| Handoff | "角色边界" | 任何从一个角色上下文跨到另一个的消息 |
| Token amplification | "多 Agent 开销" | 跨角色总 Token / 同任务单 Agent Token |
| A2A protocol | "Agent 间" | Google 2025 年类型化 Agent 间消息规范 |
| Merge coordinator | "集成器" | 运行三方合并和调解冲突的组件 |
| False approval | "评审员幻觉" | 评审员批准了有已知 bug 的 diff |

## 扩展阅读

- [SWE-AF 工厂架构](https://github.com/Agent-Field/SWE-AF) — 2026年多 Agent 工厂参考
- [MetaGPT](https://github.com/FoundationAgents/MetaGPT) — 角色化多 Agent 框架
- [AutoGen v0.4](https://github.com/microsoft/autogen) — Microsoft 的类型化 Actor 框架
- [Cognition AI（Devin）](https://cognition.ai) — 参考产品
- [Factory Droids](https://www.factory.ai) — 备选参考产品
- [Google A2A 协议](https://developers.google.com/agent-to-agent) — Agent 间消息规范
- [git worktree 文档](https://git-scm.com/docs/git-worktree) — 隔离底层
- [SWE-bench Pro](https://www.swebench.com) — 评估目标