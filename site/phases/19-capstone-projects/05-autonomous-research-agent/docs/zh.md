# Capstone 05 — 自主研究 Agent（AI Scientist 级）

> Sakana 的 AI-Scientist-v2 在 Nature 发表了论文。Agent Laboratory 运行了实验。Allen AI 公开了轨迹。2026年的形态是：计划-执行-验证树搜索，控制成本，沙箱隔离代码执行，带视觉反馈的 LaTeX 写作器，以及自动化 NeurIPS 式评审集成。毕业项目是构建一个，用30美元/篇的预算跑通全程，并经受住 Sakana 记录过的沙箱逃逸红队测试。

**类型：** 毕业项目
**语言：** Python（Agent + 沙箱），LaTeX（输出）
**前置知识：** Phase 2（机器学习）、Phase 3（深度学习）、Phase 7（Transformers）、Phase 10（从零构建LLM）、Phase 14（Agent工程）、Phase 15（自主系统）、Phase 16（多Agent）、Phase 18（安全）
**涉及阶段：** P0 · P2 · P3 · P7 · P10 · P14 · P15 · P16 · P18
**时长：** 40小时

## 问题

2026年，自主研究 Agent 跨越了一个门槛。Sakana AI 的 AI-Scientist-v2 在 Nature 发表了通过工作坊同行评审的生成论文。ShinkaEvolve（ICLR 2026）将方法线扩展到进化假设。AMD 的 Agent Laboratory 提供了可复现轨迹。这些 Agent 不是魔法——它们是在候选实验树上运行计划-执行-验证循环，有成本上限、种子隔离沙箱和自动化评审。功力在于循环、预算和安全体系。

你通过在一个狭窄领域（如，100M参数 Transformer 上的注意力稀疏性消融实验）针对种子想法实现一个 Agent 来学习这个循环。价值不在于第一次运行就发现新东西。价值在于基础设施：树搜索、实验沙箱、写作-评审循环、红队报告。Sakana 团队记录了沙箱逃逸失败案例；你的 Agent 必须通过同样的红队测试。

## 核心概念

Agent 是最佳优先树搜索。节点是实验规范：(假设, 配置, 代码, 预期结果)。扩展步骤以小编辑（更换优化器、调整 batch size、消融某个组件）提出子节点。每个子节点在全新沙箱中运行，有硬资源上限。结果输入打分函数，按（新颖性 × 质量 × 剩余预算）对节点排序。树在预算耗尽时停止生长，然后写出最佳分支。

写作器是多模态的。它生成 LaTeX 草稿，编译，渲染图表，将渲染后的 PDF 反馈给 Claude Opus 4.7 的视觉模式进行布局、图表可读性和论点-证据对齐的评审。五个 LLM 评审组成的评审团给出 NeurIPS 风格分数（新颖性、严谨性、清晰度、可复现性、影响力）；如果平均分低于阈值，论文退回给写作器附带评审意见。

安全是承重结构。每个实验都在 E2B 或 Daytona 沙箱中运行，无网络出口，有界 wall-clock，固定资源限制。Agent 的代码生成步骤通过策略层，阻止逃逸沙箱的系统调用。红队报告复现了 Sakana 记录的攻击面（fork bomb、文件系统逃逸、LLM 写的网络调用）。

## 架构

```
种子想法 + 领域
      |
      v
  文献搜索（Semantic Scholar + OpenAlex + FAISS 缓存）
      |
      v
  LangGraph 计划-执行-验证树
      |
      v
  +--- 扩展节点 ----+      每节点沙箱
  |                  |      （E2B / Daytona）
  v                  v      资源上限
  child_1          child_k  无网络出口
  |                  |      确定性种子
  v                  v
  运行实验            运行实验
  |                  |
  v                  v
  按（新奇度、质量、预算）打分
      |
      v
  最佳分支 -> LaTeX 写作器
      |
      v
  编译 + 视觉评审（Opus 4.7 vision）
      |
      v
  评审团（5个LLM裁判，NeurIPS评分标准）
      |
      v
  paper.pdf + review.md + trace.json
```

## 技术栈

- 编排：LangGraph，含检查点和人工审批门控
- 树搜索：实验节点上的自定义最佳优先搜索（Sakana v2 的 AB-MCTS 风格）
- 沙箱：每实验一个 E2B 实例，Docker-in-Docker 备选；通过 cgroups 限制资源
- 文献：Semantic Scholar Graph API + OpenAlex + 本地 FAISS 摘要缓存
- 写作器：LaTeX 模板 + Claude Opus 4.7（vision 模式）用于图表评审和布局
- 评审团：5个裁判集成（Opus 4.7、GPT-5.4、Gemini 3 Pro、DeepSeek R1、Qwen3-Max），加权聚合
- 实验框架：PyTorch 2.5 用于物理实验，W&B 用于日志
- 可观测性：Langfuse 用于 Agent 轨迹，$30 硬预算/篇

## 动手实现

1. **种子和领域范围界定。** 取一个种子想法（如"研究 sub-1B Transformer 注意力图中的稀疏性模式"）。定义搜索空间：模型、数据集、算力预算。

2. **文献检索。** 查询 Semantic Scholar + OpenAlex 获取50篇最相关的被引论文；本地缓存摘要；生成1页领域摘要。

3. **树脚手架。** 用种子假设初始化根节点。实现 `expand(node) -> children`，每个子节点一个配置变更的小编辑提案。实现 `score(node)` 为加权的新颖性 × 质量 × 预算项。

4. **沙箱封装。** 每个实验运行 `docker run --network=none --memory=8g --cpus=2 --pids-limit=256 --read-only`（或等效的 E2B 策略）。种子写入沙箱；输出以只读方式挂载出来。

5. **计划-执行-验证循环。** `plan` 提案子节点。`execute` 运行沙箱，捕获日志和指标。`verify` 对指标运行单元检查（loss 是否下降？消融是否隔离了效果？）。失败节点在树上存储失败原因。

6. **写作器。** 预算耗尽后，选择最佳分支。用 matplotlib 渲染图表。通过 Claude Opus 4.7 生成 LaTeX 草稿，上下文包含分支轨迹。编译。将编译后的 PDF 反馈给 Opus 4.7 vision 进行评审。迭代。

7. **评审团。** 五个裁判在新颖性、严谨性、清晰度、可复现性、影响力上对草稿评分（NeurIPS 风格标准）。如果均值 < 4.0/5，退回给写作器附评审意见。3次重写后硬停止。

8. **红队。** 构建或集成针对沙箱的对抗任务集：fork bomb、网络泄露尝试、文件系统逃逸、LLM 写的 shell 元字符。确认全部被阻止。写出发现报告。

9. **可复现性。** 每篇论文附带树搜索轨迹 JSON、种子、W&B 运行链接、沙箱配置和 README，从零复现论文。

## 用现成库

```bash
$ ai-scientist run --seed "sub-1B Transformer 中的注意力稀疏性" --budget 30
[文献]   50篇论文，摘要12秒生成
[树]     扩展8个节点，预算 12/30
[执行]   节点#3 sparsity=top-8, loss=2.83（目前最佳）
[执行]   节点#6 sparsity=top-4, loss=3.12（较差）
[执行]   ...
[树]     选择以节点#3 为根的分支（新奇度 0.62，质量 0.81）
[写作]   LaTeX 草稿 v1 完成
[视觉]   评审：图2图例太小，论点-证据 OK
[写作]   经过3次编辑后草稿 v2
[评审]   均分 4.2/5（新颖性 3.9，严谨性 4.3，清晰度 4.1，可复现 4.5，影响力 4.2）
[完成]   paper.pdf + review.md + trace.json     花费 $28.40
```

## 产出

`outputs/skill-ai-scientist.md` 是交付物。给定一个种子想法 + 领域 + $30 预算，它运行完整流水线，输出一份可评审的论文和可复现性包。

| 权重 | 指标 | 衡量方式 |
|:-:|---|---|
| 25 | 论文质量 | 盲审评分 against 已发表工作坊论文 |
| 20 | 实验严谨性 | 基线、种子、消融；每个论点背后有结果表中对应的单元格 |
| 20 | 成本与算力纪律 | $30/篇上限，Langfuse 追踪 |
| 20 | 安全性 | 沙箱红队通过；网络策略和终止开关验证 |
| 15 | 可复现性 | 一条命令用相同种子复现论文 |
| **100** | | |

## 练习

1. 用同一领域的三个不同种子想法运行流水线。比较树搜索重叠的部分。识别重复浪费的计算。

2. 在超过$5估算成本的节点执行前添加人工介入门控。测量总成本下降了多少。

3. 将评审团替换为单一裁判。测量在已知有问题论文的 held-out 集上的误接受率。

4. 引入网络泄露红队测试：Agent 编写尝试 `curl` 外部地址的代码。确认 `--network=none` 策略阻止了它。记录尝试。

5. 将树搜索与平摊随机基线（相同预算，无扩展策略）对比。报告新颖性 × 质量收益。

## 关键术语

| 术语 | 行话 | 实际含义 |
|------|-----------------|------------------------|
| Tree search | "AB-MCTS 式扩展" | 实验节点上的最佳优先探索，新颖性×质量×预算打分 |
| Sandbox | "实验隔离" | 无网络、CPU/内存有界、种子固定的容器，只读输入 |
| Vision critique | "渲染后读取" | 将论文编译为 PDF，反馈给 VLM 进行布局和论点-证据评审 |
| Reviewer ensemble | "自动化同行评审" | 多个 LLM 裁判用 NeurIPS 标准对论文评分；加权聚合控制流水线 |
| Novelty score | "这是新的吗？" | 惩罚与50篇文献缓存接近度的启发式 |
| Cost ceiling | "$ 预算" | 每篇论文的硬性花费上限；Langfuse 计数器 + 运行前估算 |
| Red team | "沙箱逃逸审计" | 如果策略错误就会逃逸沙箱的对抗任务 |

## 扩展阅读

- [Sakana AI-Scientist-v2 仓库](https://github.com/SakanaAI/AI-Scientist-v2) — 参考生产级研究 Agent
- [Sakana AI-Scientist-v1 论文（arXiv:2408.06292）](https://arxiv.org/abs/2408.06292) — 原始方法论
- [ShinkaEvolve（Sakana ICLR 2026）](https://sakana.ai) — 进化扩展
- [Agent Laboratory（AMD）](https://github.com/SamuelSchmidgall/AgentLaboratory) — 多角色实验室框架
- [LangGraph 文档](https://langchain-ai.github.io/langgraph/) — 参考编排层
- [Semantic Scholar Graph API](https://api.semanticscholar.org/) — 文献搜索
- [E2B 沙箱](https://e2b.dev) — 参考实验隔离
- [NeurIPS 评审指南](https://neurips.cc/Conferences/2026/Reviewer-Guidelines) — 评审团编码的评分标准