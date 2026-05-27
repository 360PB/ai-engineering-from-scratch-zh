# AI 服务可靠性 — 多 Agent 事件响应、运行手册、预测检测

> AI SRE 用 RAG 化的 LLMs（接入日志、运行手册、服务拓扑）来自动化调查、文档和协调阶段。2026 年架构模式是多 Agent 编排——专业 Agent（日志、指标、运行手册）由监督 Agent 协调；AI 提出假设和查询，人类批准判断。Datadog Bits AI 和 Azure SRE Agent 以托管产品形式提供。运行手册正在进化：NeuBird Hawkeye 用对抗评估（两个模型分析同一事件；一致 = 置信，分歧 = 不确定）；运维记忆跨团队变更持续存在。自动修复保持谨慎：AI 建议，人类批准。完全自主操作范围窄（重启 pod、回滚特定部署）且有严格护栏——任何卖"设完就忘"的都在夸大。新兴前沿：事件前预测。MIT 研究报告 LLM 在历史日志 + GPU 温度 + API 错误模式上训练，预测了测试集中 89% 的中断，提前 10-15 分钟。投影：2026 年底 95% 的企业 LLM 有自动化故障转移。

**类型：** 精读
**语言：** Python（标准库，玩具级多 Agent 事件分诊模拟器）
**前置要求：** Phase 17 · 13（可观测性）、Phase 17 · 24（混沌工程）
**时长：** 约 60 分钟

## 学习目标

- 画出多 Agent AI SRE 架构：监督 Agent + 专业 Agent（日志、指标、运行手册）+ 人类批准门。
- 解释为什么自动修复是窄范围的（重启 pod、回滚部署）而非宽范围的（重架构服务）。
- 说出对抗评估模式（NeuBird Hawkeye）：两模型一致 = 置信；分歧 = 升级。
- 引用 MIT 89% 早期检测结果和操作约束：预测无执行就只是仪表板。

## 背景问题

值班工程师凌晨 3 点收到告警。"结账错误率上升。"他们查 Datadog、Loki、三个运行手册、部署日志。30 分钟后发现根因是 KV 缓存尖峰导致的 vLLM OOM。重启 pod；错误清除。

2026 年这个调查的前 20 分钟可以自动化。将日志按服务分组、关联到最近部署、匹配运行手册——都是 RAG + 工具调用。一个监督 Agent 可以在人类打开 Datadog 之前做第一轮分诊并呈现假设。

完全自主修复是另一个问题。重启 pod：安全。扩缩 GPU 池：安全（如果策略允许）。重架构服务：绝对不行。纪律是画窄线。

## 核心概念

### 多 Agent 架构

```
          事件
             │
             ▼
        监督 Agent
        /    |    \
       ▼     ▼     ▼
  日志 Agent  指标 Agent  运行手册 Agent
       │     │     │
       └─────┴─────┘
             │
             ▼
        假设 + 证据
             │
             ▼
        人类批准
             │
             ▼
        操作（窄集）
```

监督 Agent 将事件分解为子查询。专业 Agent 有工具访问（日志搜索、PromQL、文档检索）。监督 Agent 综合，向人类呈现假设 + 证据。人类批准或重定向。

### 自动修复范围

**安全（窄范围）**：重启 pod、回滚特定部署、在预批准范围内扩缩池、启用预批准特性标志。

**不安全（宽范围）**：改变服务拓扑、修改资源限制、部署新代码、改 IAM、改数据库。

任何卖"设完就忘"的都在夸大。安全集随 AI SRE 成熟而扩大，但边界是真实的。

### 对抗评估（NeuBird Hawkeye）

两个模型独立分析同一事件。如果对根因一致，置信度高。如果分歧，向人类升级并展示两个假设。简单模式，对幻觉根因的有效过滤器。

### 运维记忆

团队轮换是传统 SRE 的隐形杀手——部落知识流失。AI SRE 把运行手册 + 事后分析存在向量 DB；Agent 在每次新事件时检索。新工程师加入时，AI 有完整历史。

### 事件前预测

MIT 2025 年研究：LLM 在历史日志、GPU 温度、API 错误模式上训练，在测试集上提前 10-15 分钟预测了 89% 的中断。

现实检验：预测无执行就是仪表板。操作问题是"当我们预测时，我们做什么？"预排空？Pager？自动扩缩？答案是策略特定的。

### 2026 年产品

- **Datadog Bits AI** — Datadog 内托管 SRE copilot。
- **Azure SRE Agent** — Azure 原生。
- **NeuBird Hawkeye** — 对抗评估 + 运维记忆模式。
- **PagerDuty AIOps** — 分诊 + 去重。
- **Incident.io Autopilot** — 事件指挥官 + 协调。

### 运行手册即代码

运行手册从 Confluence 页面演进到版本化 Markdown 含结构化章节（症状、假设、验证、操作）。结构化运行手册给更好 RAG 检索。任何 AI-SRE 上线前先把非结构化运行手册变结构化。

### 必须记住的数字

- MIT 早期检测：89% 中断，提前 10-15 分钟。
- 多 Agent 分诊：监督 Agent +（日志、指标、运行手册）+ 人类。
- 安全自动修复集：重启 pod、回滚部署、在范围内扩缩。
- 对抗评估：两模型独立；一致 = 置信。

## 用现成库

`code/main.py` 模拟多 Agent 分诊：日志 Agent 发现错误，指标 Agent 发现 CPU 尖峰，运行手册 Agent 匹配已知问题。监督 Agent 排列假设优先级。

## 产出

本课产出 `outputs/skill-ai-sre-plan.md`。给定当前 on-call、事件量、团队成熟度，设计 AI SRE 上线计划。

## 练习

1. 运行 `code/main.py`。如果日志和指标 Agent 分歧，监督 Agent 如何解决？
2. 为你的服务定义三个"安全"自动修复操作。为每个说明理由。
3. 写结构化运行手册模板：章节、必填字段、验证命令。
4. 预测检测在提前 12 分钟触发。你的策略是 Pager、预排空还是两者？
5. 论证 3 人团队在 2026 年是否应该采用 AI SRE。考虑成熟度、容量、风险。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| AI SRE | "on-call Agent" | LLM 支撑的事件调查 + 协调 |
| 监督 Agent | "协调器" | 顶层 Agent 将事件分解为子查询 |
| 专业 Agent | "领域 Agent" | 有工具访问的子 Agent（日志、指标、运行手册） |
| 自动修复 | "AI 修它" | 窄范围预批准操作；非宽范围重架构 |
| 运维记忆 | "向量运行手册" | 事后分析 + 运行手册存在向量 DB 用于 RAG |
| 对抗评估 | "双模型检查" | 独立分析；一致 = 置信 |
| NeuBird Hawkeye | "对抗评估那个" | 含对抗评估 + 记忆模式的产品 |
| Bits AI | "Datadog SRE Agent" | Datadog 托管 AI SRE |
| 事件前预测 | "早期检测" | 中断预测提前 10-15 分钟 |

## 扩展阅读

- [incident.io — AI SRE Complete Guide 2026](https://incident.io/blog/what-is-ai-sre-complete-guide-2026)
- [InfoQ — Human-Centred AI for SRE](https://www.infoq.com/news/2026/01/opsworker-ai-sre/)
- [DZone — AI in SRE 2026](https://dzone.com/articles/ai-in-sre-whats-actually-coming-in-2026)
- [Datadog Bits AI](https://www.datadoghq.com/product/bits-ai/)
- [NeuBird Hawkeye](https://www.neubird.ai/)
- [awesome-ai-sre](https://github.com/agamm/awesome-ai-sre)