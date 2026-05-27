# LLM 生产混沌工程

> 混沌工程用于 LLMs 在 2026 年是自己的学科。上生产实验前的前提条件：定义的 SLI/SLO、trace+metric+log 可观测性、自动回滚、运行手册、on-call。架构有四个平面：控制（实验调度器）、目标（服务、基础设施、数据存储）、安全（护栏 + 中止 + 流量过滤器）、可观测性（指标 + traces + 日志）、反馈（进入 SLO 调整）。护栏是强制性的：燃烧率警报在日错误预算消耗 >2 倍预期时暂停实验；抑制窗口 + trace-ID 关联去重告警噪音。节奏：每周小规模金丝雀 + SLO 审查；每月游戏日 + 事后分析；每季度跨团队韧性审计 + 依赖映射。LLM 特有实验：内存过载、网络故障、供应商中断、畸形提示词、KV 缓存驱逐风暴。工具：Harness Chaos Engineering（LLM 派生建议、爆炸半径缩减、MCP 工具集成）；LitmusChaos（CNCF）；Chaos Mesh（CNCF Kubernetes 原生）。

**类型：** 精读
**语言：** Python（标准库，玩具级混沌实验运行器）
**前置要求：** Phase 17 · 23（AI 服务可靠性）、Phase 17 · 13（可观测性）
**时长：** 约 60 分钟

## 学习目标

- 说出五个混沌工程前提条件（SLI/SLO、可观测性、回滚、运行手册、on-call）并解释跳过任何一个如何破坏实践。
- 画出四个平面（控制、目标、安全、可观测性）和进入 SLO 的反馈循环。
- 枚举五个 LLM 特有实验（内存过载、网络故障、供应商中断、畸形提示词、KV 驱逐风暴）。
- 根据技术栈在 Harness、LitmusChaos、Chaos Mesh 中选择工具。

## 背景问题

传统技术栈的混沌测试已很成熟。LLM 栈增加了新的失败模式。带毒字符的 4K-token 提示词使 tokenizer 停滞 12 秒。上游供应商 429；你的网关重试；你的服务 OOM 在重试放大的并发下。KV 缓存在突发负载下驱逐风暴导致重 prefill 级联饱和计算资源。

这些在单元测试里都不出现。混沌工程是在用户遇到之前发现它们的方法。

## 核心概念

### 前提条件

不要在没有以下条件时在生产做混沌：

1. **SLI/SLO** — 已定义的服务级指标和目标。
2. **可观测性** — traces、指标、日志，接仪表板。
3. **自动回滚** — Phase 17 · 20 策略标志回滚。
4. **运行手册** — 结构化，Phase 17 · 23。
5. **on-call** — 有人响应。

缺少任何一条，混沌就会变成真实事件。

### 四个平面 + 反馈

**控制平面** — 实验调度器（Litmus workflow、Chaos Mesh schedule、Harness UI）。

**目标平面** — 服务、pod、节点、负载均衡器、数据存储。

**安全平面** — 杀死开关、抑制窗口、爆炸半径限制、错误预算门。

**可观测性平面** — 正常指标 + trace-ID 关联，区分混沌引起与自然失败。

**反馈循环** — 发现反馈进 SLO 调整、运行手册更新、代码修复。

### 护栏是强制性的

- **燃烧率警报**：日错误预算消耗超过预期 2 倍时暂停实验。
- **抑制窗口**：实验爆炸半径内在实验期间静默非实验告警。
- **Trace-ID 关联**：所有实验引起错误带标记，以便 on-call 去重。

### 五个 LLM 特有实验

1. **内存过载** — 通过高并发发送长上下文请求强制 KV 缓存抢占风暴。观察：服务优雅降级还是崩溃？

2. **网络故障** — 切断推理网关和供应商间的连通性。观察：回退是否在 SLA 内启动？（Phase 17 · 19）

3. **供应商中断模拟** — OpenAI 100% 429。观察：路由是否故障转移到 Anthropic？（Phase 17 · 16、19）

4. **畸形提示词** — 注入 tokenizer 停滞载荷（如深度嵌套 unicode、巨大 UTF-8 码点）。观察：单个请求是否锁住 worker？

5. **KV 驱逐风暴** — 通过饱和 vLLM 块预算强制驱逐。观察：LMCache 恢复还是服务降级？

### 节奏

- **每周** — 预发小规模金丝雀实验，也许 5% 生产。
- **每月** — 特定场景的预定游戏日；跨团队参加；事后分析。
- **每季度** — 跨团队韧性审计；依赖图更新。

### 工具链

- **Harness Chaos Engineering** — 商业；AI 派生实验建议；爆炸半径缩减；MCP 工具集成。
- **LitmusChaos** — CNCF 毕业；Kubernetes 基于 workflow。
- **Chaos Mesh** — CNCF 沙箱；Kubernetes 原生 CRD 风格。
- **Gremlin** — 商业；广泛支持。
- **AWS FIS / Azure Chaos Studio** — 托管云产品。

### 小步开始

第一个实验：在稳态流量下杀死一个 decode 副本。观察重路由和恢复。如果这看起来安全，升级到网络混沌。

第一个 LLM 特有实验：注入一个供应商 429，持续 5 分钟。观察回退。大多数团队发现回退没有完全测过。

### 必须记住的数字

- 四个平面：控制、目标、安全、可观测性。
- 燃烧率暂停：日预算消耗超过预期 2 倍。
- 节奏：每周金丝雀、每月游戏日、每季度审计。
- 五个 LLM 实验：内存、网络、供应商、畸形提示词、KV 风暴。

## 用现成库

`code/main.py` 模拟三个带安全平面门的混沌实验。报告哪些实验会触发燃烧率中止。

## 产出

本课产出 `outputs/skill-chaos-plan.md`。给定技术栈和成熟度，选出前三个实验和工具。

## 练习

1. 运行 `code/main.py`。哪个实验触发燃烧率门？为什么？
2. 为基于 vLLM 的 RAG 服务设计前五个混沌实验。含成功标准。
3. 你的燃烧率警报暂停了一个实验。你如何确定根因——混沌还是自然？
4. 论证混沌应该生产运行还是仅预发。什么时候生产是正确答案？
5. 说出通用网络混沌无法复现的三个 LLM 特有失败模式。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| SLI / SLO | "服务目标" | 指标 + 目标；必须前提 |
| 爆炸半径 | "范围" | 受实验影响的服务/用户集合 |
| 燃烧率警报 | "预算门" | 错误预算消耗率 >2 倍预期时触发 |
| 游戏日 | "每月演练" | 预定跨团队混沌演习 |
| LitmusChaos | "CNCF workflow" | 毕业 CNCF Kubernetes 混沌工具 |
| Chaos Mesh | "CNCF CRD" | CNCF 沙箱 Kubernetes 原生混沌 |
| Harness CE | "商业 AI 辅助" | Harness 混沌含 AI 建议 |
| 畸形提示词 | "tokenizer 炸弹" | 使 token 化停滞的输入 |
| KV 驱逐风暴 | "抢占级联" | 批量驱逐触发重 prefill |

## 扩展阅读

- [DevSecOps School — Chaos Engineering 2026 Guide](https://devsecopsschool.com/blog/chaos-engineering/)
- [Ankush Sharma — Observability for LLMs (book)](https://www.amazon.com/Observability-Large-Language-Models-Engineering-ebook/dp/B0DJSR65TR)
- [LitmusChaos (CNCF)](https://litmuschaos.io/)
- [Chaos Mesh (CNCF)](https://chaos-mesh.org/)
- [Harness Chaos Engineering](https://www.harness.io/products/chaos-engineering)
- [AWS FIS](https://aws.amazon.com/fis/)