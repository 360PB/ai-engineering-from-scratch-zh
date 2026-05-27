# LLM 财务运营 — 单位经济学与多租户归因

> 传统 FinOps 在 LLM 支出上失效。成本是 token 交易，不是资源运行时间。标签不适用——一次 API 调用是交易，不是资产。工程决策（提示词设计、上下文窗口、输出长度）是财务决策。2026 年 playbook 有三个归因维度在第一天就要 instrument：per-user（`user_id`）用于席位定价和 expansion、per-task（`task_id` + `route`）用于产品表面成本和优先级、per-tenant（`tenant_id`）用于单位经济学和续约。四个 token 层——提示词、工具、记忆、响应——一个桶藏住所有支出。多租户产品的强制执行阶梯：每租户限流（预期峰值 2-3 倍，清晰 429 + retry-after）；每日支出上限（合同上限 1.5-3 倍；触发限流收紧 + alert）；支出 z-score > 4 时 kill switch（自动暂停 + page on-call）。归因模式：tag-and-aggregate、遥测连接器（trace-ID → billing；最高精度）、采样+推算、模型分配、事件溯源、实时流。单位指标：每解决查询成本、每生成产物成本——不是 $/M token。永远在请求创建时 instrument；追溯 tagging 总会漏。

**类型：** 精读
**语言：** Python（标准库，玩具级成本归因模拟器带 kill switch）
**前置要求：** Phase 17 · 13（可观测性）、Phase 17 · 14（缓存）
**时长：** 约 60 分钟

## 学习目标

- 解释为什么传统 FinOps（标签 + 层级）在 LLM 支出上失效，并说出三个新归因维度。
- 枚举四个 token 层（提示词、工具、记忆、响应）并说明单一 bucket 如何掩盖成本。
- 为多租户产品设计强制执行阶梯（限流 → 每日支出上限 → kill switch）。
- 选一个单位指标（每解决查询/产物成本）替代 $/M token。

## 背景问题

你的账单说 $40,000。你不知道：
- 哪个租户花的。
- 哪个产品功能驱动的。
- 是否有单个用户在滥用。
- 是提示词膨胀、工具调用还是记忆放大造成的。

Provider 端 tag-and-aggregate 对云资源有效（EC2、S3）因为标签传到账单行项目。LLM API 调用不自动打标签——你必须在调用点 stamp user/task/tenant 并贯穿。追溯归因总会漏边缘情况。

## 核心概念

### 三个归因维度

**Per-user**（`user_id`）：谁花了多少。驱动席位定价、expansion 对话、识别高用量用户。

**Per-task**（`task_id` + `route`）：哪个产品表面花了多少。驱动功能优先级、杀死昂贵功能决策。

**Per-tenant**（`tenant_id`）：哪个客户盈利。驱动单位经济学、续约定价、分层阈值。

在调用点第一天就 instrument 全部三个。追溯总是更差。

### 四个 token 层

| 层 | 示例 | 占总量典型比例 |
|----|------|--------------|
| 提示词 | system + user 输入 | 40-60% |
| 工具 | 工具调用结果反馈 | 20-40%（Agent 工作负载） |
| 记忆 | 之前对话/检索文档 | 10-30% |
| 响应 | 模型输出 | 10-30% |

四层混在一起让优化盲目。在归因 schema 中分层拆开。

### 强制执行阶梯

1. **限流** 按租户。预期峰值 2-3 倍。返回 429 加 `Retry-After`。租户感到阻力；无意外账单。

2. **每日支出上限** 按租户。合同上限 1.5-3 倍。触发：收紧限流 + alert 客户成功。

3. **Kill switch** 支出 z-score > 4 相对租户基线。自动暂停租户；page on-call；升级到 ops + CS。

### 归因模式

- **Tag-and-aggregate**：stamp 元数据 header；稍后聚合。简单；粗糙。
- **遥测连接器**：traces 通过 trace ID join 到账单。最高精度。成熟团队的做法。
- **采样 + 推算**：采样 5-10%，乘以。成本有效用于粗略支出；漏尾部。
- **基于模型分配**：回归推断成本驱动因素。用于无标签的遗留数据。
- **事件溯源**：成本作为流（Kafka / Kinesis）中的事件。实时。
- **实时流**：仪表板亚秒更新。

### 成本每 X 是单位指标

$/M token 是供应商话术。产品指标：

- 每解决工单成本。
- 每生成文章成本。
- 每成功 Agent 任务成本。
- 每用户会话分钟成本。

把成本绑定到产品结果。否则优化无锚。

### 成本归因 trace 形状

```
trace_id: abc123
  user_id: u_42
  tenant_id: t_7
  task_id: task_classify_doc
  route: model_haiku
  layers:
    prompt_tokens: 1800
    tool_tokens: 600
    memory_tokens: 400
    response_tokens: 150
  cost_usd: 0.0135
  cached_input: true
  batch: false
```

每条调用发出。存在数据湖。按维度聚合。Phase 17 · 13 可观测性栈是它所在的地方。

### 复合节省栈

叠加：缓存 + 批处理 + 路由 + 网关。全上：
- L2 缓存（Phase 17 · 14）：输入约 10 倍便宜。
- 批处理（Phase 17 · 15）：5 折。
- 路由到便宜模型（Phase 17 · 16）：成本降低 60%。
- 网关效率（Phase 17 · 19）：冗余 + 重试。

最佳情况叠加：约朴素基线的 5-10%。大多数团队用了 2-3 个杠杆；很少全用四个。

### 必须记住的数字

- 归因维度：per-user、per-task、per-tenant。
- 四个 token 层：提示词、工具、记忆、响应。
- Kill switch：支出 z-score > 4。
- 单位指标：每解决查询成本，不是 $/M token。
- 叠加优化：约基线的 5-10% 可实现。

## 用现成库

`code/main.py` 模拟带三层强制执行阶梯的多租户 LLM 服务。注入一个滥用租户并演示 kill switch 触发。

## 产出

本课产出 `outputs/skill-finops-plan.md`。给定产品和规模，设计归因 schema 和强制执行阶梯。

## 练习

1. 运行 `code/main.py`。Kill switch 在什么 z-score 触发？你怎么选阈值？
2. 设计每租户、每任务成本仪表板。你先建哪 5 个视图？
3. 你最大租户单位经济学为负。提出三个干预措施，按客户影响排序。
4. 计算支持产品每解决工单成本：每工单 3M token，约 800 工单/天，GPT-5 缓存费率。
5. 论证追溯 tagging 是否可行。什么时候可接受？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Per-user 归因 | "用户级成本" | `user_id` 在每条调用上 stamp |
| Per-task 归因 | "功能成本" | `task_id` + `route` 识别产品表面 |
| Per-tenant 归因 | "客户成本" | `tenant_id`；驱动单位经济学 |
| 四个 token 层 | "成本层级" | prompt + tool + memory + response |
| 限流 | "429 guard" | 网关处按租户上限强制执行 |
| 每日支出上限 | "每日上限" | 租户范围预算带 alert |
| Kill switch | "自动暂停" | 支出 z-score > 4 触发自动挂起 |
| 每解决成本 | "产品单位指标" | 成本绑定产品结果，不是 token |
| 遥测连接器 | "trace 到账单" | 最高精度归因模式 |
| 叠加优化 | "缓存+批处理+路由+网关" | 复合节省到约基线 5-10% |

## 扩展阅读

- [FinOps Foundation — FinOps for AI Overview](https://www.finops.org/wg/finops-for-ai-overview/)
- [FinOps School — Cost per Unit 2026 Guide](https://finopsschool.com/blog/cost-per-unit/)
- [Digital Applied — LLM Agent Cost Attribution 2026](https://www.digitalapplied.com/blog/llm-agent-cost-attribution-guide-production-2026)
- [PointFive — Managed LLMs in Azure OpenAI](https://www.pointfive.co/blog/finops-for-ai-economics-of-managed-llms-in-azure-open-ai)