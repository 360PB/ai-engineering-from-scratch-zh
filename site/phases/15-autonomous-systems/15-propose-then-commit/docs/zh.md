# Human-in-the-Loop：先提议后执行

> 2026 年关于 HITL 的共识是具体的。它不是"Agent 提问、用户点击批准"，而是先提议后执行：提议的动作持久化到持久化存储并带上幂等键；以意图、数据血缘、涉及的权限、爆炸半径和回滚计划呈现给审查者；仅在得到确认后提交；执行后验证以确认副作用确实发生。LangGraph 的 `interrupt()` 配合 PostgreSQL 检查点、Microsoft Agent Framework 的 `RequestInfoEvent`、Cloudflare 的 `waitForApproval()` 都实现了相同的形态。最典型的失败模式是橡皮图章式批准：不经审查就点击"批准"。已记录在案的缓解措施是挑战-应答式清单。

**类型：** 课程
**语言：** Python（标准库，先提议后执行状态机 + 幂等性）
**前置要求：** 第 15 阶段 · 12（持久化执行）、第 15 阶段 · 14（断路器）
**时长：** 约 60 分钟

## 问题

Agent 执行了一个动作。用户需要决定：批准还是不批准。如果决策是即时的，它可能不是真正的审查。如果决策是结构化的，它虽然慢但可信。工程问题是：如何让结构化审查成为最容易的选择。

2023 年代的 HITL 模式是同步提示："Agent 想要向 X 发送邮件，内容为 Y——批准？"用户点击批准。大家觉得系统很安全。实际上这种界面被大量橡皮图章式批准：用户批准很快，批准行为几乎无法预测任何信息，而当 Agent 出错时，审计日志显示用户无法回忆的一长串批准记录。

2026 年的模式——先提议后执行——将 HITL 移到了持久化介质上，附加了结构化元数据，并要求正向提交。所有托管 Agent SDK 都发布了一个版本：LangGraph `interrupt()`、Microsoft Agent Framework `RequestInfoEvent`、Cloudflare `waitForApproval()`。API 名称不同，但形态一致。

## 概念

### 先提议后执行状态机

1. **提议。** Agent 生成一个提议动作。持久化到持久化存储（PostgreSQL、Redis、Durable Object）。包含：
   - intent（Agent 为什么做这件事）
   - data lineage（什么来源导致了这个提议）
   - permissions touched（涉及哪些范围 / 文件 / 端点）
   - blast radius（最坏情况是什么）
   - rollback plan（如果提交了，如何撤销）
   - idempotency key（每个提议唯一；重复提交返回同一条记录）
2. **呈现。** 审查者看到带有全部元数据的提议。审查者是人（不是 Agent 自己审查自己）。
3. **提交。** 正向确认。动作执行。
4. **验证。** 执行后，重新读取目标资源确认副作用。如果验证步骤失败，系统处于已知的不良状态，触发告警。

### 幂等键

没有幂等键，重试瞬时故障可能导致已批准的动作被执行两次。具体例子：用户批准"从 A 向 B 转账 $100"。网络抖动。工作流重试。用户只批准了一次，但转账执行了两次。幂等键将批准绑定到单一唯一副作用；第二次执行是空操作。

这与 Stripe 和 AWS API 使用的幂等性模式相同。在 Agent 批准中复用这一模式在 Microsoft Agent Framework 文档中有明确说明。

### 持久化：为什么批准不随进程消失

批准等待区是 Agent 不拥有的状态。工作流处于暂停状态（第 12 课）。批准到来时，工作流从准确的那个点恢复。这就是为什么 LangGraph 将 `interrupt()` 与 PostgreSQL 检查点配合使用，而不是仅仅使用内存状态——两天后的批准仍然能找到完整的工作流。

### 橡皮图章式批准及挑战-应答缓解措施

HITL 的默认 UI（"批准"/"拒绝"按钮）产生快速批准，没有真正的审查。已记录在案的缓解措施：挑战-应答清单，要求在启用批准按钮之前对具体问题给出正向回答。具体形态：

- "你理解这个操作涉及什么资源吗？[ ]"
- "你验证过爆炸半径是可接受的吗？[ ]"
- "如果失败，你有回滚计划吗？[ ]"

这不是为了繁琐而繁琐——而是一个强制函数。无法勾选这些框的审查者要么要求澄清（升级），要么拒绝（安全默认值）。Anthropic Agent 安全研究明确指出，清单驱动的 HITL 是橡皮图章式批准模式的缓解措施。

### 什么算"有影响的"

并非每个动作都需要先提议后执行。2026 年指导：

- **有影响的操作**（始终 HITL）：不可逆写入、金融交易、对外通信、生产数据库变更、破坏性文件系统操作。
- **可逆操作**（有时 HITL）：编辑本地文件、预发环境变更、有明确回滚的可逆写入。
- **读取和检查**（不需要 HITL）：读取文件、列出资源、调用只读 API。

### 行动后验证

"提交已运行"不等于"副作用发生了"。网络分区和竞态条件可能导致工作流认为成功但后端未持久化。验证步骤在提交后重新读取目标资源确认。这与带 `RETURNING` 子句的数据库事务或 AWS `PutObject` 后的 `GetObject` 模式相同。

### 欧盟 AI 法案第 14 条

第 14 条要求对高风险 AI 系统进行有效的人工监督。"有效"不是装饰性的。法规语言明确排除橡皮图章模式。先提议后执行配合挑战-应答是在 Microsoft Agent 治理工具包合规文档中通过第 14 条审查的形态。

## 用现成库

`code/main.py` 用标准库 Python 实现了先提议后执行状态机。持久化存储是一个 JSON 文件。幂等键是 (thread_id, action_signature) 的哈希。驱动模拟三种情况：干净批准流程、重试瞬时故障（不能重复执行）和橡皮图章默认 vs 挑战-应答流程。

## 产出

`outputs/skill-hitl-design.md` 审查提议的 HITL 工作流是否符合先提议后执行形态，并标记缺失的元数据、幂等性、验证或挑战-应答层。

## 练习

1. 运行 `code/main.py`。确认已批准提议的重试使用持久化记录，不会重新执行。现在将幂等键改为包含时间戳，证明重试会重复执行。

2. 为提议记录扩展一个 `rollback` 字段。模拟一个验证步骤失败的执行，展示回滚自动触发。

3. 阅读 Microsoft Agent Framework 的 `RequestInfoEvent` 文档。识别 API 包含的一个元数据字段，而玩具引擎缺少它。添加它并解释它防止什么。

4. 为特定操作（如"发布到公开 Twitter 账号"）设计挑战-应答清单。审查者必须回答哪三个问题？为什么是这三个？

5. 选择一个同步"批准？"提示就足够的场景（不需要持久化存储）。解释原因，并命名你接受的风险类别。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|---|---|---|
| Propose-then-commit | 两阶段批准 | 持久化提议 + 正向提交 + 验证 |
| Idempotency key | 重试安全令牌 | 每个提议唯一；第二次执行是空操作 |
| Data lineage | 数据来源 | 导致提议的特定源内容 |
| Blast radius | 最坏情况 | 操作出错时的波及范围 |
| Rubber-stamp | 快速批准 | 未经真正审查就点击"批准" |
| Challenge-and-response | 强制清单 | 审查者必须正向确认具体问题 |
| RequestInfoEvent | MS Agent Framework 原语 | 带结构化元数据的持久化 HITL 请求 |
| `interrupt()` / `waitForApproval()` | 框架原语 | LangGraph / Cloudflare 的同形态等价物 |

## 延伸阅读

- [Microsoft Agent Framework — Human in the loop](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — `RequestInfoEvent`，持久化批准。
- [Cloudflare Agents — Human in the loop](https://developers.cloudflare.com/agents/concepts/human-in-the-loop/) — `waitForApproval()` 和 Durable Objects。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — HITL 作为长期风险缓解措施。
- [EU AI Act — Article 14: Human oversight](https://artificialintelligenceact.eu/article/14/) — 高风险系统的监管基线。
- [Anthropic — Claude's Constitution (January 2026)](https://www.anthropic.com/news/claudes-constitution) — 围绕监督的宪法框架。