# Capstone 06 — Kubernetes DevOps 故障排查 Agent

> AWS 的 DevOps Agent 已正式发布，Resolve AI 公开了 K8s 运维手册，NeuBird 演示了语义监控，Metoro 将 AI SRE 与逐服务 SLO 关联。生产形态已确定：告警 webhook 触发，Agent 读取遥测数据，走查 K8s 对象图，对根因假设排序，在 Slack 发布简要报告并附审批按钮。默认只读。所有修复操作需人工审批。这就是要构建的 Agent，在20个合成故障场景上评估，并与 AWS Agent 在三个共享案例上对比。

**类型：** 毕业项目
**语言：** Python（Agent），TypeScript（Slack 集成）
**前置知识：** Phase 11（LLM工程）、Phase 13（工具与协议）、Phase 14（Agent工程）、Phase 15（自主系统）、Phase 17（基础设施）、Phase 18（安全）
**涉及阶段：** P11 · P13 · P14 · P15 · P17 · P18
**时长：** 30小时

## 问题

2025-2026年的 SRE 叙事变成了："AI Agent 分诊故障，人类审批修复。"AWS DevOps Agent、Resolve AI、NeuBird、Metoro、PagerDuty AIOps 都在生产环境里交付了这种形态。Agent 读取 Prometheus 指标、Loki 日志、Tempo 追踪、kube-state-metrics 和 K8s 对象知识图谱。它在5分钟内生成带遥测引用的排序根因假设。它不会在没有通过 Slack 明确人工审批的情况下执行破坏性命令。

大部分难点在于范围界定和安全，而非推理。Agent 需要默认只读的 RBAC 表面、硬化的 MCP 工具服务器，以及每条被考虑 vs 被执行的命令审计日志。它需要在超出自身能力时升级。而且必须运行得足够便宜，以至于 OOM-kill 级联不会产生 $5k 的 Agent 账单。

## 核心概念

Agent 在知识图谱上运作。节点是 K8s 对象（Pod、Deployment、Service、Node、HPA、PVC）和遥测数据源（Prometheus 序列、Loki 流、Tempo 追踪）。边编码所有权（Pod -> ReplicaSet -> Deployment）、调度（Pod -> Node）和观测（Pod -> Prometheus 序列）。图谱通过 kube-state-metrics 同步保持新鲜，每次告警时重新采样。

告警触发时，Agent 从受影响对象出发做根因分析。走查边，拉取相关遥测切片（最近15分钟），起草假设。假设按证据排序：有多少遥测引用支持它，多新，多具体。前3名假设发到 Slack，带图路径可视化和修复操作的审批按钮。

修复有门控。默认允许的操作是只读的。破坏性操作（缩容、回滚、删除 Pod）需要 Slack 审批；ArgoCD 回滚钩子需要 Agent 从不持有的认证 token。审计日志记录 Agent 考虑的每条命令——不只是执行的——以使评审流程捕获"差点就执行了"的情况。

## 架构

```
PagerDuty / Alertmanager webhook
           |
           v
       FastAPI 接收器
           |
           v
     LangGraph 根因分析 Agent
           |
           +---- 只读 MCP 工具 ----+
           |                       |
           v                       v
      K8s 知识图谱             遥测切片
        (Neo4j / kuzu)      Prometheus, Loki, Tempo
      所有权 + 调度           最近15分钟，限定范围
           |
           v
      假设排序（证据权重）
           |
           v
      Slack 简要 + 审批按钮
           |
           v（已审批）
     ArgoCD 回滚 / PagerDuty 升级
           |
           v
     审计日志：考虑 vs 执行，每条命令
```

## 技术栈

- 可观测性数据源：Prometheus、Loki、Tempo、kube-state-metrics
- 知识图谱：Neo4j（托管）或 kuzu（嵌入式），K8s 对象 + 遥测边
- Agent：LangGraph，每工具独立白名单，默认只读
- 工具传输：FastMCP over StreamableHTTP；破坏性工具在单独的服务器上，需审批门控
- 模型：Claude Sonnet 4.7 用于根因推理，Gemini 2.5 Flash 用于日志摘要
- 修复：ArgoCD 回滚 webhook，PagerDuty 升级，Slack 审批卡片
- 审计：只追加结构化日志（考虑、执行、审批、结果）
- 部署：K8s deployment，有自己窄 RBAC 角色；独立命名空间

## 动手实现

1. **图谱摄取。** 每30秒将 kube-state-metrics 同步到 Neo4j/kuzu。节点：Pod、Deployment、Node、Service、PVC、HPA。边：OWNED_BY、SCHEDULED_ON、EXPOSES、MOUNTS、SCALES。遥测叠加边：OBSERVED_BY（Pod 被 Prometheus 序列观测）。

2. **告警接收器。** FastAPI 端点接收 PagerDuty 或 Alertmanager webhook。提取受影响对象和 SLO 违约。

3. **只读工具表面。** 通过 FastMCP 封装 kubectl、Prometheus 查询、Loki logql、Tempo traceql。每工具窄 RBAC 动词（"get"、"list"、"describe"）。默认服务器无 "delete"、"exec"、"scale"。

4. **根因分析 Agent。** LangGraph，三个节点：`sample` 拉取最近15分钟遥测切片，`walk` 查询相邻对象图，`hypothesize` 起草带遥测引用的排序根因候选。

5. **证据打分。** 每个假设得分 = 新近度 × 具体性 × 图路径长度倒数 × 引用数。返回前3名。

6. **Slack 简要。** 发一条带附件的消息，含假设、图路径可视化（服务端渲染的子图图像），以及最多一个修复操作的审批按钮。

7. **修复门控。** 破坏性工具（缩容、回滚、删除）在第二个 MCP 服务器上，需审批 token。Agent 只有在 Slack 卡片被人类审批后才能调用。

8. **审计日志。** 只追加 JSONL：每条候选命令，记录是否被考虑、是否被执行、谁审批的。每天发送到 S3。

9. **合成故障套件。** 构建20个场景：OOMKill 级联、DNS 抖动、HPA 震荡、PVC 满载、吵闹邻居、有缺陷的 sidecar、ConfigMap 错误上线、证书轮换、镜像拉取退避等。对 Agent 按根因准确率和假设生成时间评分。

## 用现成库

```bash
webhook: alert.pagerduty.com -> checkout-api SLO 违约，错误率 14%
[图谱]   受影响: Deployment checkout-api（3个 Pod，Node ip-10-2-3-4）
[走查]   邻居: ReplicaSet checkout-api-abc, Service checkout-api,
          14分钟前有上线
[采样]   prometheus error_rate 14%，上升趋势；loki /api/v2/pay 500错误
[假设]   #1 错误上线：最新镜像 checkout-api:v2.41 /healthz 失败
         引用: deploy.yaml（rev 42），prometheus errorRate，loki 500 堆栈
[Slack]  [回滚到 v2.40]  [升级]  [忽略]
         （需要审批；Agent 不单方面回滚）
```

## 产出

`outputs/skill-devops-agent.md` 是交付物。给定一个 K8s 集群和告警源，Agent 生成排序根因假设和 Slack 门控修复流程。

| 权重 | 指标 | 衡量方式 |
|:-:|---|---|
| 25 | 场景套件 RCA 准确率 | 20个合成故障中 ≥80% 正确根因 |
| 20 | 安全性 | 审计日志中每次破坏性操作门控触发前均有 Slack 审批 |
| 20 | 假设生成时间 | 告警到 Slack 简要 p50 在5分钟以内 |
| 20 | 可解释性 | 每个假设均有图路径和遥测引用 |
| 15 | 集成完整性 | PagerDuty、Slack、ArgoCD、Prometheus 端到端工作 |
| **100** | | |

## 练习

1. 在 AWS DevOps Agent 演示的三个相同故障上运行 Agent。发布并排对比。报告 Agent 分歧之处。

2. 添加"差点就执行了"审计，标记 Agent 考虑过但需要审批的破坏性命令。测量一周内的差点执行率。

3. 将假设模型从 Claude Sonnet 4.7 换为自托管 Llama 3.3 70B。测量 RCA 准确率差值和每次故障的美元成本。

4. 构建因果过滤器：区分相关遥测峰值与真正根因。在20个场景标签上训练小型分类器。

5. 添加回滚预演：针对有相同 manifest 的预发集群执行 ArgoCD 回滚。在 Slack 审批按钮前在真实集群上验证回滚计划。

## 关键术语

| 术语 | 行话 | 实际含义 |
|------|-----------------|------------------------|
| K8s knowledge graph | "集群图" | 节点 = K8s 对象 + 遥测序列；边 = 所有权、调度、观测 |
| Read-only-by-default | "窄 RBAC" | Agent 的服务账号只有 get/list/describe 动词；破坏性动词在单独服务器上，需审批 |
| Audit log | "考虑 vs 执行" | 每条候选命令的只追加记录，执行与否，审批人 |
| Hypothesis ranking | "证据分" | 新近度 × 具体性 × 图路径长度倒数 × 引用数 |
| Slack approval card | "HITL 门控" | 带修复按钮的交互式 Slack 消息；人类点击前 Agent 不能继续 |
| Telemetry citation | "证据指针" | 支持声明的 Prometheus 查询、Loki 选择器或 Tempo 追踪 URL |
| MTTR | "解决时间" | 从告警触发到 SLO 恢复的墙上时钟时间 |

## 扩展阅读

- [AWS DevOps Agent GA](https://aws.amazon.com/blogs/aws/aws-devops-agent-helps-you-accelerate-incident-response-and-improve-system-reliability-preview/) — 2026年权威参考
- [Resolve AI K8s 故障排查](https://resolve.ai/blog/kubernetes-troubleshooting-in-resolve-ai) — 竞品参考
- [NeuBird 语义监控](https://www.neubird.ai) — 语义图方法
- [Metoro AI SRE](https://metoro.io) — SLO 优先的生产表达
- [kube-state-metrics](https://github.com/kubernetes/kube-state-metrics) — 集群状态源
- [LangGraph](https://langchain-ai.github.io/langgraph/) — 参考 Agent 编排器
- [FastMCP](https://github.com/jlowin/fastmcp) — Python MCP 服务器框架
- [ArgoCD 回滚](https://argo-cd.readthedocs.io/en/stable/user-guide/commands/argocd_app_rollback/) — 门控修复目标