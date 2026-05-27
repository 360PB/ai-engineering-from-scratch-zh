# 合规 — SOC 2、HIPAA、GDPR、PCI-DSS、EU AI Act、ISO 42001

> 多框架覆盖是 2026 年企业交易的门槛。**EU AI Act**：2024 年 8 月 1 日生效。高风险要求 2026 年 8 月 2 日执行。罚款：高风险系统义务最高 €1500 万或全球年营业额 3%（Art. 99(4)）；禁止 AI 实践最高 €3500 万或 7%（Art. 99(3)）。服务于欧盟用户则全球适用。**Colorado AI Act**：2026 年 6 月 30 日生效（SB25B-004 从 2 月推迟）——高风险系统影响评估，AI 决策申诉权。弗吉尼亚州对信贷/就业/住房/教育类似。**SOC 2 Type II**：B2B AI 实事（Type II，不是 Type I，金融科技用）。**GDPR**：最大有记录 AI 特异性罚款是 €3050 万针对 Clearview AI（荷兰 DPA，2024 年 9 月）；意大利 Garante 2024 年 12 月对 OpenAI 处 €1500 万（2026 年 3 月上诉推翻，裁决仍审查中）。推理时实时 PII 删除是可辩护标准；后处理清理不够。**HIPAA**：医疗——无 BAA 不得将 PHI 发送到外部 AI 服务。**PCI-DSS**：AI 交互层覆盖需要配置 + 合同协议，非自动。**ISO 42001**：新兴 AI 治理标准，与 ISO 27001 一起成为越来越常见的采购要求。参考概况：OpenAI 持有 SOC 2 Type 2、ISO/IEC 27001:2022、ISO/IEC 27701:2019、GDPR/CCPA/HIPAA（BAA）/FERPA、ChatGPT 支付组件 PCI-DSS。跨框架映射削减审计疲劳：访问控制映射到 ISO 27001 A.5.15-5.18、GDPR Art. 32、HIPAA §164.312(a)。

**类型：** 精读
**语言：** （Python 可选——合规是政策 + 流程，不是代码）
**前置要求：** Phase 17 · 25（安全）、Phase 17 · 13（可观测性）
**时长：** 约 60 分钟

## 学习目标

- 枚举 2026 年 LLM 产品相关的七个框架并匹配到各自客户群。
- 引用 EU AI Act 执行时间线（2024 年 8 月生效；高风险 2026 年 8 月执行）和两档罚款（高风险义务 €1500 万/3%，禁止实践 €3500 万/7%）。
- 解释为什么后处理 PII 清理对 GDPR 不够，并说出推理层实时删除是可辩护标准。
- 描述跨框架控制映射（如访问控制映射到 ISO 27001 A.5.15-5.18 + GDPR Art. 32 + HIPAA §164.312(a)）。

## 背景问题

企业客户采购问你要 SOC 2 Type II、GDPR、HIPAA BAA、ISO 27001 和"EU AI Act 合规声明"。你的团队有 SOC 2 Type I。离 Type II 还有六个月，还没启动 GDPR Article 30 记录。

多框架覆盖不是 LLM 问题——是企业 SaaS 问题，外加 LLM 特异覆盖。2026 年采购团队想要一个矩阵，每行一个框架，每列一个控制，不是 PDF。

## 核心概念

### 七个框架

| 框架 | 范围 | LLM 特有要求 |
|------|------|-------------|
| SOC 2 Type II | B2B SaaS 基线 | 6-12 个月内运营的控制审计 |
| HIPAA | 美国医疗 | BAA 必需；无 PHI 协议不得离开基础设施 |
| GDPR | 欧盟用户 | 实时 PII 删除；数据主体权利；Article 30 记录 |
| PCI-DSS | 支付数据 | AI 接触支付需配置 + 合同 |
| EU AI Act | 服务欧盟用户 | 风险层级分类；高风险系统：符合性评估、文档、日志 |
| Colorado AI Act | 服务科罗拉多居民 | 影响评估；AI 决策申诉权 |
| ISO 42001 | AI 治理 | 新兴；与 ISO 27001 搭配 |

### EU AI Act 时间线

- 2024 年 8 月 1 日：生效。
- 2025 年 2 月 2 日：禁止 AI 实践执行。
- 2026 年 8 月 2 日：高风险系统执行（符合性评估、文档、日志）。
- 2027 年 8 月：协调立法下高风险系统产品。

风险层级：不可接受（禁止）、高风险（符合性 + 日志）、有限风险（透明度）、最小风险（无约束）。大多数 B2B LLM SaaS 是有限风险；高风险适用于就业、信贷、教育、执法、移民、基本服务。

罚款（Article 99）：高风险系统义务违反应 €1500 万或全球年营业额 3%（Art. 99(4)）；禁止 AI 实践违反应 €3500 万或 7%（Art. 99(3)）；以较高者为准。

### GDPR — 实时删除是标准

后处理清理（让 LLM 先看到数据再删除 PII）不是可辩护姿态——模型已经看到了数据。推理层实时删除是 2026 年标准：

- LLM 调用前实体识别。
- 一致性 token 化（Mesh 方法）保留语义。
- 仅存储删除后的提示词 + 经同意的 opt-in raw。

近期执法：€3050 万针对 Clearview AI（荷兰 DPA，2024 年 9 月）是至今最大有记录 AI 特异性 GDPR 罚款；€1500 万针对 OpenAI（意大利 Garante，2024 年 12 月）是最大 LLM 特异性罚款，虽 2026 年 3 月上诉推翻，裁决仍在审查。后处理声称在审计中失败。

### HIPAA — BAA 不可选

无 signed Business Associate Agreement 不得将 PHI 发送给外部 AI 服务。三大超大规模 LLM 平台（Bedrock、Azure OpenAI、Vertex）都提供 BAA。OpenAI direct API 提供 BAA。Anthropic direct API 提供 BAA。使用前确认。

### SOC 2 Type II

Type I：控制设计并记录。
Type II：控制在 6-12 个月内有效运营。

2026 年 B2B 采购默认要 Type II。Type I 是入门；Type II 是门槛。

常见审计驱动：访问日志（谁看了什么）、变更管理（如何部署的）、风险评估（季度）、事件响应（测试过？）。Phase 17 · 25 的审计日志直接复用。

### 跨框架映射

一个访问控制策略满足多框架控制：

| 控制 | 框架 |
|------|------|
| 访问日志 | ISO 27001 A.5.15-5.18、GDPR Art. 32、HIPAA §164.312(a) |
| 变更管理 | ISO 27001 A.8.32、PCI DSS Req. 6、HIPAA 违约通知范围 |
| 传输中加密 | ISO 27001 A.8.24、GDPR Art. 32、HIPAA §164.312(e) |
| 密钥管理 | ISO 27001 A.8.19、PCI DSS Req. 8、SOC 2 CC6.1 |

合规工具（Drata、Vanta、Secureframe）自动化此映射。大规模值得投入。

### ISO 42001 — 新兴

2023 年底发布。与 ISO 27001 一起成为越来越常见的采购要求。AI 治理框架，包括风险管理、数据质量、透明度、人工监督。

### OpenAI 的参考概况

OpenAI 持有 SOC 2 Type 2、ISO/IEC 27001:2022、ISO/IEC 27701:2019、GDPR/CCPA/HIPAA（BAA）/FERPA、ChatGPT 支付组件 PCI-DSS。这是 2026 年企业交易的门槛。

### 必须记住的数字

- EU AI Act 罚款：高风险义务最高 €1500 万/3%（Art. 99(4)）；禁止实践最高 €3500 万/7%（Art. 99(3)）。
- EU AI Act 高风险执行：2026 年 8 月 2 日。
- 最大有记录 AI 特异性 GDPR 罚款：€3050 万，Clearview AI（荷兰 DPA，2024 年 9 月）。
- 最大 LLM 特异性 GDPR 罚款：€1500 万，OpenAI（意大利 Garante，2024 年 12 月；2026 年 3 月上诉推翻）。
- SOC 2 Type II 窗口：6-12 个月运营控制。
- Colorado AI Act 生效日期：2026 年 6 月 30 日（SB25B-004 从 2 月推迟）。

## 用现成库

`code/main.py` 是 Python 中的合规映射电子表格——给定一个控制，列出满足的框架。

## 产出

本课产出 `outputs/skill-compliance-matrix.md`。给定客户群和地区，指定所需框架和控制。

## 练习

1. 你的第一个企业客户要求 SOC 2 Type II、HIPAA BAA、EU AI Act 声明。要赢得交易，最低可行合规姿态是什么？
2. 将三个假设 LLM 产品分类到 EU AI Act 风险层级。高风险有哪些变化？
3. 你意外将 PHI 发给了无 BAA 的提供商。走过事件响应流程。
4. 论证 ISO 42001 对中型市场 AI 供应商在 2026 年是否"必要"。
5. 将 LLM 审计日志字段（Phase 17 · 25）映射到至少三个框架控制。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| SOC 2 Type II | "审计控制" | 控制在 6-12 个月内独立认证有效运营 |
| HIPAA BAA | "医疗合同" | Business Associate Agreement；PHI 必需 |
| GDPR | "欧盟隐私" | 实时 PII 删除是 2026 年可辩护标准 |
| EU AI Act | "欧盟 AI 规则" | 高风险 2026 年 8 月执行；高风险义务 €1500 万/3%——禁止实践 €3500 万/7% |
| Colorado AI Act | "美国 AI 州法" | 2026 年 6 月 30 日生效（SB25B-004 推迟）；影响评估 |
| ISO 42001 | "AI 治理" | AI 风险+透明度新兴框架 |
| ISO 27001 | "安全 ISMS" | 信息安全管理体系基线 |
| 符合性评估 | "EU AI 文档包" | 高风险要求：文档、测试、日志 |
| 跨框架映射 | "一控多框架" | 单一政策满足多框架控制 |

## 扩展阅读

- [OpenAI Security and Privacy](https://openai.com/security-and-privacy/) — 参考合规概况。
- [GuardionAI — LLM Compliance 2026: ISO 42001, EU AI Act, SOC 2, GDPR](https://guardion.ai/blog/llm-compliance-guide-iso-42001-eu-ai-act-soc2-gdpr-2026)
- [Dsalta — SOC 2 Type 2 Audit Guide 2026: 10 AI Controls](https://www.dsalta.com/resources/ai-compliance/soc-2-type-2-audit-guide-2026-10-ai-powered-controls-every-saas-team-needs)
- [EU AI Act official text](https://eur-lex.europa.eu/eli/reg/2024/1689/oj) — 原始文本。
- [Colorado AI Act](https://leg.colorado.gov/bills/sb24-205) — 原始文本。
- [ISO/IEC 42001:2023](https://www.iso.org/standard/81230.html) — AI 管理系统标准。