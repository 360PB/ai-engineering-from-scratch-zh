# 对齐研究生态系统 — MATS、Redwood、Apollo、METR

> 五个组织定义了 2026 年非实验室对齐研究层。MATS（ML Alignment & Theory Scholars）：2021 年底以来 527+ 研究员，180+ 论文，10K+ 引用，h-index 47；2024 年夏季队列作为 501(c)(3) 成立，约 90 名学者和 40 名导师；2025 年前校友中 80% 从事安全/安保工作，200+ 人在 Anthropic、DeepMind、OpenAI、英国 AISI、RAND、Redwood、METR、Apollo。Redwood Research：由 Buck Shlegeris 创立的应用对齐实验室；引入 AI Control（第 10 课）；与英国 AISI 合作控制安全案例。Apollo Research：前沿实验室的部署前欺骗评估；编写了上下文欺骗（第 8 课）和迈向 AI 欺骗安全案例。METR（模型评估与威胁研究）：基于任务的能力评估、自主任务时间范围研究；"前沿 AI 安全政策的共同要素"比较实验室框架。Eleos AI Research：模型福利部署前评估（第 19 课）；进行 Claude Opus 4 福利评估。

**类型：** Learn
**语言：** 无
**前置知识：** Phase 18 · 01-27（Phase 18 先修课程）
**时间：** 约 45 分钟

## 学习目标

- 识别非实验室对齐研究生态系统的五个组织及其核心产出。
- 描述 MATS 的规模（学者、论文、h-index）及其作为人才管道的作用。
- 描述 Redwood 的 AI Control 议程及其与英国 AISI 的合作。
- 描述 METR 基于任务的评估方法。

## 问题

前沿实验室（第 18 课）在内部进行安全评估并发布选定的结果。实验室外部的生态系统是评估被验证、新的失败模式首次被发现、以及人才被培训的地方。了解生态系统有助于理解哪些研究发现被谁信任。

## 概念

### MATS（ML Alignment & Theory Scholars）

2021 年底启动。研究师徒计划；学者与资深研究员共同研究特定对齐问题，为期 10-12 周。

规模（2026 年）：
- 自成立以来 527+ 研究员。
- 180+ 论文发表。
- 10K+ 引用。
- h-index 47。
- 2024 年夏季：90 名学者 + 40 名导师；注册为 501(c)(3)。

职业去向：2025 年前校友中约 80% 从事安全/安保工作。200+ 人在 Anthropic、DeepMind、OpenAI、英国 AISI、RAND、Redwood、METR、Apollo。

### Redwood Research

应用对齐实验室。由 Buck Shlegeris 创立。引入了 AI Control 议程（第 10 课）。与英国 AISI 合作控制安全案例。为 DeepMind 和 Anthropic 提供评估设计咨询。

代表性论文：Greenblatt、Shlegeris 等人，"AI Control"（arXiv:2312.06942，ICML 2024）；对齐伪装（Greenblatt、Denison、Wright 等人，arXiv:2412.14093，与 Anthropic 合作）。

风格：特定的威胁模型、最坏情况对手、可进行压力测试的具体协议。

### Apollo Research

前沿实验室的部署前欺骗评估。编写了上下文欺骗（第 8 课，arXiv:2412.04984）。参与 2025 年 OpenAI 反欺骗训练合作。产出《迈向 AI 欺骗安全案例》（2024 年）。

风格：在可能出现欺骗的智能体设置中进行评估；三支柱分解（未对齐、目标导向性、情境感知）。

### METR（模型评估与威胁研究）

基于任务的能力评估。自主任务完成时间范围研究。"前沿 AI 安全政策的共同要素"（metr.org/common-elements，2025 年）比较实验室框架。

与 Apollo 合作 AI 欺骗安全案例草图。

风格：长时间范围任务评估、实证能力测量、框架综合。

### Eleos AI Research

模型福利部署前评估。为第 5.3 节系统卡中记录的 Claude Opus 4 福利评估提供了外部方法检查。为第 19 课福利相关声明提供外部方法检验。

### 流动路径

MATS 培养研究员。毕业生流向 Anthropic、DeepMind、OpenAI（实验室安全团队）或 Redwood、Apollo、METR、Eleos（外部评估）。外部评估员与实验室以及英国 AISI/CAISI 合作。出版物反馈给 MATS 下一代学员。

### 为何这一层重要

单一来源的评估不可靠：实验室评估自己的模型存在结构性利益冲突。外部评估员可以提出和验证实验室可能少报的失败模式。2024 年的睡眠体特工论文（第 7 课）是 Anthropic + Redwood；对齐伪装是 Anthropic + Redwood；上下文欺骗是 Apollo；反欺骗是 Apollo + OpenAI。多组织结构是质量控制。

### 本课在 Phase 18 中的位置

第 7-11 课引用了 Redwood 和 Apollo 的工作；第 18 课引用了 METR 的框架比较；第 19 课引用了 Eleos。第 28 课是生态系统组织地图，Phase 其余部分依赖于此。

## 使用它

无代码。阅读 METR 的"前沿 AI 安全政策的共同要素"作为外部综合如何为实验室内部政策工作增加价值的示例。

## 发布它

本课产出 `outputs/skill-ecosystem-map.md`。给定一个对齐声明或评估，识别组织、发表场所和方法风格，并与已知对应组织进行交叉检查。

## 练习

1. 从第 7-15 课中选择一篇论文，识别涉及的组织的个人作者。交叉检查作者是否是 MATS 校友以及当前生态系统隶属关系。

2. 阅读 METR 的"前沿 AI 安全政策的共同要素"。识别他们强调的三个跨实验室趋同点和两个最大分歧点。

3. MATS 职业去向中约 80% 从事安全/安保。论证这种选择压力是适应性的（训练该领域）还是偏见的（过滤掉异端立场）。

4. Redwood 和 Apollo 都做控制/欺骗工作但风格不同。选择一个失败模式，描述每个会如何调查。

5. Eleos AI 是唯一专注于模型福利的组织。设计一个专注于不同福利相关问题（如认知自由、机器人具身等）的假设第二个组织，并阐述其方法。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|------------------------|
| MATS | "师徒计划" | ML Alignment & Theory Scholars；2021 年以来 527+ 研究员 |
| Redwood Research | "控制实验室" | 应用对齐；AI Control 作者；英国 AISI 合作伙伴 |
| Apollo Research | "欺骗评估" | 前沿实验室的部署前欺骗评估 |
| METR | "任务时间范围评估" | 基于任务的能力评估；框架综合 |
| Eleos AI | "福利实验室" | 模型福利部署前评估 |
| 人才管道 | "MATS -> 实验室" | MATS 毕业生流向 Anthropic、DM、OpenAI、Redwood、Apollo、METR |
| 外部评估 | "非实验室检查" | 非模型生产者进行的评估；增加可信度 |

## 延伸阅读

- [MATS (ML Alignment & Theory Scholars)](https://www.matsprogram.org/) — 师徒计划
- [Redwood Research](https://www.redwoodresearch.org/) — AI Control 论文
- [Apollo Research](https://www.apolloresearch.ai/) — 欺骗评估
- [METR — Common Elements of Frontier AI Safety Policies](https://metr.org/blog/2025-03-26-common-elements-of-frontier-ai-safety-policies/) — 框架比较
- [Eleos AI Research](https://www.eleosai.org/research) — 模型福利方法