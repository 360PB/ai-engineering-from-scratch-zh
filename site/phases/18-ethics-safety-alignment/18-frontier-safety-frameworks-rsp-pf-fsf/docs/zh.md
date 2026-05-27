# 前沿安全框架——RSP、PF、FSF

> 三个主要实验室框架定义 2026 年前沿能力治理。Anthropic Responsible Scaling Policy v3.0（2026 年 2 月）引入分层 AI Safety Levels（ASL-1 到 ASL-5+），以 biosafety levels 为模型，ASL-3 于 2025 年 5 月对 CBRN 相关模型激活。OpenAI Preparedness Framework v2（2025 年 4 月）定义五个跟踪能力标准并将 Capabilities Reports 与 Safeguards Reports 分离。DeepMind Frontier Safety Framework v3.0（2025 年 9 月）引入包括新 Harmful Manipulation CCL 的 Critical Capability Levels。所有三个现在都包括竞争对手调整条款，允许在同行实验室发货无可比保障时延迟。跨实验室对齐仍然是结构性的，非术语性的："Capability Thresholds"、"High Capability thresholds"和"Critical Capability Levels"表示类似结构。

**类型：** 学习
**语言：** 无
**前置知识：** Phase 18 · 17（WMDP），Phase 18 · 07-09（欺骗失败）
**时长：** 约 75 分钟

## 学习目标

- 描述 Anthropic 的 ASL 层结构以及什么激活了 ASL-3。
- 命名 OpenAI Preparedness Framework v2 跟踪能力的五个标准。
- 描述 DeepMind 的 Critical Capability Level 结构和 Harmful Manipulation CCL。
- 解释竞争对手调整条款及其对竞争动态的重要性。
- 定义安全案例并描述三支柱结构（监控、不可读性、无能力）。

## 问题

第 7-17 课确立欺骗可能、双重用途能力存在、评估有局限。有前沿能力模型的实验室需要内部治理结构来：
- 定义何时需要新保障的阈值。
- 定义扩展前需要哪些评估。
- 描述安全案例的样子。
- 处理竞争动态问题（如果竞争对手发货没有保障，你怎么做？）。

三个 2025-2026 年框架是最新技术水平——不完美、进化中、对齐足以让治理问题是框架是否足够，而非它们是否存在。

## 概念

### Anthropic Responsible Scaling Policy v3.0（2026 年 2 月）

ASL 结构：
- ASL-1：不是前沿模型（被较弱于前沿基线吸收）。
- ASL-2：当前前沿基线；以通常保障部署。
- ASL-3：灾难性滥用风险大幅提高；CBRN 相关能力。2025 年 5 月激活。
- ASL-4：AI R&D-2 跨越门槛；可以自动化入门级 AI 研究的模型。
- ASL-5+：前沿 AI R&D；大幅加速有效扩展的模型。

v3.0 新增：
- 前沿安全路线图（以编辑形式公开）。
- 风险报告（季度，部分外部审查）。
- AI R&D 分解为 AI R&D-2 和 AI R&D-4。
- 一旦 AI R&D-4 跨越，需要肯定安全案例，识别来自追求错位目标的模型的错位风险。

### OpenAI Preparedness Framework v2（2025 年 4 月 15 日）

跟踪能力的五个标准：
- **似真。** 存在合理的威胁模型。
- **可测量。** 经验评估可能。
- **严重。** 危害大。
- **净新增。** 不是预先存在的风险放大。
- **瞬时或不可弥补。** 危害发生快或无法撤消。

满足全部五项的能力被跟踪。其他的不是。

其他 PF v2 结构：
- 将 Capabilities Reports（模型能做什么）与 Safeguards Reports（存在什么控制）分离。
- Safety Advisory Group 审查。
- 领导批准；董事会安全与安保委员会监督。
- "调整条款"：如果另一家实验室发货无可比保障，OpenAI 可能减少要求。

### DeepMind Frontier Safety Framework v3.0（2025 年 9 月）

按领域的 Critical Capability Levels（CCLs）：
- 生物武器提升
- 网络提升
- ML R&D 加速
- 有害操控（新 v3.0）：可以大幅改变高风险背景下信念/行为的模型。

v2.0（2025 年 2 月）添加了欺骗性对齐部分和 ML R&D CCL 更高安全级别。

### 跨实验室对齐

- Anthropic "Capability Thresholds"。
- DeepMind "Critical Capability Levels"。
- OpenAI "High Capability thresholds"。

没有行业标准术语。结构上对齐：前沿能力三层与发布评估标准。全部包括竞争对手调整条款（2025 年起）。

### 安全案例

安全案例是部署在最坏情况假设下可接受的书面论证。标准结构针对三个支柱：

- **监控。** 我们能检测到不良行为吗？
- **不可读性。** 模型是否缺乏执行造成危害的连贯计划的能力？
- **无能力。** 模型是否缺乏造成相关伤害的能力？

不同安全案例针对不同支柱。对于 ASL-3 CBRN 案例，**无能力**（通过遗忘）是主要目标。对于欺骗性对齐，监控和不可读性是目标。对于网络提升，全部三个相关。

### 竞争动态问题

竞争对手调整条款是有争议的。批评者认为它们创造了逐底竞争：如果所有三个实验室在竞争对手违约时减少要求，均衡向违约移动。捍卫者认为替代（单方面保障）在违约实验室安全意识较弱时产生更差结果。

UK AISI、US CAISI 和 EU AI Office（第 24 课）是外部治理对应方。实验室框架是自愿的；监管框架正在出现。

### 为什么这在 Phase 18 中重要

第 17-18 课是测量和治理层，位于欺骗和红队分析之上。第 19-24 课涵盖福利、偏见、隐私、水印和监管结构。第 28 课绘制研究生态系统（MATS、Redwood、Apollo、METR）操作化评估。

## 使用它

本课无代码。读三个主要来源：RSP v3.0、PF v2、FSF v3.0。将每个实验室的层结构相互映射，识别每个实验室定义而其他实验室没有的阈值。

## 发布它

本课生成 `outputs/skill-framework-diff.md`。给定安全框架或发布说明，将其阈值定义、所需评估和安全案例结构与 RSP v3.0、PF v2、FSF v3.0 进行比较，并标记跨实验室差距。

## 练习

1. 读 RSP v3.0、PF v2 和 FSF v3.0。编制每个实验室的 CBRN 阈值、AI R&D 阈值和部署前所需评估的表格。

2. 竞争对手调整条款在所有三个框架中（2025+）。写一段赞成它；写一段反对。识别每种立场依赖的假设。

3. 为模型跨越 Anthropic 的 AI R&D-4 阈值设计安全案例。命名三个支柱（监控、不可读性、无能力）每个需要哪些证据。

4. DeepMind 的 FSF v3.0 引入 Harmful Manipulation CCL。提出三个经验测量表明模型已跨越此阈值。

5. 读 METR 的"Common Elements of Frontier AI Safety Policies"（2025）。命名三个最强跨实验室趋同和两个最大分歧。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|------------------------|
| RSP | Anthropic 的框架 | Responsible Scaling Policy；ASL 层；v3.0 2026 年 2 月 |
| PF | OpenAI 的框架 | Preparedness Framework；五个标准；v2 2025 年 4 月 |
| FSF | DeepMind 的框架 | Frontier Safety Framework；CCL；v3.0 2025 年 9 月 |
| ASL-3 | biosafety level 3 类比 | Anthropic 的 CBRN 相关能力层；2025 年 5 月激活 |
| CCL | 关键能力级别 | DeepMind 的阈值结构；按领域 |
| 安全案例 | 正式论证 | 部署在最坏情况 U 下可接受的书面论证 |
| 调整条款 | 竞争对手违约允许 | 如果竞争对手发货无可比保障，减少要求的框架条款 |

## 延伸阅读

- [Anthropic — Responsible Scaling Policy v3.0（2026 年 2 月）](https://www.anthropic.com/responsible-scaling-policy) — ASL 层、路线图、AI R&D 分解
- [OpenAI — Updating the Preparedness Framework（2025 年 4 月 15 日）](https://openai.com/index/updating-our-preparedness-framework/) — 五个标准、调整条款
- [DeepMind — Strengthening our Frontier Safety Framework（2025 年 9 月）](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/) — CCL v3.0、有害操控
- [METR — Common Elements of Frontier AI Safety Policies（2025）](https://metr.org/blog/2025-03-26-common-elements-of-frontier-ai-safety-policies/) — 跨实验室比较