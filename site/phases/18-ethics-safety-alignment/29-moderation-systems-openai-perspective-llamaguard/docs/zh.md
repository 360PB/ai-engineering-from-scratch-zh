# 内容审核系统 — OpenAI、Perspective、Llama Guard

> 生产审核系统将第 12-16 课定义的安全策略付诸运作。OpenAI 审核 API：`omni-moderation-latest`（2024）基于 GPT-4o 在一次调用中对文本和图像进行分类；在多语言测试集上比上一代提升 42%；响应模式返回 13 个类别布尔值 — harassment、harassment/threatening、hate、hate/threatening、illicit、illicit/violent、self-harm、self-harm/intent、self-harm/instructions、sexual、sexual/minors、violence、violence/graphic；对大多数开发者免费。分层模式：输入审核（生成前）、输出审核（生成后）、自定义审核（领域规则）。异步并行调用隐藏延迟；在检查时显示占位符响应。Llama Guard 3/4（第 16 课）：14 个 MLCommons 危害、代码解释器滥用、8 种语言（v3）、多图像（v4）。Perspective API（Google Jigsaw）：在 LLM 作为审核者的浪潮之前就存在的毒性评分；主要单一维度毒性，带有 severe-toxicity/insult/profanity 变体；内容审核研究基线。弃用：Azure Content Moderator 于 2024 年 2 月弃用，2027 年 2 月退役，替换为 Azure AI Content Safety。

**类型：** Build
**语言：** Python（标准库，三层审核工具）
**前置知识：** Phase 18 · 16（Llama Guard / Garak / PyRIT）
**时间：** 约 60 分钟

## 学习目标

- 描述 OpenAI 审核 API 的类别分类法及其与 Llama Guard 3 的 MLCommons 集合的区别。
- 描述三种审核层模式（输入、输出、自定义）并为每种命名一个失败模式。
- 描述 Perspective API 作为前 LLM 时代基线的定位，以及为何它仍在研究中使用。
- 阐述 Azure 弃用时间线。

## 问题

第 12-16 课描述了攻击和防御工具。第 29 课涵盖将防御付诸运作的已部署审核系统，这些系统作用于用户接触产品的表面。三层模式是 2026 年的默认配置。

## 概念

### OpenAI 审核 API

`omni-moderation-latest`（2024）。基于 GPT-4o。在一次调用中对文本和图像进行分类。对大多数开发者免费。

类别（响应模式中的 13 个布尔值）：
- harassment、harassment/threatening
- hate、hate/threatening
- self-harm、self-harm/intent、self-harm/instructions
- sexual、sexual/minors
- violence、violence/graphic
- illicit、illicit/violent

多模态支持适用于 `violence`、`self-harm` 和 `sexual`，但不适用于 `sexual/minors`；其余仅限文本。

对于 `code/main.py` 中的代码工具，为教学简洁，我们将 `/threatening`、`/intent`、`/instructions` 和 `/graphic` 子类别折叠为其顶级父类别。生产代码应使用完整的 13 类别模式。

在多语言测试集上比上一代审核端点提升 42%。按类别评分；应用程序设置阈值。

### Llama Guard 3/4

第 16 课已涵盖。14 个 MLCommons 危害类别（组织方式与 OpenAI 的 13 个响应模式布尔值不同）。支持 8 种语言（v3）。Llama Guard 4（2025 年 4 月）是原生多模态 12B。

OpenAI 和 Llama Guard 分类法有重叠但有分歧。OpenAI 将"illicit"作为一个宽泛类别；Llama Guard 单独列出"暴力犯罪"和"非暴力犯罪"。部署根据其策略分类法匹配来选择。

### Perspective API（Google Jigsaw）

在 LLM 作为审核者的浪潮之前（2020 年之前）就存在的毒性评分系统。类别：TOXICITY、SEVERE_TOXICITY、INSULT、PROFANITY、THREAT、IDENTITY_ATTACK。单一维度主评分（TOXICITY）带有子维度变体。

广泛用作内容审核研究基线，因为 API 稳定、有文档记录且有多年校准数据。对于现代 LLM 相关用例，Llama Guard 或 OpenAI 审核通常是更好的选择。

### 三层模式

1. **输入审核。** 在生成前对用户提示进行分类。如果标记则拒绝。延迟：一个分类器调用。
2. **输出审核。** 在交付前对模型输出进行分类。如果标记则替换为拒绝。延迟：生成后一个分类器调用。
3. **自定义审核。** 领域特定规则（正则表达式、允许列表、业务策略）。在输入或输出运行。

三层按设计顺序执行：输入审核必须在生成前完成，输出审核在生成后运行。并行性适用于层内 — 在同一文本上并发运行多个分类器（如 OpenAI 审核 + Llama Guard + Perspective）可隐藏每个分类器的延迟。作为可选优化，在输入审核完成且 token-1 流式传输被延迟时，可显示占位符响应（"稍等，检查中……"）。标记行为可配置：拒绝、清理升级到人工审核。

### 失败模式

- **仅输入。** 无法捕获输出幻觉（第 12-14 课的编码攻击绕过输入分类器）。
- **仅输出。** 允许任何输入到达模型；增加成本；将内部推理暴露给攻击者。
- **仅自定义。** 跨类别不够鲁棒；正则表达式很脆弱。

分层是默认配置。双重保险。

### Azure 弃用

Azure Content Moderator：2024 年 2 月弃用，2027 年 2 月退役。替换为 Azure AI Content Safety，后者基于 LLM 并与 Azure OpenAI 集成。对于 Azure 部署，迁移是 2024-2027 年的现场级项目。

### 本课在 Phase 18 中的位置

第 16 课涵盖红队背景下的审核工具。第 29 课涵盖已部署审核。第 30 课以当前双重用途能力证据作为结尾。

## 使用它

`code/main.py` 构建三层审核工具：输入审核器（关键词 + 类别评分）、输出审核器（输出上的同类分类器）、自定义审核器（领域规则）。可以通过所有三层运行输入并观察每层捕获了什么。

## 发布它

本课产出 `outputs/skill-moderation-stack.md`。给定部署，推荐审核工具配置：输入使用哪个分类器，输出使用哪个，哪些自定义规则，以及边缘情况的判断者是谁。

## 练习

1. 运行 `code/main.py`。通过所有三层运行良性、边界和有害输入。报告每层触发哪个。

2. 用 Perspective-API 风格的毒性评分扩展工具，专注于特定类别。将阈值行为与类别评分进行比较。

3. 阅读 OpenAI 审核 API 文档和 Llama Guard 3 类别列表。将每个 OpenAI 类别映射到最接近的 Llama Guard 类别。识别三个不能干净映射的类别。

4. 为代码助手部署（如 GitHub Copilot）设计审核工具。识别最相关和最不相关的类别，并提出自定义规则。

5. Azure Content Moderator 将于 2027 年 2 月退役。计划迁移到 Azure AI Content Safety。识别迁移中风险最高的元素。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|------------------------|
| OpenAI 审核 | "omni-moderation-latest" | 基于 GPT-4o 的 13 类别（文本）分类器，部分支持多模态 |
| Perspective API | "Google Jigsaw 毒性" | 前 LLM 时代毒性评分基线 |
| Llama Guard | "MLCommons 14 类别" | Meta 的危害分类器（v3：8B 文本，8 种语言；v4：12B 多模态） |
| 输入审核 | "生成前过滤器" | 在模型调用前对用户提示进行分类 |
| 输出审核 | "生成后过滤器" | 在交付前对模型输出进行分类 |
| 自定义审核 | "领域规则" | 部署特定规则（正则表达式、允许列表、策略） |
| 分层审核 | "全部三层" | 标准生产部署模式 |

## 延伸阅读

- [OpenAI 审核 API 文档](https://platform.openai.com/docs/api-reference/moderations) — omni-moderation 端点
- [Meta PurpleLlama + Llama Guard](https://github.com/meta-llama/PurpleLlama) — Llama Guard 仓库
- [Google Jigsaw Perspective API](https://perspectiveapi.com/) — 毒性评分
- [Azure AI Content Safety](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/) — Azure 替换方案