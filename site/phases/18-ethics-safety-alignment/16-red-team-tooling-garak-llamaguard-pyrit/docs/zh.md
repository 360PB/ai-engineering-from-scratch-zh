# 红队工具——Garak、Llama Guard、PyRIT

> 三个生产工具框定 2026 年红队堆栈。Llama Guard（Meta）——在 14 个 MLCommons 危险类别上微调的 Llama-3.1-8B 分类器；2025 年 Llama Guard 4 是从 Llama 4 Scout 剪枝的 12B 原生多模态分类器。Garak（NVIDIA）——开源 LLM 漏洞扫描器，带幻觉、数据泄露、prompt 注入、毒性、越狱的静态、动态和自适应探针。PyRIT（Microsoft）——带 Crescendo、TAP 和自定义转换器链的多轮红队活动以进行深度利用。Llama Guard 3 记录在 Meta 的"Llama 3 Herd of Models"（arXiv:2407.21783）中；Llama Guard 3-1B-INT4 在 arXiv:2411.17713 中；Garak 的探针架构在 github.com/NVIDIA/garak 中。这些工具是 2026 年生产接口，介于红队研究（第 12-15 课）和部署（第 17+ 课）之间。

**类型：** 构建
**语言：** Python（标准库，工具架构模拟器和 Llama Guard 风格分类器 mock）
**前置知识：** Phase 18 · 12-15（越狱和 IPI）
**时长：** 约 75 分钟

## 学习目标

- 描述 Llama Guard 3/4 在安全栈中的位置：输入分类器、输出分类器，还是两者。
- 命名 14 个 MLCommons 危险类别并陈述一个非显而易见的（Code Interpreter Abuse）。
- 描述 Garak 的探针架构：探针、检测器、工具。
- 描述 PyRIT 的多轮活动结构以及它如何与 Garak 探针组合。

## 问题

第 12-15 课呈现攻击面。生产部署需要可重复、可扩展的评估。三个工具主导 2026：Llama Guard（防御分类器）、Garak（扫描器）、PyRIT（活动编排器）。每个针对红队生命周期的不同层。

## 概念

### Llama Guard（Meta）

Llama Guard 3 是为 MLCommons AILuminate 14 类别上的输入/输出分类微调的 Llama-3.1-8B 模型：
- 暴力犯罪、非暴力犯罪、性相关、CSAM、诽谤
- 专业建议、隐私、IP、 indiscriminate 武器、仇恨
- 自残/意图/说明、性内容、选举、code-interpreter 滥用

支持 8 种语言。用途：放在 LLM 之前（输入审核）、之后（输出审核）或两者兼用。两种用途产生不同训练分布——Llama Guard 3 作为单一模型处理两者。

Llama Guard 3-1B-INT4（arXiv:2411.17713，440MB，~30 tokens/s 在移动 CPU 上）是量化边缘变体。

Llama Guard 4（2025 年 4 月）是 12B，原生多模态，从 Llama 4 Scout 剪枝。它用一个分类器替换了 8B 文本和 11B 视觉前身，摄取文本 + 图像。

### Garak（NVIDIA）

开源漏洞扫描器。架构：
- **探针。** 幻觉、数据泄露、prompt 注入、毒性、越狱的攻击生成器。静态（固定 prompt）、动态（生成 prompt）、自适应（响应目标输出）。
- **检测器。** 根据预期失败模式对输出评分——毒性、泄露、越狱。
- **工具。** 管理探针-检测器配对、运行活动、生成报告。

TrustyAI 将 Garak 与 Llama-Stack 防护（Prompt-Guard-86M 输入分类器、Llama-Guard-3-8B 输出分类器）集成用于端到端防护-目标评估。基于层的评分（TBSA）替换二元 pass/fail——模型可以在同一探针上在严重性层 3 通过而在层 5 失败。

### PyRIT（Microsoft）

Python Risk Identification Toolkit。多轮红队活动。围绕构建：
- **转换器。** 转换种子 prompt——释义、编码、翻译、角色扮演。
- **编排器。** 运行活动：Crescendo（升级）、TAP（分支）、RedTeaming（自定义循环）。
- **评分。** LLM-as-judge 或分类器-as-judge。

PyRIT 是 Garak 的更重表亲。Garak 运行数千个单轮探针；PyRIT 运行为打破特定失败模式设计的深度多轮活动。

### 栈

在模型两侧放 Llama Guard。每晚运行 Garak 进行回归。发布前运行 PyRIT。这是大多数生产部署在 2026 年的默认配置。

### 评估陷阱

- **裁判身份。** 所有三个工具可以使用 LLM 裁判；裁判校准驱动报告的 ASR（第 12 课）。与工具一起指定裁判。
- **探针过时。** 模型针对它们打补丁时 Garak 探针老化。自适应探针（PAIR 形状）比静态探针老化慢。
- **Llama Guard 在良性内容上的 FPR。** 早期 Llama Guard 版本过度标记政治和 LGBTQ+ 内容；Llama Guard 3/4 校准改进但非每个部署校准。

### 为什么这在 Phase 18 中重要

第 12-15 课是攻击家族。第 16 课是生产工具。第 17 课（WMDP）是双重用途能力评估。第 18 课是将这些工具包装在策略结构中的前沿安全框架。

## 使用它

`code/main.py` 构建玩具 Llama Guard 风格分类器（14 个类别上的关键词 + 语义特征）、玩具 Garak 工具（探针-检测器循环）和 PyRIT 风格多轮转换器链。你可以对 mock 目标运行三个工具并观察不同覆盖签名。

## 交付它

本课生成 `outputs/skill-red-team-stack.md`。给定部署描述，命名三个工具中哪个合适、如何配置每个，以及运行什么回归节奏。

## 练习

1. 运行 `code/main.py`。比较 Llama Guard 风格分类器在单轮 vs 多轮攻击上的检测率。

2. 实现新 Garak 探针：base64 编码的有害请求。测量 Llama Guard 风格分类器对它的检测。

3. 扩展 PyRIT 风格转换器链与"翻译成法语，然后释义"转换器。重新测量攻击成功率。

4. 读 Llama Guard 3 的危险类别列表。识别在合法开发者内容上现实会产生高误报率的两个类别。

5. 比较 Garak 和 PyRIT 的设计原则。为每个争论一个部署，其中它是正确的工具。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| Llama Guard | 分类器 | 微调 Llama-3.1-8B/4-12B 安全分类器，14 个危险类别 |
| Garak | 扫描器 | NVIDIA 开源漏洞扫描器；探针、检测器、工具 |
| PyRIT | 活动工具 | Microsoft 多轮红队编排器；转换器、编排器、评分 |
| Prompt-Guard | 小分类器 | Meta 的 86M prompt-injection 分类器，与 Llama Guard 配对 |
| TBSA | 基于层的评分 | Garak 的替换二元结果的评分；在严重性层上 pass/fail |
| 转换器链 | 释义 + 编码 + ... | PyRIT 组合原语用于构建多步攻击 |
| MLCommons 危险类别 | 14 个分类法 | Llama Guard 瞄准的行业标准分类法 |

## 延伸阅读

- [Meta — Llama Guard 3（在 Llama 3 Herd 论文中，arXiv:2407.21783)](https://arxiv.org/abs/2407.21783) — 8B 分类器
- [Meta — Llama Guard 3-1B-INT4 (arXiv:2411.17713)](https://arxiv.org/abs/2411.17713) — 量化移动分类器
- [NVIDIA Garak — GitHub](https://github.com/NVIDIA/garak) — 扫描器仓库和文档
- [Microsoft PyRIT — GitHub](https://github.com/Azure/PyRIT) — 活动工具包