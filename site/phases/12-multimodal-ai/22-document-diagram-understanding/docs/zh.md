# 文档与图表理解

> 文档不是照片。PDF、科学论文、发票或手写表单有布局、表格、图表、脚注、页眉和纯图像理解无法捕获的语义结构。VLM 之前的堆栈是一个管道：Tesseract OCR + LayoutLMv3 + 表格提取启发式。VLM 浪潮用无 OCR 模型取代了它——Donut（2022）、Nougat（2023）、DocLLM（2023）直接发出结构化标记。到 2026 年前沿是"以 2576px 原生将页面图像喂给 Claude Opus 4.7"，结构化标记输出免费获得。本节解读文档 AI 的三个时代弧线。

**类型：** Build
**语言：** Python（标准库，布局感知文档解析器骨架）
**前置知识：** Phase 12 · 05（LLaVA），Phase 5（NLP）
**时间：** 约 180 分钟

## 学习目标

- 解释文档 AI 的三个时代：OCR 管道、OCR-free、VLM-native。
- 描述 LayoutLMv3 的三个输入流：文本、布局（bbox）、图像 patches，统一掩码。
- 比较 Donut（OCR-free，图像 → 标记）、Nougat（科学论文 → LaTeX）、DocLLM（布局感知生成）、PaliGemma 2（VLM-native）。
- 为新任务选择文档模型（发票、科学论文、手写表单、中文收据）。

## 问题背景

"理解这个 PDF" 表面简单实际困难。信息存在于：

- 文本内容（90% 的信号）。
- 布局（页眉、脚注、侧边栏、双栏格式）。
- 表格（行、列、合并单元格）。
- 图形和图表。
- 手写注释。
- 字体和排版（标题 vs 正文）。

原始 OCR 转储文本并丢失其余。需要发票的系统需要知道"总计：$1,245"来自右下角，而非脚注。

## 核心概念

### 时代 1——OCR 管道（2021 年之前）

经典堆栈：

1. PDF → 每页图像。
2. Tesseract（或商业 OCR）提取带逐词边界框的文本。
3. 布局分析器识别块（页眉、表格、段落）。
4. 表格结构识别器解析表格。
5. 域规则 + 正则表达式提取字段。

对干净打印文本有效。在手写、倾斜扫描、复杂表格、非英语脚本上失败。每个失败模式需要自定义异常路径。

### TrOCR（2021）

TrOCR（Li 等，arXiv:2109.10282）用合成 + 真实文本图像上训练的 transformer 编码器-解码器替换 Tesseract 的经典 CNN-CTC。手写和多语言文本上干净胜出。仍是管道（检测器然后 TrOCR 然后布局），但 OCR 步骤显著改进。

### 时代 2——OCR-free（2022-2023）

首个 OCR-free 模型说：完全跳过检测，直接将图像像素映射到结构化输出。

Donut（Kim 等，arXiv:2111.15664）：
- 编码器-解码器 transformer，编码器是 Swin-B。
- 输出是表单理解的 JSON、摘要的 markdown 或任何任务特定 schema。
- 无 OCR，无布局，无检测。

Nougat（Blecher 等，arXiv:2308.13418）：
- 专门在科学论文上训练。
- 输出 LaTeX / markdown。
- 处理公式、多栏布局、图形。
- 每个 arXiv 解析器调用的模型。

这些是专家，不是通才。Donut 在科学论文上失败；Nougat 在发票上失败。

### LayoutLMv3（2022）

不同路线。LayoutLMv3（Huang 等，arXiv:2204.08387）保留 OCR 但添加布局理解：

- 三个输入流：OCR 文本 token、逐 token 2D 边界框、图像 patches。
- 跨所有三种模态的掩码训练目标（掩码文本、掩码 patches、掩码布局）。
- 下游：分类、实体提取、表格 QA。

LayoutLMv3 是 OCR 基文档理解的巅峰。在表单和发票上强。需要上游 OCR。在标准化文档基准上精度最高。

### DocLLM（2023）

DocLLM（Wang 等，arXiv:2401.00908）是 LayoutLM 的生成式孪生。在布局 token 条件下生成自由形式答案。QA 更好；仍依赖 OCR 输入。

### 时代 3——VLM-native（2024+）

2024 年 VLM 变得足够好，可以完全取代管道。将完整页面图像以高分辨率喂给 VLM，问问题，得答案。

- LLaVA-NeXT 336-tile AnyRes 对小文档有效。
- Qwen2.5-VL 动态分辨率原生处理 2048+ 像素。
- Claude Opus 4.7 支持 2576px 文档。
- PaliGemma 2（2025 年 4 月）专门为文档 + 手写训练。

VLM-native 与 OCR-管道的差距快速缩小。到 2026 年，VLM-native 在以下方面胜出：

- 场景文本（手写 + 打印、混合脚本）。
- 带合并单元格的复杂表格。
- 嵌入文本中的数学公式。
- 带文本注释的图形。

OCR 管道仍在以下方面胜出：

- 大规模纯扫描工作负载，每页延迟很重要。
- 管道可靠性（确定性失败 vs VLM 幻觉）。
- 需要可审计 OCR 输出的监管环境。

### Claude 4.7 / GPT-5 前沿

在 2576 像素原生输入下，前沿 VLM 以接近人类精度做文档理解。2026 年初的基准数字：

- DocVQA：Claude 4.7 ~95.1，PaliGemma 2 ~88.4，Nougat ~77.3，管道 LayoutLMv3 ~83。
- ChartQA：Claude 4.7 ~92.2，GPT-4V ~78。
- VisualMRC：Claude 4.7 ~94。

封闭模型差距主要是分辨率和基础 LLM 规模。7B 开源模型差几分但在追赶。

### 数学公式和 LaTeX 输出

科学论文需要精确的 LaTeX 输出用于公式。Nougat 为此训练。用 LaTeX 目标训练的 VLM（Qwen2.5-VL-Math、Nougat 衍生）产生可用的 LaTeX。没有显式 LaTeX 训练，VLM 产生可读但不精确的转录。

2026 年科学论文流水线：链式 Nougat 处理 PDF，然后 VLM 处理棘手页面。

### 手写

仍是最难的子任务。混合打印 + 手写（医生笔记、填写表单）是 OCR 管道仍以成本优势击败 VLM 的地方。手写专用 VLM 正在改进（Claude 4.7、PaliGemma 2）。

### 2026 年配方

新文档 AI 项目：

- 纯打印发票规模化：LayoutLMv3 + 规则，成本效益高。
- 混合文档（科学 + 手写 + 表单）：VLM-native（PaliGemma 2 或 Qwen2.5-VL）。
- 完整 arXiv 摄取：Nougat 用于数学，VLM 用于图形。
- 监管：OCR 管道 + VLM 验证器用于交叉检查。

## 使用方法

`code/main.py`：

- 玩具布局感知分词器：给定（文本，bbox）对，产生 LayoutLMv3 风格输入。
- Donut 风格任务 schema 生成器：表单 JSON 模板。
- 跨 OCR 管道、Donut、Nougat 和 VLM-native 每页 token 预算比较。

## 输出作品

本节生成 `outputs/skill-document-ai-stack-picker.md`。给定文档 AI 项目（领域、规模、质量、监管），在 OCR 管道、OCR-free 专家和 VLM-native 之间选择。

## 练习

1. 你的项目每天 1000 万张发票。哪个堆栈在不损失准确率的情况下最小化每页成本？

2. 为什么 LayoutLMv3 在表单 QA 上优于纯 CLIP-VLM，但在场景文本上不如？bbox 流放弃了什么？

3. Nougat 生成 LaTeX。提出一个 VLM-native 输出在 LaTeX 保真度上击败 Nougat 的测试用例，以及 Nougat 赢的用例。

4. 阅读 PaliGemma 2 论文（Google，2024）。与 PaliGemma 1 相比，提升文档准确率的关键训练数据添加是什么？

5. 设计一个监管安全的混合体：OCR 管道作为主要，VLM 作为次要交叉检查。如何解决分歧？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| OCR pipeline | "Tesseract 风格" | 分阶段堆栈：检测 -> OCR -> 布局 -> 规则；确定性，脆弱 |
| OCR-free | "Donut 风格" | 跳过显式 OCR 的图像到输出 transformer；单模型 |
| Layout-aware | "LayoutLM" | 输入包括逐 token bbox 坐标；跨模态统一掩码 |
| VLM-native | "前沿 VLM" | 以高分辨率直接向 Claude/GPT/Qwen VLM 喂页面图像；无管道 |
| DocVQA | "文档基准" | 文档 VQA 标准；引用最多的分数 |
| Markup output | "LaTeX / MD" | 结构化输出格式而非自由格式文本；支持下游自动化 |

## 延伸阅读

- [Li 等 — TrOCR (arXiv:2109.10282)](https://arxiv.org/abs/2109.10282)
- [Blecher 等 — Nougat (arXiv:2308.13418)](https://arxiv.org/abs/2308.13418)
- [Huang 等 — LayoutLMv3 (arXiv:2204.08387)](https://arxiv.org/abs/2204.08387)
- [Kim 等 — Donut (arXiv:2111.15664)](https://arxiv.org/abs/2111.15664)
- [Wang 等 — DocLLM (arXiv:2401.00908)](https://arxiv.org/abs/2401.00908)