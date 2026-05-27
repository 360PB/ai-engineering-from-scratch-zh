# 水印——SynthID、Stable Signature、C2PA

> 三种技术结构化 2026 年 AI 生成内容来源。SynthID（Google DeepMind）——2023 年 8 月推出图像水印，2024 年 5 月（Gemin + Veo）推出文本+视频，2024 年 10 月通过 Responsible GenAI Toolkit 开源文本，2025 年 11 月与 Gemini 3 Pro 一起推出统一多媒体检测器。文本水印以不可感知的方式调整下一个 token 采样概率；图像/视频水印在压缩、裁剪、滤镜、帧率变化下存活。Stable Signature（Fernandez 等，ICCV 2023，arXiv:2303.15435）——微调潜在扩散解码器使每个输出包含固定消息；裁剪（10% 内容）生成图像在 FPR<1e-6 下检测率 >90%。后续"Stable Signature is Unstable"（arXiv:2405.07145，2024 年 5 月）——微调移除水印同时保持质量。C2PA——加密签名、防篡改元数据标准（C2PA 2.2 Explainer 2025）。水印和 C2PA 是互补的：元数据可以剥离但携带更丰富的来源；水印在转码后持续但携带更少信息。

**类型：** 构建
**语言：** Python（标准库，token 水印嵌入 + 检测）
**前置知识：** Phase 10 · 04（采样），Phase 01 · 09（信息论）
**时长：** 约 75 分钟

## 学习目标

- 描述 token 级水印（SynthID-text 风格）及其可检测的机制。
- 描述 Stable Signature 以及 2024 年移除攻击如何破解它。
- 陈述 C2PA 的作用以及为什么它与水印互补。
- 描述关键限制：模型特定信号、释义稳健性和保持意义攻击（arXiv:2508.20228）。

## 问题

2023-2024 年深度伪造和 AI 生成内容以规模进入政治和消费者环境。水印是提议的技术来源信号：在创建时标记生成物，稍后检测。2025 年证据：没有水印是无条件鲁棒的，但与 C2PA 元数据分层，组合提供可用的来源故事。

## 概念

### 文本水印（SynthID-text 风格）

Kirchenbauer 等 2023 机制，由 Google 生产化：

1. 在每个解码步骤，将前一个 K 个 token 的哈希映射为词汇表的伪随机分区为"绿色"和"红色"集合。
2. 通过向绿色 logit 添加 δ 来偏置采样朝向绿色集合。
3. 生成物包含比偶然更多的绿色 token。

检测：重新哈希每个前缀，计算生成中绿色 token 数，计算 z-score。z-score 对有水印文本 >0，对人类文本 ~0。

属性：
- 对读者不可感知（δ 足够小，质量损失很小）。
- 可用词汇表分区函数检测。
- 对释义不鲁棒——重写文本破坏信号。

SynthID-text 于 2024 年 10 月通过 Google Responsible GenAI Toolkit 开源。

### Stable Signature（图像）

Fernandez 等 ICCV 2023。微调潜在扩散解码器使每个生成图像包含固定二进制消息嵌入在潜在表示中。检测从潜在解码。裁剪（10% 内容）生成图像在 FPR<1e-6 下检测率 >90%。

2024 年 5 月"Stable Signature is Unstable"（arXiv:2405.07145）：对解码器进行微调可以移除水印同时保持图像质量。对抗后生成微调便宜；水印的对抗鲁棒性有限。

### SynthID 统一检测器（2025 年 11 月）

与 Gemini 3 Pro 一起：多媒体检测器，从文本、图像、音频和视频中读取 SynthID 信号在一个 API 中。统一 Google 来源栈。

### C2PA

Content Provenance and Authenticity 联盟。加密签名防篡改元数据标准。C2PA 2.2 Explainer（2025）。C2PA manifest 记录来源声明（谁创建、何时、什么转换），由创建者的密钥签名。

与水印互补：
- 元数据可以剥离；水印不能（容易）。
- 元数据丰富（完整来源链）；水印携带比特。
- C2PA 依赖平台 adoption；水印自动嵌入。

Google 在 Search、Ads 和"About this image"中集成两者。

### 限制

- **模型特定。** SynthID 水印来自 SynthID 启用模型的生成。没有 SynthID 的模型生成不受水印保护，因此"无 SynthID 信号"不是真实性的证明。
- **释义。** 文本水印不保持释义生存。
- **转换攻击。** arXiv:2508.20228（2025）显示保持意义攻击破坏文本水印和许多图像水印。
- **微调移除。** 按"Stable Signature is Unstable"，后生成微调移除嵌入水印。

### EU AI Act 第 50 条

AI 生成内容标签的透明度代码（第一稿 2025 年 12 月，第二稿 2026 年 3 月，预计 2026 年 6 月最终版 per European Commission status page）。需要技术层的监管层。深度伪造必须被标记。

### 为什么这在 Phase 18 中重要

第 22-23 课关于模型发出的内容（私人数据、来源信号）。第 27 课涵盖训练数据治理。第 24 课是需要这些技术措施的监管框架。

## 使用它

`code/main.py` 构建玩具文本水印。Token 是 0..N-1 的整数；水印采样偏向哈希定义的绿色集合。检测器计算绿色 token z-score。你可以在 1000 token 生成上观察检测，观看释义破坏信号，并测量人类文本上的假阳性率。

## 交付它

本课生成 `outputs/skill-provenance-audit.md`。给定带来源声明的内容部署，审计：水印机制（如果有）、C2PA 签名链（如果有）、每个的对抗鲁棒性，以及每个模态的覆盖。

## 练习

1. 运行 `code/main.py`。报告水印 1000 token 生成 vs 人类创作文本的 z-score。在 95% 置信阈值识别假阳性率。

2. 实现替换 30% token 为同义词的释义攻击。重新测量 z-score。

3. 读 Kirchenbauer 等 2023 第 6 节关于鲁棒性。为什么文本水印在释义下失败但图像水印在裁剪下存活？

4. 设计使用 SynthID-text + C2PA 元数据的部署。描述消费者看到的来源链。识别每个组件的一种失败模式。

5. 2024 年"Stable Signature is Unstable"结果显示微调移除图像水印。设计限制此攻击的部署控制——例如，要求签名发布微调检查点。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| SynthID | Google 的水印 | 跨模态来源信号；文本、图像、音频、视频 |
| Token 水印 | Kirchenbauer 风格 | 通过绿色 token z-score 可检测的偏置采样文本水印 |
| Stable Signature | 图像水印 | 微调解码器水印；ICCV 2023 |
| C2PA | 元数据标准 | 加密签名防篡改来源元数据 |
| 释义稳健性 | 释义会破坏它吗 | 文本水印属性；目前有限 |
| 微调移除 | 对抗去水印 | 通过解码器微调移除图像水印的攻击 |
| 跨模态检测器 | 统一 SynthID | 2025 年 11 月跨模态 API |

## 延伸阅读

- [Kirchenbauer 等 — A Watermark for Large Language Models (ICML 2023, arXiv:2301.10226)](https://arxiv.org/abs/2301.10226) — token 水印机制
- [Fernandez 等 — Stable Signature (ICCV 2023, arXiv:2303.15435)](https://arxiv.org/abs/2303.15435) — 图像水印论文
- ["Stable Signature is Unstable" (arXiv:2405.07145)](https://arxiv.org/abs/2405.07145) — 移除攻击
- [Google DeepMind — SynthID](https://deepmind.google/models/synthid/) — 跨模态水印
- [C2PA 2.2 Explainer (2025)](https://c2pa.org/specifications/specifications/2.2/explainer/Explainer.html) — 元数据标准