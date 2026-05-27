# 音频-语言模型：从 Whisper 到 Audio Flamingo 3 的弧线

> Whisper（Radford 等，2022 年 12 月）解决了语音识别——68 万小时弱监督多语言语音，简单的编码器-解码器 transformer，一个基准，使每个后续 ASR 发布都引用它。但识别不是推理。问"录音里有什么乐器"或"说话人表达什么情绪"或"第 3 分钟发生了什么"需要音频理解，不是转录。Qwen-Audio、SALMONN、LTU 和 NVIDIA 的 Audio Flamingo 3（AF3，2025 年 7 月）逐步构建了这个堆栈：保留 Whisper 类编码器，接上 Q-former，在音频文本指令数据上训练，添加思维链推理。本节走读这条弧线。

**类型：** Build
**语言：** Python（标准库，对数梅尔频谱图 + 音频 Q-former 骨架）
**前置知识：** Phase 6（Speech and Audio），Phase 12 · 03（Q-Former）
**时间：** 约 180 分钟

## 学习目标

- 从波形计算对数梅尔频谱图：窗口化、FFT、滤波器组、对数变换。
- 比较编码器选项：Whisper 编码器、BEATs、AF-Whisper 混合。何时每个胜出。
- 构建音频 Q-former：N 个可学习 queries 交叉关注频谱图 patches。
- 解释级联（Whisper-then-LLM）vs 端到端音频-LLM 训练：为什么端到端对推理规模化更好。

## 问题背景

语音识别被 Whisper 解决了。OCR-音频是商品。但"商品"在转录处停止。如果模型不能对其听到的内容推理——时间、说话人、情绪、音乐结构、环境声音——仅转录不能驱动产品功能。

三条明显路线：

1. 级联：Whisper 转录，LLM 推理转录。对纯语音场景有效。对音乐、环境音频、多说话人重叠、情绪无效。

2. 端到端音频-LLM：音频编码器直接将音频 token 送入 LLM，跳过转录。保留声学信息（情绪、说话人、环境）。需要新训练数据。

3. 混合：音频编码器 + 文本解码器，两者都能转录和推理。Qwen-Audio 和 Audio Flamingo 选这条路。

## 核心概念

### 对数梅尔频谱图：输入特征

每个音频编码器以相同特征开始：对数梅尔频谱图。

1. 重采样到 16 kHz。
2. 短时傅里叶变换，25ms 窗口，10ms 跳步。
3. 取 FFT 结果的幅度。
4. 应用梅尔滤波器组（通常 80 个滤波器在对数空间 0-8000 Hz）到感知频率。
5. 对数压缩（log(1 + x)）动态范围。

结果：形状 (T, 80) 的 2D 数组，其中 T 是时间帧数。对于 30 秒片段在 100 Hz 帧率：(3000, 80)。

### Whisper 的编码器

Whisper 的编码器是一个 12 层 ViT 风格 transformer，处理对数梅尔频谱图作为时间帧序列。输出：每时间帧一个隐藏状态向量。

对于 ASR，Whisper 的解码器是一个交叉注意力 transformer，在编码器输出上生成文本 token。标准编码器-解码器。

对于 ALM（音频-LLM），你需要编码器输出作为不同 LLM 的输入。模式：Whisper 编码器冻结，Q-former 可训练，LLM 冻结或调优。

### BEATs 和音频特定编码器

Whisper 在语音主导数据上训练。它对音乐和环境音频较弱。

BEATs（Chen 等，2022）是自监督 transformer，在 AudioSet 上训练。在相同参数计数下比 Whisper 更好地捕获音乐和环境声音。

AF-Whisper（Audio Flamingo 3 的混合）：拼接 Whisper + BEATs 特征作为音频输入。Whisper 携带语言信号，BEATs 携带声学信号。

### 音频 Q-former

与 BLIP-2 的视觉 Q-former 相同模式。固定数量的可学习 queries（通常 32 或 64）交叉关注音频编码器的输出帧。Queries 成为 LLM 消费的音频 token。

训练对齐阶段：Q-former 单独，在音频文本对（AudioCaps、Clotho）上用对比 + 标题损失训练。指令阶段：端到端，解冻 LLM，在指令数据上训练。

### 弧线——SALMONN、Qwen-Audio、AF3

SALMONN（Tang 等，2023）：Whisper + BEATs + Q-former + LLaMA。首个具有严肃推理能力的开源音频-LLM。在 MMAU 上基准约 0.55 综合分。

Qwen-Audio（Chu 等，2023）：类似架构，在更丰富的数据集上训练，为多轮对话调优。MMAU 约 0.60。

LTU——Listen, Think, Understand（Gong 等，2023）：显式推理数据，专注于音频片段上的思维链。更小但更专注。

Audio Flamingo 3（Goel 等，2025 年 7 月）：当前开源 SOTA。80 亿 LLM 骨干（Qwen2 7B），Whisper-large 编码器拼接 BEATs，64-query Q-former，在 100 万+ 音频文本指令对上训练。MMAU 0.72，在一些子任务上匹配专有前沿。

AF3 还引入了按需音频思维链：模型可以选择在最终答案前发出思考 token（"让我先识别乐器：..."）。启用思考时，复杂推理任务准确率提升 3-5 分。

### 级联 vs 端到端

级联流水线：
1. Whisper 转录音频 → 文本。
2. LLM 推理文本。

对"总结这个播客"完美有效。以下情况失败：
- "这首歌的情绪是什么？"——情绪在声音里，不在文字里。
- "说话的是 Alice 还是 Bob？"——需要说话人识别。
- "爆炸发生在第几秒？"——时间定位在文本中丢失。
- "这是真实的还是生成的音频？"——深度伪造检测需要声学特征。

端到端保留声学信号。Qwen-Audio 和 AF3 原生处理音乐、环境和情绪。

### 2026 年生产配方

新音频理解产品：

- 如果需要：转录是目标，无音乐，无情绪推理——选级联。
- 如果需要：音乐、情绪、多说话人或复杂音频推理——选 AF3 / Qwen-Audio 家族。

级联更便宜更简单。端到端更有能力。

### MMAU——音频推理基准

MMAU（大规模多模态音频理解）是 2024-2025 年音频推理基准：

- 跨语音、音乐、环境声音的 10,000 个音频文本 QA 对。
- 涵盖分类、时间推理、因果推理、开放式 QA。
- 测试级联流水线系统性地遗漏的内容。

开源 SOTA（AF3）0.72；专有前沿约 0.78（Gemini 2.5 Pro、Claude Opus 4.7）。差距小于 VideoMME 的开源-封闭 delta，表明音频-LLM 日益成熟。

## 使用方法

`code/main.py`：

- 在标准库中实现对数梅尔频谱图计算：窗口化、朴素 DFT、梅尔滤波器组。
- 音频 Q-former 骨架：给定编码器输出帧，计算 Q、K、V、注意力和发出 N 个 token。
- 玩具任务上的级联 vs 端到端比较。

## 输出作品

本节生成 `outputs/skill-audio-llm-pipeline-picker.md`。给定音频任务（转录、音乐标记、情绪推理、多说话人 diarization、环境分类），选择级联、端到端 AF3 或混合。

## 练习

1. 计算 30 秒片段在 16kHz、25ms 窗口、10ms 跳步、80 个梅尔 bin 下的对数梅尔频谱图维度。在 48kHz 下如何变化？

2. 为什么 Whisper 在音乐上表现不佳？BEATs 捕获了 Whisper 没有捕获的什么音频特征？

3. 64 queries vs 32 queries 的音频 Q-former：在什么任务复杂度下 64 值？32 为什么节省计算？

4. 阅读 AF3 第 4 节关于按需思考。提出三个思维链帮助最大的音频任务。

5. 使用 AF3 的输出实现最小 diarization 流水线。如何标记说话人变化？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| Log-Mel spectrogram | "梅尔特征" | 经梅尔滤波器组后的 (时间, 频率) 2D 对数幅度值数组 |
| Audio Q-former | "音频 Perceiver" | 从音频编码器输出到固定长度 queries 的交叉注意力瓶颈，送入 LLM |
| Cascaded | "ASR-then-LLM" | Whisper 转录然后文本 LLM 推理的流水线；丢失声学信息 |
| End-to-end | "音频-LLM" | 音频特征通过 Q-former 直接进入 LLM；保留声学信号 |
| BEATs | "AudioSet 音频编码器" | 在 AudioSet 上训练的自监督 transformer；在音乐 + 环境声音上强 |
| MMAU | "音频推理基准" | 跨语音、音乐、环境的 10k QA 对；2024 年评估标准 |
| On-demand thinking | "音频 CoT" | 模型可以选择在最终答案前发出推理 token，准确率提升 3-5 分 |

## 延伸阅读

- [Radford 等 — Whisper (arXiv:2212.04356)](https://arxiv.org/abs/2212.04356)
- [Chu 等 — Qwen-Audio (arXiv:2311.07919)](https://arxiv.org/abs/2311.07919)
- [Goel 等 — Audio Flamingo 3 (arXiv:2507.08128)](https://arxiv.org/abs/2507.08128)
- [Tang 等 — SALMONN (arXiv:2310.13289)](https://arxiv.org/abs/2310.13289)
- [Gong 等 — LTU (arXiv:2305.10790)](https://arxiv.org/abs/2305.10790)