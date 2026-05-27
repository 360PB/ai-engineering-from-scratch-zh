# 全模态模型：Qwen2.5-Omni 与 Thinker-Talker 分裂

> GPT-4o 在 2024 年 5 月的产品演示具有破坏性，不是因为底层模型而是因为产品形态——一个语音界面，你说，模型看到摄像头看到的内容，然后以低于 250ms 的延迟回话。开源生态在 2024 年和 2025 年其余时间竞相达到该产品表面。Qwen2.5-Omni（2025 年 3 月）是参考开源设计：一个 Thinker（大型文本生成 transformer）加一个 Talker（并行语音生成 transformer），通过流式语音 token 连接。Mini-Omni 简化了它，Moshi 匹配其延迟，GLM-4-Voice 扩展到中文。本节解读 Thinker-Talker 架构和使流式实时对话工作的延迟预算。

**类型：** Build
**语言：** Python（标准库，流式流水线延迟模拟器 + VAD 循环）
**前置知识：** Phase 12 · 19（音频-LLM），Phase 12 · 16（任意到任意）
**时间：** 约 180 分钟

## 学习目标

- 将推理流水线分裂为 Thinker（文本推理）和 Talker（语音合成）并解释为什么并行流式工作。
- 逐组件计算对话交互的首音频字节时间（TTFAB）预算。
- 描述 TMRoPE 在 Thinker 内跨视觉、音频和文本的时间对齐位置编码。
- 说出三种实时对话模式：半双工、轮次接续、全双工。

## 问题背景

实时语音助手需要做很多，快速：

1. 听到用户。实时语音分词，语音活动检测（VAD）知道他们何时说完话。
2. 可选看到。摄像头输入在 2-4 FPS，通过流式送入 Thinker 以及音频。
3. 思考。根据对话历史组合响应。
4. 说话。合成音频 token，解码到波形，流式到用户扬声器。

每步增加延迟。对话感需要总往返 < 500ms——低于此，用户停止注意延迟。GPT-4o 声称约 250ms。Moshi 约 160ms。Qwen2.5-Omni 约 350-500ms。

每个组件都需要流式。没有什么可以是"批量处理然后解码"。

## 核心概念

### Thinker 和 Talker

Qwen2.5-Omni 的分解：

- Thinker：7B-80B 文本生成 transformer。消费交错的文本 + 图像 + 音频 token。输出表示要说什么的文本 token。
- Talker：较小的语音生成 transformer（200M-1B）。消费 Thinker 的文本输出 token 加最近的语音上下文 token。输出离散语音 token（残差-VQ 索引）。
- 语音解码器：流式波形解码器（SNAC、MoVQGAN 族），将语音 token 实时转换为音频样本。

分离很重要。Thinker 必须大以实现良好推理。Talker 可以小，因为它的职责是本地的——将文本转换为语音 token。更大的 Talker 不是更有表现力；而是更慢。

并行运行两者：

1. Thinker 发出文本 token t_i。
2. Talker 消费 t_i（通过流式）并发出语音 token s_i, s_{i+1}, ..., s_{i+k}。
3. 语音解码器消费到来的语音 token 并发出音频样本。
4. 当 Thinker 在文本 token t_{i+3} 时，Talker 已经为 t_0..t_{i+2} 流式音频。

### TMRoPE——时间对齐多模态位置

Thinker 需要整合图像帧（如以 4 FPS 到达）、音频帧（以 50 帧/秒到达）和对话历史中的文本。朴素序列顺序（所有图像，然后所有音频，然后文本）丢失时间对齐。

TMRoPE 为每个 token 分配绝对时间戳。视觉 token 在 t=2.3s。音频 token 在 t=2.32s。用户"停"的文本 token 在 t=2.35s。RoPE 按时间戳旋转注意力；模型将它们视为时间上并发。

这是"他在说 hello 的同时挥手"工作的基础设施——模型在同一概念时刻看到视频帧和音频。

### 流式语音合成

语音 token 必须流式。Mini-Omni（Xie & Wu，2024）引入"语言模型可以听，边思考边说话"：Thinker 输出 token 和 Talker 输出 token 在同一序列中交错。Talker 在 Thinker 提交下一个文本 token 时立即触发。无批量边界。

Moshi（Défossez 等，2024 年 10 月）是最快的开源实现。在单块 A100 上 160ms TTFAB。架构：一个单独的 7B transformer，在交替位置发出文本和语音 token，带将思考流与说话流分离的"内心独白"。这实际上是 Thinker + Talker 融合为一个模型，经过精心训练。

### VAD 和轮次接续

语音活动检测在输入侧运行。两种模式：

- 半双工：用户说话，模型听。模型说话，用户听。通过 VAD 静音检测清晰交接（约 200ms）。
- 全双工：两者可以同时说话。模型可以插话（"嗯哼"）或打断。难得多。Moshi 支持这个。

Qwen2.5-Omni 默认支持半双工，通过静音阈值进行轮次接续。全双工需要应用层处理。

### Qwen3-Omni（2025 年 11 月）

后继者。Qwen3-80B Thinker，更大 Talker，改进的 TMRoPE-v2。延迟接近 GPT-4o 的 250ms。开源权重。在 OmniBench 上与 Gemini 2.0 Live 竞争。

### 生产延迟预算

典型流式交互：

- 麦克风 → 音频 token：40-80ms。
- Prefill（prompt + 历史）：100-200ms 在 7B 上，70B 上更多。
- 第一个 Thinker 文本 token：40ms。
- Talker 处理第一个文本 token：20ms。
- 第一个语音 token 提交：40ms。
- 残差-VQ 解码：30ms。
- 语音波形解码：50-80ms。

总 TTFAB：7B 时 320-510ms，70B 时 600-900ms。前沿质量通常意味着 70B+；因此前沿延迟差距。

### Token 率数学

在 16kHz 语音下每 50 Hz 基础语音 token 50 token/秒，Talker 必须发出 ≥50 tok/s 才能跟上。在 H100 上典型 LLM 吞吐量 30-80 tok/s，较小（200-300M）Talker 够快；7B Talker 会跟不上。

这就是为什么存在小型专用 Talker 模型而不是"就用主模型"。

## 使用方法

`code/main.py`：

- 用模拟 token 发射率模拟 Thinker-Talker 流水线。
- 计算可配置模型大小和麦克风采样率的 TTFAB。
- 演示带 VAD 静音阈值的半双工轮次接续。

## 输出作品

本节生成 `outputs/skill-omni-streaming-budget.md`。给定实时语音产品的目标 TTFAB 和功能集（视觉输入、双语、全双工），选择 Qwen2.5-Omni、Qwen3-Omni、Moshi 或 Mini-Omni 并确定 Thinker/Talker 规模。

## 练习

1. 你的目标 TTFAB 是 300ms。在 7B Thinker 和 300M Talker 上，写出每个组件的延迟。

2. Qwen2.5-Omni 使用 TMRoPE。描述用户开始说话在 t=1s 且摄像头在 t=1.2s 捕捉到手势的 prompt 中模型看到的内容。

3. 全双工支持需要模型在听时发出音频。提出一种教会这个的训练数据格式。

4. 阅读 Moshi 论文第 4 节。描述"内心独白"分离及为什么它避免了 Thinker-Talker 分裂。

5. 计算吞吐量预算：Talker 必须以多快发出 token 才能跟上 16kHz 下每基础层 50 token/秒的语音？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| Thinker | "推理大脑" | 大型文本生成 transformer，产生要说什么 |
| Talker | "语音生成嘴巴" | 小型 transformer，从 Thinker 的文本产生离散语音 token |
| TTFAB | "延迟预算" | 首音频字节时间：用户语音结束到第一个音频样本输出 |
| TMRoPE | "时间对齐 RoPE" | 跨视觉、音频、文本使用绝对时间戳的位置编码 |
| Half-duplex | "轮次接续" | 用户和模型交替；VAD 静音检测用户完成 |
| Full-duplex | "同时" | 模型可以同时说话和听；可以插话 |
| Inner monologue | "Moshi 分离" | 单模型设计，其中思考流和说话流交错 |

## 延伸阅读

- [Xu 等 — Qwen2.5-Omni (arXiv:2503.20215)](https://arxiv.org/abs/2503.20215)
- [Qwen Team — Qwen3-Omni (arXiv:2509.17765)](https://arxiv.org/html/2509.17765v1)
- [Xie & Wu — Mini-Omni (arXiv:2408.16725)](https://arxiv.org/abs/2408.16725)
- [Défossez 等 — Moshi (arXiv:2410.00037)](https://arxiv.org/abs/2410.00037)
- [Zeng 等 — GLM-4-Voice (arXiv:2412.02612)](https://arxiv.org/abs/2412.02612)