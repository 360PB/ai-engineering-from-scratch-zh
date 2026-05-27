# Whisper 架构

> 音频是一幅频率随时间变化的图像。Whisper 是一个吃梅尔频谱图并说话的 ViT。

**类型：** 学习
**语言：** Python
**前置知识：** Phase 7 · 05（完整 Transformer）、Phase 7 · 08（编码器-解码器）、Phase 7 · 09（ViT）
**时间：** 约 45 分钟

## 问题

Whisper 之前，自动语音识别（ASR）的 SOTA 意味着 wav2vec 2.0 和 HuBERT——自监督特征提取器 + 微调头。高质量，昂贵的数据管道，领域脆弱。多语言语音识别需要每个语系单独的模型。

Whisper 做了三个赌注：

1. **训练一切。** 680,000 小时的弱标签音频，来自互联网 97 种语言的爬取。无干净学术语料。无音素标签。
2. **多任务单一模型。** 一个解码器联合训练转录、翻译、语音活动检测、语言 ID 和时间戳——通过任务 token。
3. **标准编码器-解码器 Transformer。** 编码器消费 log-mel 频谱图。解码器自回归生成文本 token。无声码器，无 CTC，无 HMM。

结果：Whisper large-v3 对口音、噪声和零干净标签语言都具有鲁棒性。它是 2026 年每个开源语音助手和大多数商业语音助手的事实标准前端。

## 概念

![Whisper 管道：音频 → mel → 编码器 → 解码器 → 文本](../assets/whisper.svg)

### 步骤 1 — 重采样 + 窗口

16 kHz 音频。裁剪/填充到 30 秒。计算 log-mel 频谱图：80 个梅尔容器，10 ms 步长 → ~3,000 帧 × 80 特征。这就是 Whisper 看到的"输入图像"。

### 步骤 2 — 卷积干

两个 Conv1D 层，kernel 3，stride 2，将 3,000 帧减少到 1,500。减半序列长度而不添加大量参数。

### 步骤 3 — 编码器

24 层（large）Transformer 编码器 over 1,500 时间步。正弦位置编码、自注意力、GELU FFN。产生 1,500 × 1,280 隐藏状态。

### 步骤 4 — 解码器

24 层 Transformer 解码器。它自回归生成来自 BPE 词汇的 token，该词汇是 GPT-2 的超集，带一些音频专用特殊 token。

### 步骤 5 — 任务 token

解码器 prompt 以控制 token 开始，告诉模型要做什么：

```
<|startoftranscript|>  <|en|>  <|transcribe|>  <|0.00|>
```

或

```
<|startoftranscript|>  <|fr|>  <|translate|>   <|0.00|>
```

模型按此约定训练。你通过前缀控制任务。这是 2026 年指令微调应用于语音的等效，但应用于语音。

### 步骤 6 — 输出

束搜索（宽度 5）带对数概率阈值。当 `<|notimestamps|>` token 不存在时，每 0.02 秒音频预测一次时间戳。

### Whisper 规格

| 模型 | 参数 | 层数 | d_model | 头数 | VRAM (fp16) |
|-------|--------|--------|---------|-------|-------------|
| Tiny | 39M | 4 | 384 | 6 | ~1 GB |
| Base | 74M | 6 | 512 | 8 | ~1 GB |
| Small | 244M | 12 | 768 | 12 | ~2 GB |
| Medium | 769M | 24 | 1024 | 16 | ~5 GB |
| Large | 1550M | 32 | 1280 | 20 | ~10 GB |
| Large-v3 | 1550M | 32 | 1280 | 20 | ~10 GB |
| Large-v3-turbo | 809M | 32 | 1280 | 20 | ~6 GB（4 层解码器） |

Large-v3-turbo（2024）将解码器从 32 层减少到 4。8 倍更快的解码，<1 WER 点退化。这个解码速度提升是 Whisper-turbo 成为 2026 年实时语音 agent 默认的原因。

### Whisper 不做什么

- 无日志（谁在说话）。配合 pyannote 使用。
- 本地无实时流式——30 秒窗口是固定的。现代包装器（`faster-whisper`、`WhisperX`）通过 VAD + 重叠bolt上流式。
- 超过 30 秒无外部分块的长格式。没有外部分块。实践中效果很好，因为人类语音转录很少需要超过 30 秒的长程上下文。

### 2026 年格局

| 任务 | 模型 | 备注 |
|------|-------|------|
| 英语 ASR | Whisper-turbo、Moonshine | Moonshine 在边缘上快 4 倍 |
| 多语言 ASR | Whisper-large-v3 | 97 种语言 |
| 流式 ASR | faster-whisper + VAD | 150 ms 延迟目标可实现 |
| TTS | Piper、XTTS-v2、Kokoro | 编码器-解码器模式，但 Whisper 形状 |
| 音频 + 语言 | AudioLM、SeamlessM4T | 单个 transformer 中的文本 token + 音频 token |

## 构建

参见 `code/main.py`。我们不训练 Whisper——我们构建 log-mel 频谱图管道 + 任务 token prompt 格式化器。这些是你在生产中实际接触的部分。

### 步骤 1：合成音频

生成 440 Hz 正弦波在 16 kHz 采样 1 秒。16,000 采样。

### 步骤 2：log-mel 频谱图（简化）

完整的 mel 频谱图需要 FFT。我们做一个简化的分帧 + 每帧能量版本，展示管道而不需要 `librosa`：

```python
def frame_signal(x, frame_size=400, hop=160):
    frames = []
    for start in range(0, len(x) - frame_size + 1, hop):
        frames.append(x[start:start + frame_size])
    return frames
```

帧 = 25 ms，hop = 10 ms。匹配 Whisper 的窗口化。每帧能量代替梅尔容器用于教学。

### 步骤 3：填充到 30 秒

Whisper 始终处理 30 秒块。填充（或裁剪）频谱图到 3,000 帧。

### 步骤 4：构建 prompt token

```python
def whisper_prompt(lang="en", task="transcribe", timestamps=True):
    tokens = ["<|startoftranscript|>", f"<|{lang}|>", f"<|{task}|>"]
    if not timestamps:
        tokens.append("<|notimestamps|>")
    return tokens
```

这就是整个任务控制界面。一个 4 token 前缀。

## 使用

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe("meeting.wav", language="en", task="transcribe")
print(result["text"])
print(result["segments"][0]["start"], result["segments"][0]["end"])
```

更快，OpenAI 兼容：

```python
from faster_whisper import WhisperModel
model = WhisperModel("large-v3-turbo", compute_type="int8_float16")
segments, info = model.transcribe("meeting.wav", vad_filter=True)
for s in segments:
    print(f"{s.start:.2f} - {s.end:.2f}: {s.text}")
```

**2026 年何时选择 Whisper：**

- 带一个模型的多语言 ASR。
- 对有噪声、多样音频的鲁棒转录。
- 研究/原型 ASR——最快的起点。

**何时选择其他：**

- 边缘超低延迟流式——Moonshine 在匹配质量下快 4 倍。
- 需要 <200 ms 的实时对话 AI——专用流式 ASR。
- 说话人日志——Whisper 不做；bolt on pyannote。

## 发货

参见 `outputs/skill-asr-configurator.md`。为新的语音应用选择 ASR 模型、解码参数和预处理管道。

## 练习

1. **简单。** 运行 `code/main.py`。确认 1 秒信号 @ 16 kHz，10 ms hop 的帧数约 100 帧。30 秒：约 3,000 帧。
2. **中等。** 使用 `numpy.fft` 构建完整的 log-mel 频谱图。验证 80 个梅尔容器与 `librosa.feature.melspectrogram(n_mels=80)` 在数值误差内一致。
3. **困难。** 实现流式推理：将音频分块为 10 秒窗口，2 秒重叠，对每个块运行 Whisper，合并转录。在 5 分钟播客样本上测量词错误率 vs 单次通过。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------|--------|
| 梅尔频谱图 | "音频图像" | 2D 表示：频率容器在一轴，时间帧在另一轴；每格对数缩放能量。 |
| Log-mel | "Whisper 看到的内容" | 通过 log 的梅尔频谱图；近似人类响度感知。 |
| 帧 | "一个时间片" | 25 ms 采样窗口；以 10 ms 步长重叠。 |
| 任务 token | "语音的 prompt 前缀" | 解码器 prompt 中如 `<\|transcribe\|>` / `<\|translate\|>` 的特殊 token。 |
| 语音活动检测（VAD） | "找语音" | 门控在 ASR 前去除静音；大幅削减成本。 |
| CTC | "连接时序分类" | 经典 ASR 损失用于无对齐训练；Whisper 不使用。 |
| Whisper-turbo | "小解码器，完整编码器" | large-v3 编码器 + 4 层解码器；8 倍更快的解码。 |
| Faster-whisper | "生产包装器" | CTranslate2 重实现；int8 量化；比 OpenAI 参考快 4 倍。 |

## 延伸阅读

- [Radford et al. (2022). Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — Whisper 论文。
- [OpenAI Whisper repo](https://github.com/openai/whisper) — 参考代码 + 模型权重。读 `whisper/model.py` 从上到下看 Conv1D 干 + 编码器 + 解码器，约 400 行。
- [OpenAI Whisper — `whisper/decoding.py`](https://github.com/openai/whisper/blob/main/whisper/decoding.py) — 束搜索 + 任务 token 逻辑在步骤 5-6 中描述；500 行，完全可读。
- [Baevski et al. (2020). wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477) — 前体；在某些设置下仍是 SOTA 特征。
- [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) — 生产包装器，比参考快 4 倍。
- [Jia et al. (2024). Moonshine: Speech Recognition for Live Transcription and Voice Commands](https://arxiv.org/abs/2410.15608) — 2024 年边缘友好 ASR，Whisper 形状但更小。
- [HuggingFace blog — "Fine-Tune Whisper For Multilingual ASR with 🤗 Transformers"](https://huggingface.co/blog/fine-tune-whisper) — 包括 mel 频谱图预处理器和 token 时间戳处理的规范微调配方。
- [HuggingFace `modeling_whisper.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/modeling_whisper.py) — 完整实现（编码器、解码器、交叉注意力、生成），镜像课程的架构图。