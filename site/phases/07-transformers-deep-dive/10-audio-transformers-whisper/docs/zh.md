# 音频 Transformer — Whisper 架构

> 音频是时间上的频率图像。Whisper 是一个吞噬梅尔频谱图并回话的 ViT。

**类型：** 学习
**语言：** Python
**前置知识：** Phase 7 第 5 课（完整 Transformer）、Phase 7 第 8 课（编码器-解码器）、Phase 7 第 9 课（ViT）
**时长：** ~45 分钟

## 问题

Whisper（OpenAI，Radford et al. 2022）之前，自动语音识别（ASR）的最先进是 wav2vec 2.0 和 HuBERT——自监督特征提取器加上微调 head。质量高，数据管道昂贵，领域脆弱。多语言语音识别需要每个语系单独的模型。

Whisper 下了三个赌注：

1. **在所有数据上训练。** 680,000 小时从互联网抓取的弱标注音频，跨越 97 种语言。没有干净的学术语料。没有音素标签。
2. **多任务单模型。** 一个解码器联合训练转录、翻译、语音活动检测、语言 ID 和时间戳，任务 token。
3. **标准编码器-解码器 transformer。** 编码器消耗对数梅尔频谱图。解码器自回归产生文本 token。无声码器，无 CTC，无 HMM。

结果：Whisper large-v3 在口音、噪声和零干净标注数据的语言上具有鲁棒性。它是 2026 年每个开源语音助手和大多数商业语音助手的默认语音前端。

## 概念

![Whisper 流程：音频 → 梅尔 → 编码器 → 解码器 → 文本](../assets/whisper.svg)

### 第一步 — 重采样 + 窗口

音频 16 kHz。裁剪/填充到 30 秒。计算对数梅尔频谱图：80 个梅尔 bins，10 ms 步长 → ~3,000 帧 × 80 特征。这就是 Whisper 看到的"输入图像"。

### 第二步 — 卷积干

两个 Conv1D 层，核 3，步长 2，将 3,000 帧减少到 1,500。序列长度减半而不添加太多参数。

### 第三步 — 编码器

24 层（large 的情况）transformer 编码器，超过 1,500 时间步。正弦位置编码、自注意力、GELU FFN。产生 1,500 × 1,280 隐藏状态。

### 第四步 — 解码器

24 层 transformer 解码器。它自回归产生来自 BPE 词汇的 token，该词汇是 GPT-2 的超集，带有少量音频特定特殊 token。

### 第五步 — 任务 token

解码器提示以控制 token 开头，告诉模型要做什么：

```
<|startoftranscript|>  <|en|>  <|transcribe|>  <|0.00|>
```

或

```
<|startoftranscript|>  <|fr|>  <|translate|>   <|0.00|>
```

模型在此约定上训练。你通过前缀控制任务。2026 年的指令微调等效，但应用于语音。

### 第六步 — 输出

束搜索（宽度 5）加对数概率阈值。当 `<|notimestamps|>` token 不存在时，每 0.02 秒音频预测时间戳。

### Whisper 规模

| 模型 | 参数 | 层数 | d_model | 头数 | VRAM（fp16） |
|------|------|------|---------|------|-------------|
| Tiny | 39M | 4 | 384 | 6 | ~1 GB |
| Base | 74M | 6 | 512 | 8 | ~1 GB |
| Small | 244M | 12 | 768 | 12 | ~2 GB |
| Medium | 769M | 24 | 1024 | 16 | ~5 GB |
| Large | 1550M | 32 | 1280 | 20 | ~10 GB |
| Large-v3 | 1550M | 32 | 1280 | 20 | ~10 GB |
| Large-v3-turbo | 809M | 32 | 1280 | 20 | ~6 GB（4 层解码器） |

Large-v3-turbo（2024）将解码器从 32 层削减到 4 层。8× 更快的解码，<1 WER 点回归。那是为什么 Whisper-turbo 是 2026 年实时语音代理默认的原因。

### Whisper 不做的事

- 没有 diarization（谁在说话）。与 pyannote 配对。
- 本机无实时流——30 秒窗口是固定的。现代包装器（`faster-whisper`、`WhisperX`）通过 VAD + 重叠附加流式处理。
- 超过 30 秒无长上下文，无需外部分块。实际上效果很好，因为人类语音转录很少需要长程上下文。

### 2026 年格局

| 任务 | 模型 | 备注 |
|------|------|------|
| 英语 ASR | Whisper-turbo、Moonshine | Moonshine 在边缘快 4 倍 |
| 多语言 ASR | Whisper-large-v3 | 97 种语言 |
| 流式 ASR | faster-whisper + VAD | 150 ms 延迟目标可实现 |
| TTS | Piper、XTTS-v2、Kokoro | 编码器-解码器模式，但 Whisper 形状 |
| 音频 + 语言 | AudioLM、SeamlessM4T | 文本 token + 音频 token 在一个 transformer 中 |

## 构建

见 `code/main.py`。我们不训练 Whisper——我们构建对数梅尔频谱图流程 + 任务 token 提示格式化器。那些是你在生产中实际接触的部分。

### 第一步：合成音频

生成 1 秒 440 Hz 正弦波，采样 16 kHz。16,000 个样本。

### 第二步：对数梅尔频谱图（简化）

完整梅尔频谱图需要 FFT。我们做一个简化的分帧 + 每帧能量版本，展示流程而不需要 `librosa`：

```python
def frame_signal(x, frame_size=400, hop=160):
    frames = []
    for start in range(0, len(x) - frame_size + 1, hop):
        frames.append(x[start:start + frame_size])
    return frames
```

Frame = 25 ms，hop = 10 ms。匹配 Whisper 的窗口化。每帧能量代替梅尔 bins 用于教学。

### 第三步：填充到 30 秒

Whisper 始终处理 30 秒块。将频谱图填充（或裁剪）到 3,000 帧。

### 第四步：构建提示 token

```python
def whisper_prompt(lang="en", task="transcribe", timestamps=True):
    tokens = ["<|startoftranscript|>", f"<|{lang}|>", f"<|{task}|>"]
    if not timestamps:
        tokens.append("<|notimestamps|>")
    return tokens
```

这就是整个任务控制面。一个 4 token 前缀。

## 使用

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe("meeting.wav", language="en", task="transcribe")
print(result["text"])
print(result["segments"][0]["start"], result["segments"][0]["end"])
```

更快、OpenAI 兼容：

```python
from faster_whisper import WhisperModel
model = WhisperModel("large-v3-turbo", compute_type="int8_float16")
segments, info = model.transcribe("meeting.wav", vad_filter=True)
for s in segments:
    print(f"{s.start:.2f} - {s.end:.2f}: {s.text}")
```

**2026 年何时选 Whisper：**

- 单模型多语言 ASR。
- 嘈杂、多样音频的鲁棒转录。
- 研究/原型 ASR——最快的起点。

**何时选其他的：**

- 超低延迟边缘流式处理——Moonshine 在匹配质量下比 Whisper 快。
- 需要 <200 ms 的实时会话 AI——专用流式 ASR。
- 说话人 diarization——Whisper 不做这个；附加 pyannote。

## 交付

见 `outputs/skill-asr-configurator.md`。该 skill 为新的语音应用选择 ASR 模型、解码参数和预处理流程。

## 练习

1. **简单。** 运行 `code/main.py`。确认 16 kHz 1 秒信号，10 ms hop，帧数约 100。30 秒：约 3,000 帧。
2. **中等。** 使用 `numpy.fft` 构建完整对数梅尔频谱图。验证 80 个梅尔 bins 在数值误差内匹配 `librosa.feature.melspectrogram(n_mels=80)`。
3. **困难。** 实现流式推理：将音频分块为 10 秒窗口，2 秒重叠，在每个块上运行 Whisper，合并转录。在 5 分钟播客样本上测量 vs 单次通行的词错误率。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|----------|---------|
| 梅尔频谱图 | "音频图像" | 2D 表示：一轴上的频率 bins，另一轴上的时间帧；每格对数缩放能量。 |
| 对数梅尔 | "Whisper 看到的" | 通过 log 的梅尔频谱图；近似人类响度感知。 |
| 帧 | "一个时间切片" | 25 ms 的样本窗口；以 10 ms 步长重叠。 |
| 任务 token | "语音提示前缀" | 解码器提示中的 `<|transcribe|>` / `<|translate|>` 等特殊 token。 |
| 语音活动检测（VAD） | "找语音" | 在 ASR 之前去除静音的门控；大幅降低成本。 |
| CTC | "连接主义时间分类" | 无对齐训练的经典 ASR 损失；Whisper 不使用它。 |
| Whisper-turbo | "小解码器，完整编码器" | large-v3 编码器 + 4 层解码器；8× 更快的解码。 |
| Faster-whisper | "生产包装器" | CTranslate2 重实现；int8 量化；比 OpenAI 参考快 4 倍。 |

## 延伸阅读

- [Radford et al. (2022). Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — Whisper 论文。
- [OpenAI Whisper 仓库](https://github.com/openai/whisper) — 参考代码 + 模型权重。读 `whisper/model.py` 从上到下看 Conv1D 干 + 编码器 + 解码器，约 400 行。
- [OpenAI Whisper — `whisper/decoding.py`](https://github.com/openai/whisper/blob/main/whisper/decoding.py) — 束搜索 + 任务 token 逻辑在第 5–6 步描述，500 行，完全可读。
- [Baevski et al. (2020). wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477) — 前身；在某些设置中仍是 SOTA 特征。
- [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) — 生产包装器，比参考快 4 倍。
- [Jia et al. (2024). Moonshine: Speech Recognition for Live Transcription and Voice Commands](https://arxiv.org/abs/2410.15608) — 2024 年边缘友好 ASR，Whisper 形状但更小。
- [HuggingFace 博客 — "Fine-Tune Whisper For Multilingual ASR with 🤗 Transformers"](https://huggingface.co/blog/fine-tune-whisper) — 包括梅尔频谱图预处理和 token-时间戳处理的标准微调配方。
- [HuggingFace `modeling_whisper.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/modeling_whisper.py) — 完整实现（编码器、解码器、交叉注意力、生成），镜像本课架构图。