# 流式语音到语音 — Moshi、Sesame CSM、Orpheus

> 语音到语音对话：不做文本兜底，直接从音频到音频。Moshi 以 200 ms 全双工做到这一点。关键不是端到端模型，而是语义-声学分 split 在离散 token 上做自回归。

**类型：** 构建
**语言：** Python
**前置知识：** Phase 6 · 04（ASR）、Phase 6 · 07（TTS）、Phase 6 · 13（神经音频编解码器）、Phase 6 · 11（实时音频）
**时间：** 约 60 分钟

## 问题

传统管道：音频 → ASR（文本）→ LLM（文本）→ TTS（音频）。三个模型，两次模态转换，两次延迟累积。

流式语音到语音：音频 → 模型 → 音频。单一模型，直接输入输出。听起来更自然，可以携带韵律和情感，而不仅是文字。

2026 年有三个主要实现：

1. **Moshi（Kyutai，2024）。** 200 ms 全双工。语义-声学分 split + 流式流式推理。
2. **Sesame CSM（2025）。** 实时对话。类似的 token 级方法。
3. **Orpheus-3B（2025）。** 开源 24 kHz 语音到语音。

关键洞察：不需要端到端音频模型。你可以在离散音频 token 上运行自回归 LM，语义 token 控制内容，声学 token 控制音色和韵律。

## 概念

![流式语音到语音：音频 → 编码 → LM → 解码 → 音频（所有流式）](../assets/streaming-s2s.svg)

### 三种实现路线

**路线 1：级联流式（Whisper + TTS）。** Whisper 做 ASR，流式输出文本到 LLM，TTS 流式输出。最简单，但有文本兜底——韵律丢失。

**路线 2：离散 token LM（Moshi）。** 音频编码为离散 token（Mimi 编解码器）。语言模型在 token 上运行。解码器生成音频波形。所有流式。200 ms 全双工。

**路线 3：端到端音频 LM（当前研究）。** 原始音频波形到原始音频波形。不需要中间离散表示。还在研究阶段，质量不如路线 2。

### Moshi 架构（Mimi + 自回归 LM）

```
输入音频 → Mimi 编码 → 语义 token（内容）+ 声学 token（音色）
                    ↓
        自回归 LM（预测下一个语义 token，声学 token）
                    ↓
        Mimi 解码 → 输出音频波形
```

语义 token 预测：条件是文本（来自 ASR 的部分转录）+ 之前的语义 token。声学 token 预测：条件是语义 token + 参考音频（用于音色）。

**语义-声学分 split 的关键：** 语义 token 编码"说了什么"。声学 token 编码"怎么说"（韵律、音色、情感）。这种分解使零镜头声音克隆和可控韵律成为可能。

### 流式实现细节

流式语音到语音的困难在于延迟。所有阶段必须同时运行：

```python
async def streaming_s2s(audio_input, reference_audio):
    # 持续流式
    while True:
        # 1. 接收麦克风音频，VAD 检测
        mic_chunk = await mic_stream.read()
        if vad.is_speech(mic_chunk):
            # 2. 持续流式编码
            codes = mimi.encode(mic_chunk)
            semantic = codes[:, 0]   # 码本 0
            acoustic = codes[:, 1:] # 码本 1-7

            # 3. 流式 LM 推理（条件：语义 + 参考音频）
            lm_output = await lm.step(semantic, reference_audio)

            # 4. 解码为音频
            audio_out = mimi.decode(lm_output)
            await speaker.play(audio_out)
```

Moshi 的技巧：Flush 技巧（来自 Lesson 14）用于语音结束后立即得到转录。LM 预测语义 token（条件是文本），然后预测声学 token。两者都流式。

### Orpheus-3B（开源替代）

```python
from TTS.api import TTS

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
# Orpheus 3B：给定文本和参考音频，生成语音
audio = tts.tts(
    text="Hello, how can I help you today?",
    speaker_wav="reference_voice.wav",
    language="en"
)
```

不是真正的流式，但开源语音到语音质量最高。

### 2026 年语音到语音模型

| 模型 | 延迟 | 全双工 | 声音克隆 | 许可 |
|------|------|--------|---------|------|
| Moshi（Kyutai） | 200 ms | 是 | 零镜头 | 研究 |
| Sesame CSM | ~300 ms | 是 | 零镜头 | 商业 |
| Orpheus-3B | ~500 ms | 否 | 零镜头 | Apache-2.0 |
| GPT-4o-realtime | ~320 ms | 是 | 零镜头 | API |
| ElevenLabs Turbo | ~400 ms | 部分 | 零镜头 | 商业 API |

## 构建

### 步骤 1：Mimi 编码

```python
from moshi.models import loaders
mimi = loaders.get_mimi()

# 输入音频 16kHz
with torch.no_grad():
    codes = mimi.encode(waveform)  # (1, 8, frames)

semantic = codes[:, 0]   # 内容 token
acoustic = codes[:, 1:]  # 音色/韵律 token
```

### 步骤 2：语义 token LM

```python
def lm_step(semantic_tokens, text_prefix, lm_model):
    # 给定当前语义上下文 + 文本 prefix，预测下一个语义 token
    ctx = build_context(semantic_tokens, text_prefix)
    next_token = lm_model.predict(ctx)
    return next_token
```

### 步骤 3：声学 token 生成

```python
def generate_acoustic(semantic_token, reference_audio, codec_model):
    # 条件：语义 token + 参考音频（音色）
    acoustic_tokens = codec_model.predict_next(
        semantic=semantic_token,
        reference=reference_audio
    )
    return acoustic_tokens
```

### 步骤 4：Mimi 解码

```python
def decode_to_waveform(semantic, acoustic):
    codes = torch.cat([semantic.unsqueeze(1), acoustic], dim=1)
    with torch.no_grad():
        waveform = mimi.decode(codes)
    return waveform
```

## 使用

| 场景 | 选择 |
|------|------|
| 研究、流式、全双工 | Moshi（Kyutai） |
| 商业实时对话 | Sesame CSM 或 GPT-4o Audio |
| 开源语音到语音 | Orpheus-3B（需适配流式） |
| 商业 API | ElevenLabs Turbo |
| 需要文本兜底 | faster-whisper + Kokoro |

## 坑

- **全双工的 AEC 要求。** 如果助理的声音进入麦克风并触发 ASR，你会得到反馈循环。需要 AEC。
- **语义-声学同步。** 生成过程中，语义 token 和声学 token 必须同步。不能先预测所有语义再生成声学——韵律会落后。
- **流式中断。** 在流式生成期间检测打断更难，因为输出已部分播放。需要提前规划 TTS 停止点。
- **多语言支持。** Moshi 主要针对英语。商业系统如 GPT-4o Audio 支持更多语言。
- **音频格式转换。** 编解码器以特定采样率输出（Mimi 24 kHz）。需要重采样才能与其他组件匹配。

## 发货

保存为 `outputs/skill-s2s-picker.md`。为给定延迟和质量要求选择流式语音到语音实现路线。

## 练习

1. **简单。** 运行 `code/main.py`。模拟语义-声学分 split：输入文本，生成语义 token，然后生成对应的声学 token。
2. **中等。** 安装 Orpheus-3B 或 Kokoro。生成 10 秒语音。用参考音频克隆声音。报告 SECS 和 MOS。
3. **困难。** 构建最小流式语音到语音：Mimi 编码 + 小型 LM（1-2 层）在语义 token 上 + Mimi 解码。在你自己的语音上测试。测量延迟。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------|--------|
| 语音到语音（S2S） | 直接音频转换 | 输入和输出都是音频，不经过文本。 |
| 全双工 | 双向同时 | 用户和助理可以同时说话。 |
| 语义-声学分 | 内容 vs 音色 | 语义 token 编码说了什么，声学 token 编码怎么说。 |
| 流式推理 | 增量生成 | 边接收边生成，不等待完整输入。 |
| Flush 技巧 | Kyutai 低延迟技巧 | VAD 结束 → 强制 STT 输出 → 立即开始回复。 |
| 流式解码 | token 级输出 | 每个语义/声学 token 生成后立即解码为音频。 |

## 延伸阅读

- [Kyutai (2024). Moshi](https://kyutai.org/Moshi.pdf) — 200 ms 全双工架构。
- [Kyutai (2024). Mimi codec](https://kyutai.org/codec-explainer) — 语义-声学分 split 的驱动。
- [Sesame CSM](https://arxiv.org/abs/2407.02178) — 实时对话语音模型。
- [Orpheus-3B](https://github.com/CartesiaAI/orpheus-tts) — 开源语音到语音 3B。
- [GPT-4o Audio](https://openai.com/index/gpt-4o-audio) — OpenAI 的语音 API。
- [Streaming E2E-TTS survey](https://arxiv.org/abs/2305.10106) — 流式 TTS 的完整综述。