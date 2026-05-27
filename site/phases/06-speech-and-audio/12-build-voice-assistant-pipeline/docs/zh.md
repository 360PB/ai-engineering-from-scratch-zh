# 构建语音助手管道

> VAD → ASR → LLM → TTS → 音频渲染。每个阶段都有自己的延迟、错误模式和工程决策。将它们粘合在一起的是状态机、缓冲和中断处理。

**类型：** 构建
**语言：** Python
**前置知识：** Phase 6 · 04（ASR）、Phase 6 · 07（TTS）、Phase 6 · 11（实时音频）、Phase 6 · 14（VAD）、Phase 12 · 05（LangChain）
**时间：** 约 75 分钟

## 问题

单独运行 Whisper 效果好。单独运行 Kokoro TTS 效果也好。将它们连接成一个感觉像对话的管道是另一回事：

1. **流式 vs 批量。** ASR 输出部分转录。TTS 需要完整句子才能产生好的音频。中间的句子边界检测需要语言理解，不是简单的静音。
2. **延迟链。** 每个阶段有延迟。用户说完了 → 检测 → ASR → LLM → TTS → 音频。在你开口之前已经有几百毫秒过去了。
3. **全双工。** 用户可以在你说话时打断。你需要停止 TTS、清除 LLM 输出、切换到 ASR——同时 TTS 不会产生可听见的爆音。

2026 年最佳实践：将管道分解为阶段，每个阶段独立流式，并在它们之间使用有界缓冲区和状态机。

## 概念

![语音助手管道：VAD → ASR → 对齐 → LLM → TTS → 渲染](../assets/voice-assistant.svg)

### 完整管道

```
麦克风音频流
  │
  ▼
VAD（Silero，20ms 块）→ 语音段
  │
  ▼
流式 ASR（faster-whisper 或 Parakeet）→ 部分转录 + 最终转录
  │
  ▼
句子对齐（标点模型或实时检测）→ 完整句子
  │
  ▼
LLM（Groq/vLLM/Cerebras）→ 回复文本（流式 token）
  │
  ▼
流式 TTS（Kokoro 或 ElevenLabs）→ 音频块
  │
  ▼
音频渲染 + 打断处理 → 扬声器

用户说话 ← → 打断检测
```

### 状态机

```python
class VoiceAssistant:
    def __init__(self):
        self.state = "idle"
        self.transcript_buffer = []
        self.llm_stream = None
        self.tts_task = None
        self.audio_out = RingBuffer(capacity=32000)

    def on_vad_start(self):
        self.state = "listening"
        # 记录开始

    def on_vad_end(self, audio_segment):
        self.state = "processing"
        # 送入 ASR
        final_text = asr.transcribe(audio_segment)
        # 检测句子边界
        sentences = split_sentences(final_text)
        for sent in sentences:
            self.respond(sent)

    def respond(self, text):
        # 流式 LLM
        self.llm_stream = llm.stream(text)
        for token in self.llm_stream:
            # 流式 TTS
            audio = tts.stream(token)
            self.audio_out.write(audio)
            self.render()

    def on_barge_in(self):
        # 用户打断
        if self.tts_task:
            self.tts_task.cancel()
        if self.llm_stream:
            self.llm_stream.cancel()
        self.audio_out.clear()
        self.state = "listening"
```

### 句子边界检测

TTS 需要完整句子才能产生好的韵律。使用标点模型或简单的规则：

- 基于规则：`text.endswith(('.', '?', '!'))`
- 小型标点模型：`openai/whisper-large-v3` 有语言检测
- 流式：`s.enter_context(nlp.sentence_boundary())` 来自一些 NLTK 替代品

策略：缓冲直到检测到句子结束符，然后一次性发送 TTS。

### 语音活动检测 → 流式 ASR 连接

```python
def on_speech_end(audio_segment):
    # 将最终音频片段发送到 ASR
    text = asr_model.transcribe(segment, language=lang)
    # 追加到对话历史
    conversation_history.append({"role": "user", "content": text})

    # 检测句子边界并流式回复
    sentences = split_sentences(text)
    for sentence in sentences:
        stream_response(sentence)
```

### 音频渲染细节

```python
import sounddevice as sd

stream = sd.OutputStream(
    samplerate=24000,
    channels=1,
    callback=audio_callback,
    blocksize=480  # 20 ms @ 24 kHz
)
stream.start()

def audio_callback(outdata, frames, time, status):
    chunk = audio_buffer.read(frames)
    if len(chunk) < frames:
        outdata[:len(chunk)] = chunk
        outdata[len(chunk):] = 0
    else:
        outdata[:] = chunk
```

## 2026 年参考栈

| 组件 | 推荐 | 备注 |
|------|------|------|
| VAD | Silero VAD 4.0 | <1ms, MIT |
| 流式 ASR | faster-whisper + VAD | 150ms 延迟 |
| LLM | Groq (Llama 3.3 70B) | 首个 token ~500ms |
| 流式 TTS | Kokoro-80k (24 kHz) | ~100ms 预热 |
| 音频渲染 | sounddevice + 环形缓冲 | 跨平台 |
| 中断检测 | LiveKit Agents | 完整框架 |
| 全双工 | Moshi (Kyutai) | 端到端参考 |

## 构建

### 步骤 1：麦克风输入循环

```python
import sounddevice as sd

def mic_loop(asr_model, callback, blocksize=5120, samplerate=16000):
    # blocksize=5120 @ 16kHz = 320ms
    def audio_callback(indata, frames, time, status):
        audio = indata[:, 0]
        callback(audio)
    stream = sd.InputStream(
        samplerate=samplerate,
        blocksize=blocksize,
        device=None,  # 默认麦克风
        channels=1,
        dtype='float32',
        callback=audio_callback
    )
    stream.start()
    return stream
```

### 步骤 2：句子分割

```python
import re

def split_sentences(text):
    # 基于规则的句子边界
    # 保留中文标点
    sentences = re.split(r'[。！？\n]+', text)
    return [s.strip() for s in sentences if s.strip()]
```

### 步骤 3：流式 LLM + TTS

```python
async def stream_response(user_text):
    history.append({"role": "user", "content": user_text})

    # 构建 prompt
    prompt = build_prompt(history)

    # 流式 LLM
    llm_stream = await llm.stream(prompt)

    # 流式 TTS
    tts_stream = tts.stream()

    async for token in llm_stream:
        audio = await tts_stream.send(token)
        speaker.play(audio)

    await tts_stream.finish()
```

### 步骤 4：打断处理

```python
def on_interrupt():
    global tts_stream, llm_stream

    # 取消 TTS
    if tts_stream:
        tts_stream.cancel()
        tts_stream = None

    # 取消 LLM
    if llm_stream:
        llm_stream.cancel()
        llm_stream = None

    # 清除音频缓冲
    audio_buffer.clear()

    # 切换到聆听模式
    state = "listening"
```

## 使用

对于生产部署：

- **LiveKit Agents 框架。** 2026 年生产级语音代理的参考实现。包括 VAD、ASR、LLM、TTS 和中断处理的完整管道。
- **VAD-first 设计。** 始终先运行 VAD。仅当检测到语音时才触发 ASR。节省成本，减少幻觉。
- **两阶段确认。** 部分转录用于实时反馈；最终转录用于 LLM。

## 坑

- **标点缺失。** Whisper 输出无标点。TTS 需要标点才能正确放置停顿。在 ASR 和 LLM 之间插入标点模型。
- **LLM 延迟。** Groq 的首个 token 约 500 ms。对话式 AI 可能仍然感觉慢。缓存常见响应（如"你好"）。
- **TTS 首 chunk 延迟。** Kokoro 在首个请求上有 100-200 ms 预热。在对话空闲时预热 TTS。
- **音频格式不匹配。** ASR 需要 16 kHz 单声道 float32。TTS 输出 24 kHz 单声道 int16 或 float32。使用环形缓冲和格式转换。
- **全双工 AEC。** 没有回声消除，TTS 输出重新进入麦克风触发 ASR。WebRTC AEC3 是最小解决方案。

## 发货

保存为 `outputs/skill-voice-assistant-architect.md`。设计一个完整的语音助手管道，包含延迟预算、组件选择和中断处理。

## 练习

1. **简单。** 运行 `code/main.py`。模拟管道：VAD 模拟 → ASR 模拟 → LLM 模拟 → TTS 模拟。打印每个阶段的延迟。
2. **中等。** 用 `sounddevice` 接入你的麦克风，运行 VAD（Silero）+ 流式 ASR（faster-whisper）。实时打印部分转录。
3. **困难。** 构建完整管道：麦克风 → VAD → faster-whisper 流式 → Llama 3.3 70B via Groq → Kokoro TTS → 扬声器。包括打断处理。测量玻璃到玻璃延迟。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------|--------|
| 管道 | 端到端语音助手 | 连接 VAD、ASR、LLM、TTS 的状态机。 |
| 玻璃到玻璃 | 端到端延迟 | 用户说完到听到响应的总延迟。 |
| 句子边界 | 标点检测 | 将连续文本分割为完整句子的逻辑。 |
| 打断处理 | 打断检测和响应 | 在 TTS 播放时检测用户语音并停止播放。 |
| 流式渲染 | 实时音频输出 | 在 TTS 生成时播放音频，无需等待完整句子。 |
| AEC | 回声消除 | 去除扬声器到麦克风的反馈以避免 ASR 触发。 |

## 延伸阅读

- [LiveKit Agents](https://docs.livekit.io/agents/) — 生产级语音代理框架。
- [Kyutai Moshi](https://kyutai.org/Moshi.pdf) — 200ms 全双工端到端参考。
- [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime) — WebRTC + TTS + 内置打断。
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — 流式 ASR 生产包装器。
- [Kokoro TTS](https://github.com/hexgrad/Kokoro) — 快速 24kHz TTS。