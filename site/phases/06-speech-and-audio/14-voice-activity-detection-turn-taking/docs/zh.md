# 语音活动检测与轮次管理 — Silero、Cobra 和 Flush 技巧

> 每个语音 agent 都死于两个决定：用户是否在说话，以及是否说完了？VAD 回答第一个。轮次检测（VAD + 沉默尾随 + 语义端点模型）回答第二个。任何一个搞错，你的助手要么切断用户要么永远不停。

**类型：** 构建
**语言：** Python
**前置知识：** Phase 6 · 11（实时音频）、Phase 6 · 12（语音助手）
**时间：** 约 45 分钟

## 问题

语音 agent 在每个 20 ms 块上做出三个不同的决定：

1. **这一帧是语音吗？** — VAD。二值，每帧。
2. **用户开始新话语了吗？** — 开始检测。
3. **用户结束了吗？** — 端点检测（轮次结束）。

朴素答案（能量阈值）在任何噪声上失败——交通、键盘、人声嘈杂。2026 年答案：Silero VAD（开源、深度学习）+ 轮次检测模型（语义端点）+ VAD 校准的沉默尾随。

## 概念

![VAD 级联：能量 → Silero → 轮次检测器 → Flush 技巧](../assets/vad-turn-taking.svg)

### 三层 VAD 级联

**第 1 层：能量门。** 最便宜。阈值 RMS @ -40 dBFS。过滤明显静音但在阈值以上任何噪声上触发。

**第 2 层：Silero VAD**（2020-2026，MIT）。100 万参数。在 6000+ 种语言上训练。在单个 CPU 线程上每 30 ms 块约 1 ms 运行。5% FPR 下 87.7% TPR。开源默认。

**第 3 层：语义轮次检测器。** LiveKit 的轮次检测模型（2024-2026）或你自己的小型分类器。区分"句子中间的停顿"和"说完了"。使用语言学上下文（语调 + 近期词汇），而不仅是静音。

### 关键参数及其默认值

- **阈值。** Silero 输出概率；分类 speech @ > 0.5（默认）或 > 0.3（敏感）。低阈值 = 少首词截断，多误报。
- **最小语音持续时间。** 拒绝短于 250 ms 的语音——通常是咳嗽或椅子噪音。
- **沉默尾随（端点）。** VAD 返回 0 后，等待 500-800 ms 再声明轮次结束。太短 → 打断用户。太长 → 感觉迟钝。
- **预滚缓冲。** 在 VAD 触发前保留 300-500 ms 音频。防止"嘿"被截断。

### Flush 技巧（Kyutai 2025）

流式 STT 模型有前瞻延迟（Kyutai STT-1B 为 500 ms，STT-2.6B 为 2.5 s）。通常你在语音结束后等待那么长时间才能得到转录。Flush 技巧：当 VAD 触发语音结束时，**向 STT 发送一个 flush 信号**迫使其立即输出。STT 以约 4× 实时处理，所以 500 ms 缓冲约在 125 ms 内完成。

端到端：125 ms VAD + flush STT = 对话语延迟。

### 2026 年 VAD 比较

| VAD | 5% FPR 下 TPR | 延迟 | 许可 |
|-----|--------------|---------|---------|
| WebRTC VAD（Google，2013） | 50.0% | 30 ms | BSD |
| Silero VAD（2020-2026） | 87.7% | ~1 ms | MIT |
| Cobra VAD（Picovoice） | 98.9% | ~1 ms | 商业 |
| pyannote 分段 | 95% | ~10 ms | MIT-ish |

Silero 是正确的默认。Cobra 是合规/准确升级。2026 年生产中没有能量单独 VAD 的位置。

## 构建

### 步骤 1：能量门

```python
def energy_vad(chunk, threshold_dbfs=-40.0):
    rms = (sum(x * x for x in chunk) / len(chunk)) ** 0.5
    dbfs = 20.0 * math.log10(max(rms, 1e-10))
    return dbfs > threshold_dbfs
```

### 步骤 2：Python 中的 Silero VAD

```python
from silero_vad import load_silero_vad, get_speech_timestamps

vad = load_silero_vad()
audio = torch.tensor(waveform_16k, dtype=torch.float32)
segments = get_speech_timestamps(
    audio, vad, sampling_rate=16000,
    threshold=0.5,
    min_speech_duration_ms=250,
    min_silence_duration_ms=500,
    speech_pad_ms=300,
)
for s in segments:
    print(f"{s['start']/16000:.2f}s - {s['end']/16000:.2f}s")
```

### 步骤 3：轮次结束状态机

```python
class TurnDetector:
    def __init__(self, silence_hangover_ms=500, min_speech_ms=250):
        self.state = "idle"
        self.speech_ms = 0
        self.silence_ms = 0
        self.silence_hangover_ms = silence_hangover_ms
        self.min_speech_ms = min_speech_ms

    def update(self, is_speech, chunk_ms=20):
        if is_speech:
            self.speech_ms += chunk_ms
            self.silence_ms = 0
            if self.state == "idle" and self.speech_ms >= self.min_speech_ms:
                self.state = "speaking"
                return "START"
        else:
            self.silence_ms += chunk_ms
            if self.state == "speaking" and self.silence_ms >= self.silence_hangover_ms:
                self.state = "idle"
                self.speech_ms = 0
                return "END"
        return None
```

### 步骤 4：Flush 技巧骨架

```python
def flush_on_end(stt_client, audio_buffer):
    stt_client.send_audio(audio_buffer)
    stt_client.send_flush()
    return stt_client.recv_transcript(timeout_ms=150)
```

STT（Kyutai、Deepgram、AssemblyAI）必须支持 flush 才能工作。Whisper 流式不——它是分块的，总是等待块。

## 使用

| 场景 | VAD 选择 |
|------|---------|
| 开源、快速、通用 | Silero VAD |
| 商业呼叫中心 | Cobra VAD |
| 设备端（手机） | Silero VAD ONNX |
| 研究/日志 | pyannote 分段 |
| 零依赖后备 | WebRTC VAD（传统） |
| 需要轮次结束质量 | Silero + LiveKit 轮次检测器分层 |

经验法则：除非你真的没有其他选择，否则不要发布纯能量 VAD。

## 坑

- **固定阈值。** 在安静时工作，在有噪声时失败。要么在设备上校准要么切换到 Silero。
- **沉默尾随太短。** 助理在句子中间打断。对话语音 500-800 ms 是最佳点。
- **尾随太长。** 感觉迟钝。用目标用户 A/B 测试。
- **无预滚缓冲。** 丢失用户音频的前 200-300 ms。始终保持滚动预滚。
- **忽略语义端点。** "嗯，让我想想……"包含长停顿。用户讨厌在思考中途被切断。使用 LiveKit 的轮次检测器或类似。

## 发货

保存为 `outputs/skill-vad-tuner.md`。为工作负载选择 VAD 模型、阈值、尾随、预滚和轮次检测策略。

## 练习

1. **简单。** 运行 `code/main.py`。它模拟语音 + 静音 + 语音 + 咳嗽序列并测试三层 VAD。
2. **中等。** 安装 `silero-vad`，处理 5 分钟录音，调优阈值以最小化首词截断和误触发。报告精确率/召回率。
3. **困难。** 构建一个迷你轮次检测器：Silero VAD + 最后 10 个词嵌入上的 3 层 MLP（使用 sentence-transformers）。在手工标注的轮次结束数据集上训练。相对于仅 Silero 提升 10% F1。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------|--------|
| VAD | 语音检测器 | 二值每帧：这是语音吗？ |
| 轮次检测 | 端点 | VAD + 沉默尾随 + 语义端点。 |
| 沉默尾随 | 说话后等待 | 声明轮次结束前等待的时间；500-800 ms。 |
| 预滚 | 预语音缓冲 | 在 VAD 触发前保留 300-500 ms 音频。 |
| Flush 技巧 | Kyutai hack | VAD → flush-STT → 125 ms 而非 500 ms 延迟。 |
| 语义端点 | "他们意思是停止了吗？" | ML 分类器看词语，而不仅是静音。 |
| TPR @ FPR 5% | ROC 点 | 标准 VAD 基准；Silero 87.7%，WebRTC 50%。 |

## 延伸阅读

- [Silero VAD](https://github.com/snakers4/silero-vad) — 参考开源 VAD。
- [Picovoice Cobra VAD](https://picovoice.ai/products/cobra/) — 商业准确率领先。
- [Kyutai — Unmute + flush 技巧](https://kyutai.org/stt) — 亚 200 ms 工程技巧。
- [LiveKit — 轮次检测](https://docs.livekit.io/agents/logic/turns/) — 生产中的语义端点。
- [WebRTC VAD](https://webrtc.googlesource.com/src/) — 传统基线。
- [pyannote 分段](https://github.com/pyannote/pyannote-audio) — 日志级分段。