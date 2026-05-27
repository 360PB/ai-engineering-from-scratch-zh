# 实时音频处理

> 批处理管道处理文件。实时管道在下一批 20 毫秒到达之前处理下一批 20 毫秒。每个对话 AI、广播工作室和电话机器人都在这个延迟预算上生死存亡。

**类型：** 构建
**语言：** Python
**前置知识：** Phase 6 · 02（频谱图）、Phase 6 · 04（ASR）、Phase 6 · 07（TTS）
**时间：** 约 75 分钟

## 问题

你想要一个感觉有生命力的语音助手。人类对话轮次延迟约 230 ms（静音到响应）。超过 500 ms 感觉像机器人；超过 1500 ms 感觉坏了。2026 年完整 **听 → 理解 → 响应 → 说话** 环路的预算：

| 阶段 | 预算 |
|------|------|
| 麦克风 → 缓冲 | 20 ms |
| VAD | 10 ms |
| ASR（流式） | 150 ms |
| LLM（首个 token） | 100 ms |
| TTS（首个块） | 100 ms |
| 渲染 → 扬声器 | 20 ms |
| **总计** | **约 400 ms** |

Moshi（Kyutai，2024）达到 200 ms 全双工。GPT-4o-realtime（2024）约 320 ms。2022 年级联管道发货在 2500 ms。10 倍改进来自三个技术：（1）无处不流式，（2）异步流水线带部分结果，（3）可中断生成。

## 概念

![流式音频管道：环形缓冲、VAD 门控、中断](../assets/real-time.svg)

**帧 / 块 / 窗口。** 实时音频以固定大小块流动。常见选择：20 ms（在 16 kHz 下 320 采样）。下游所有东西必须跟上这个节拍。

**环形缓冲。** 固定大小循环缓冲区。生产者线程写新帧，消费者线程读。防止热路径上的分配。大小 ≈ 最大延迟 × 采样率；2 秒 16 kHz 环形 = 32,000 采样。

**VAD（语音活动检测）。** 当没人说话时门控下游工作。Silero VAD 4.0（2024）在 CPU 上每 30 ms 帧 <1 ms 运行。`webrtcvad` 是较旧的替代品。

**流式 ASR。** 音频到达时发出部分转录的模型。Parakeet-CTC-0.6B 流式模式（NeMo，2024）在 320 ms 延迟下实现 2–5% WER。Whisper-Streaming（Macháček 等，2023）将 Whisper 分块以近乎流式运行，约 2 秒延迟。

**中断。** 当用户说话而助手在说话时，你必须（a）检测打断，（b）停止 TTS，（c）丢弃剩余的 LLM 输出。全在 100 ms 内完成，否则用户感觉助理是聋的。

**WebRTC Opus 传输。** 20 ms 帧，48 kHz，自适应比特率 8–128 kbps。浏览器和移动端的标准。LiveKit、Daily.co、Pion 是 2026 年构建语音应用的主流栈。

**抖动缓冲。** 网络数据包乱序/延迟到达。抖动缓冲器重新排序和平滑；太小 → 明显间隙，太大 → 延迟。典型 60–80 ms。

### 常见陷阱

- **线程竞争。** Python 的 GIL + 重量级模型可能饿死音频线程。使用 C 回调音频库（sounddevice、PortAudio）并让 Python 离开热路径。
- **重采样延迟。** 管道内重采样增加 5–20 ms。要么预先重采样，要么使用零延迟重采样器（PolyPhase，`soxr_hq`）。
- **TTS 预热。** 即使是 Kokoro 这样的快速 TTS 在首次请求上也有 100–200 ms 预热。在首次真实轮次前缓存模型并用虚拟运行预热。
- **回声消除。** 没有 AEC，TTS 输出重新进入麦克风并在助理自己的声音上触发 ASR。WebRTC AEC3 是开源默认。

## 构建

### 步骤 1：环形缓冲

```python
import collections

class RingBuffer:
    def __init__(self, capacity):
        self.buf = collections.deque(maxlen=capacity)
    def write(self, frame):
        self.buf.extend(frame)
    def read(self, n):
        return [self.buf.popleft() for _ in range(min(n, len(self.buf)))]
    def level(self):
        return len(self.buf)
```

容量决定最大缓冲延迟。32,000 采样 @ 16 kHz = 2 秒。

### 步骤 2：VAD 门控

```python
def simple_energy_vad(frame, threshold=0.01):
    return sum(x * x for x in frame) / len(frame) > threshold ** 2
```

生产中替换为 Silero VAD：

```python
import torch
vad, _ = torch.hub.load("snakers4/silero-vad", "silero_vad")
is_speech = vad(torch.tensor(frame), 16000).item() > 0.5
```

### 步骤 3：流式 ASR

```python
# Parakeet-CTC-0.6B 流式 via NeMo
from nemo.collections.asr.models import EncDecCTCModelBPE
asr = EncDecCTCModelBPE.from_pretrained("nvidia/parakeet-ctc-0.6b")
# chunk_ms=320 ms, look_ahead_ms=80 ms
for chunk in audio_stream():
    partial_text = asr.transcribe_streaming(chunk)
    print(partial_text, end="\r")
```

### 步骤 4：中断处理器

```python
class Dialog:
    def __init__(self):
        self.tts_task = None

    def on_user_speech(self, frame):
        if self.tts_task and not self.tts_task.done():
            self.tts_task.cancel()   # 打断
        # 然后送入流式 ASR

    def on_final_user_utterance(self, text):
        self.tts_task = asyncio.create_task(self.reply(text))

    async def reply(self, text):
        async for tts_chunk in llm_then_tts(text):
            speaker.write(tts_chunk)
```

依赖于异步 I/O 和可取消 TTS 流式。WebRTC peerconnection.stop() 在音频轨上是标准方法。

## 使用

2026 年栈：

| 层 | 选择 |
|------|------|
| 传输 | LiveKit（WebRTC）或 Pion（Go） |
| VAD | Silero VAD 4.0 |
| 流式 ASR | Parakeet-CTC-0.6B 或 Whisper-Streaming |
| LLM 首个 token | Groq、Cerebras、vLLM-streaming |
| 流式 TTS | Kokoro 或 ElevenLabs Turbo v2.5 |
| 回声消除 | WebRTC AEC3 |
| 端到端原生 | OpenAI Realtime API 或 Moshi |

## 坑

- **为安全缓冲 500 ms。** 缓冲器 *就是* 你的延迟底线。缩小它。
- **不固定线程。** 音频回调在优先级低于 UI 的线程上 = 高负载下卡顿。
- **TTS 块太小。** 低于 200 ms 的块使声码器伪影可听见。320 ms 块是最佳点。
- **无抖动缓冲。** 真实网络有抖动；没有平滑你会得到爆音。
- **单次错误处理。** 音频管道必须是防崩溃的。一次异常杀死会话。

## 发货

保存为 `outputs/skill-realtime-designer.md`。用每阶段具体延迟预算设计实时音频管道。

## 练习

1. **简单。** 运行 `code/main.py`。模拟环形缓冲 + 能量 VAD；为假 10 秒流打印各阶段延迟。
2. **中等。** 使用 `sounddevice` 构建一个直通循环，以 20 ms 帧处理你的麦克风并在每帧打印 VAD 状态。
3. **困难。** 用 `aiortc` 构建全双工回声测试：浏览器 → WebRTC → Python → WebRTC → 浏览器。用 1 kHz 脉冲测量玻璃到玻璃延迟。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------|--------|
| 环形缓冲 | 循环队列 | 固定大小、无锁（或 SPSC 锁定）FIFO 用于音频帧。 |
| VAD | 静音门 | 模型或启发式标记语音 vs 非语音。 |
| 流式 ASR | 实时 STT | 音频到达时发出部分文本；有界前瞻。 |
| 抖动缓冲 | 网络平滑器 | 重新排序乱序数据包的队列；典型 60–80 ms。 |
| AEC | 回声消除 | 减去扬声器到麦克风的反馈路径。 |
| 打断 | 用户中断 | 系统检测用户在中途说话；必须取消播放。 |
| 全双工 | 双向同时 | 用户和机器人可以同时说话；Moshi 是全双工。 |

## 延伸阅读

- [Macháček et al. (2023). Whisper-Streaming](https://arxiv.org/abs/2307.14743) — 分块近乎流式 Whisper。
- [Kyutai (2024). Moshi](https://kyutai.org/Moshi.pdf) — 全双工 200 ms 延迟。
- [LiveKit Agents framework (2024)](https://docs.livekit.io/agents/) — 生产音频代理编排。
- [Silero VAD repo](https://github.com/snakers4/silero-vad) — 亚 1 ms VAD，Apache 2.0。
- [WebRTC AEC3 论文](https://webrtc.googlesource.com/src/+/main/modules/audio_processing/aec3/) — 开源回声消除。