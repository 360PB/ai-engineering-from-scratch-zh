# 顶点课 03 — 实时语音助手（ASR → LLM → TTS）

> 一个体验正确的语音 Agent 端到端延迟低于 800ms，能感知用户何时停止说话，能处理打断，能在不动音频的情况下调用工具。Retell、Vapi、LiveKit Agents、Pipecat 在 2026 年都达到了这个水准。它们用同样的形态做到这一点：流式 ASR、轮次检测器、流式 LLM、流式 TTS，全程 WebRTC，每个节点都有激进的延迟预算。构建一个，测量 WER 和 MOS 及误截断率，并在丢包条件下运行。

**类型：** 顶点课
**语言：** Python（Agent + 管道），TypeScript（Web 客户端）
**前置要求：** Phase 6（语音和音频）、Phase 7（Transformer）、Phase 11（LLM 工程）、Phase 13（工具）、Phase 14（Agent）、Phase 17（基础设施）
**涉及的 Phase：** P6 · P7 · P11 · P13 · P14 · P17
**时间：** 30 小时

## 问题

语音是 2025-2026 年发展最快的 AI UX 类别。技术天花板每个季度都在下降。OpenAI Realtime API、Gemini 2.5 Live、Cartesia Sonic-2、ElevenLabs Flash v3、LiveKit Agents 1.0、Pipecat 0.0.70 都让 800ms 内首次音频输出成为可能。门槛不只是延迟，而是交互感受：不打断用户、不被用户打断、从半句打断中恢复、在对话中调用工具而不卡住音频、在不稳定的移动网络中存活。

靠三个 REST 调用拼接是做不到的。架构必须全程管线流式。构建它，失败模式就会显现：针对电话音频调优的 VAD 在背景电视前误触发、等待永远不会来的标点的轮次检测器、缓冲 400ms 才输出的 TTS。顶点课的任务是在负载下逐个修复这些问题，并发布一份延迟-质量报告。

## 概念

管道有五个流式阶段：**音频输入**（来自浏览器或 PSTN 的 WebRTC）、**ASR**（来自 Deepgram Nova-3 或 faster-whisper 的流式部分转录）、**轮次检测**（VAD 加上读取部分转录完成提示的小型轮次检测模型）、**LLM**（一旦轮次被判定完成就开始流式输出 Token）、**TTS**（在首个 LLM Token 后的约 200ms 内流式输出音频）。

三个跨领域问题。**打断（Barge-in）**：当用户在 Agent 说话时开始说话，TTS 取消，ASR 立即接管。**工具调用**：对话中的函数调用（天气、日历）必须在侧通道运行而不卡住音频；如果延迟超过 300ms，Agent 预填充一个确认 Token（"稍等……"）。**背压**：在丢包条件下，部分转录被保留，VAD 提高语音门阈值，Agent 避免在未确认消息上说话。

测量标准是量化的。WER 在 15dB SNR 的 Hamming VAD 基准上低于 8%。100 次测量通话的首次音频输出 p50 低于 800ms。误截断率低于 3%。TTS MOS 高于 4.2。单个 g5.xlarge 上 50 路并发通话。这些数字是交付物。

## 架构

```
browser / Twilio PSTN
        |
        v
   WebRTC / SIP edge
        |
        v
  LiveKit Agents 1.0  (or Pipecat 0.0.70)
        |
   +----+--------------+--------------+-----------------+
   |                   |              |                 |
   v                   v              v                 v
  ASR              VAD v5         turn-detector     side-channel
(Deepgram         (Silero)          (LiveKit)        tools
 Nova-3 /         speech-gate    completion score    (weather,
 Whisper-v3)       per 20ms        on partials         calendar)
   |                   |              |
   +--------+----------+--------------+
            v
        LLM (streaming)
     GPT-4o-realtime / Gemini 2.5 Flash /
     cascaded Claude Haiku 4.5
            |
            v
        TTS streaming
     Cartesia Sonic-2 / ElevenLabs Flash v3
            |
            v
     audio back to caller
            |
            v
   OpenTelemetry voice traces -> Langfuse
```

## 技术栈

- 传输：LiveKit Agents 1.0（WebRTC）加 Twilio PSTN 网关；Pipecat 0.0.70 作为备选框架
- ASR：Deepgram Nova-3（流式，首个 partial < 300ms）或 faster-whisper Whisper-v3-turbo 自托管
- VAD：Silero VAD v5 加 LiveKit 轮次检测器（读取部分转录的小型 Transformer）
- LLM：OpenAI GPT-4o-realtime（紧密集成）、Gemini 2.5 Flash Live 或级联 Claude Haiku 4.5（流式补全，单独音频路径）
- TTS：Cartesia Sonic-2（最低首字节延迟）、ElevenLabs Flash v3 或开源 Orpheus（自托管）
- 工具：FastMCP 侧通道处理天气/日历/预订；工具超过 300ms 时 Agent 发出填充语
- 可观测：OpenTelemetry 语音 span，Langfuse 语音 trace 含音频回放
- 部署：单个 g5.xlarge（24GB VRAM）用于自托管 Whisper + Orpheus；托管 API 用于最低延迟

## 构建步骤

1. **WebRTC 会话。** 启动一个 LiveKit room 和一个流式麦克风音频的 Web 客户端。在服务端附加一个加入 room 的 agent worker。

2. **ASR 流式。** 将 20ms PCM 帧送入 Deepgram Nova-3（或 GPU 上的 faster-whisper）。订阅部分和最终转录。记录每个 partial 的延迟。

3. **VAD 和轮次检测器。** 在帧流上运行 Silero VAD v5。在语音结束事件时，用最新的部分转录触发 LiveKit 轮次检测器。只有当 VAD 说 500ms 静音且轮次检测器完成分 > 0.6 时，才提交"轮次完成"。

4. **LLM 流。** 轮次完成时，用运行中的对话加最终转录启动 LLM 调用。流式输出 Token。首个 Token 出现时移交给 TTS。

5. **TTS 流。** Cartesia Sonic-2 流式返回音频块。首个块必须在首个 LLM Token 后 200ms 内离开服务端。发送音频块到 LiveKit room；客户端通过 WebRTC 抖动缓冲播放。

6. **打断。** 当 VAD 在 TTS 播放时检测到新用户语音，立即取消 TTS 流，丢弃剩余 LLM 输出，重新就绪 ASR。发布一个 `tts_canceled` span。

7. **工具侧通道。** 将天气和日历注册为函数调用工具。调用时并发触发；如果 300ms 内未解析，让 LLM 发出"稍等，让我查一下"作为填充语；工具返回后恢复。

8. **评测承载体。** 录制 100 次通话。计算 WER（对照保留的转录）、误截断率（TTS 在用户说半句时被取消）、首次音频输出 p50、TTS MOS（人工或 NISQA）、抖动-丢包测试（丢弃 3% 的包）。

9. **负载测试。** 在单个 g5.xlarge 上用合成呼叫器驱动 50 路并发通话。测量持续首次音频输出 p95。

## 使用示例

```
caller: "what is the weather in tokyo tomorrow"
[asr  ] partial @280ms: "what is the"
[asr  ] partial @540ms: "what is the weather"
[turn ] completion score 0.82 at @820ms; commit
[llm  ] first token @960ms
[tool ] weather.tokyo tomorrow -> 68/52 partly cloudy @1140ms
[tts  ] first audio-out @1040ms: "Tokyo tomorrow will be partly cloudy..."
turn latency: 1040ms user-stop -> audio-out
```

## 交付标准

`outputs/skill-voice-agent.md` 是交付物。给定一个领域（客户支持、排程或 kiosk），它启动 LiveKit Agent，配有调优到测量标准之上的 ASR/VAD/LLM/TTS 管道。评分标准如下：

| 权重 | 指标 | 测量方式 |
|:-:|---|---|
| 25 | 端到端延迟 | 跨 100 次录制通话，首次音频输出 p50 低于 800ms |
| 20 | 轮次质量 | Hamming VAD 基准上误截断率低于 3% |
| 20 | 工具调用正确性 | 对话中工具调用返回正确数据而不卡住音频 |
| 20 | 丢包下可靠性 | 注入 3% 丢包后的 WER 和轮次稳定性 |
| 15 | 评测承载体完整性 | 可复现测量，配置公开 |
| **100** | | |

## 练习

1. 将 Deepgram Nova-3 换成 g5.xlarge 上的 faster-whisper v3 turbo。测量延迟和 WER 差距。识别 CPU vs GPU 决策在哪些地方重要。

2. 增加一个打断仲裁策略：当用户在工具调用期间打断时 Agent 做什么？比较三种策略（硬取消、完成工具后停止、排队下一轮）。

3. 运行对抗性轮次检测器测试：用户在句中长时间停顿。调优 VAD 静音阈值和轮次检测器分数阈值以获得最低误截断且不超过 900ms。

4. 通过 Twilio 在 PSTN 上部署同一 Agent。对比 PSTN 首次音频输出与 WebRTC。解释抖动缓冲和编解码器差异。

5. 为非英语语言（日语、西班牙语）添加语音活动检测。测量 Silero VAD v5 误触发率对比语言特定微调版本。

## 关键术语

| 术语 | 别人怎么称呼 | 实际含义 |
|------|-----------------|------------------------|
| Turn detection | "End of utterance" | 给定 VAD 静音和部分转录，决定用户是否说完的分类器 |
| Barge-in | "Interruption handling" | 当 VAD 检测到新用户语音时取消 TTS 播放中段 |
| First-audio-out | "Latency" | 从用户停止说话到首个音频包离开服务端的时间 |
| VAD | "Speech gate" | 将音频帧分类为语音或静音的模型；Silero VAD v5 是 2026 年默认方案 |
| Jitter buffer | "Audio smoothing" | 客户端缓冲，短暂保留包以吸收网络方差 |
| Filler | "Acknowledgment token" | 工具慢时 Agent 发出的短句以避免沉默 |
| MOS | "Mean opinion score" | 感知语音质量评分；NISQA 是自动化代理 |

## 延伸阅读

- [LiveKit Agents 1.0](https://github.com/livekit/agents) — 参考 WebRTC Agent 框架
- [Pipecat](https://github.com/pipecat-ai/pipecat) — 备选 Python 优先流式 Agent 框架
- [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime) — 集成语音模型参考
- [Deepgram Nova-3 文档](https://developers.deepgram.com/docs) — 流式 ASR 参考
- [Silero VAD v5](https://github.com/snakers4/silero-vad) — VAD 参考模型
- [Cartesia Sonic-2](https://docs.cartesia.ai) — 低延迟 TTS 参考
- [Retell AI 架构](https://docs.retellai.com) — 生产语音 Agent 架构
- [Vapi.ai 生产栈](https://docs.vapi.ai) — 备选生产参考