# 语音 Agent：Pipecat 与 LiveKit

> 语音 Agent 是 2026 年的一类正式生产级产品。Pipecat 提供 Python 帧流水线（VAD → STT → LLM → TTS → 传输层）。LiveKit Agents 将 AI 模型通过 WebRTC 与用户连接。高品质技术栈的端到端延迟目标为 450–600ms。

**类型：** 概念学习
**语言：** Python（标准库）
**前置知识：** 第 14 阶段 · 01（Agent 循环），第 14 阶段 · 12（工作流模式）
**时间：** 约 60 分钟

## 学习目标

- 描述 Pipecat 的帧流水线：DOWNSTREAM（源→汇）和 UPSTREAM（控制）。
- 列出标准语音流水线的各个阶段以及 Pipecat 支持的传输层。
- 解释 LiveKit Agents 的两种语音 Agent 类（MultimodalAgent、VoicePipelineAgent）及其各自的适用场景。
- 总结 2026 年生产级延迟预期及如何影响架构选型。

## 问题背景

语音 Agent 不是把 TTS 简单焊接到文本循环上。延迟预算极其严苛（约 600ms），部分音频是常态，打断检测本身就是一个模型，传输层涵盖电话 SIP 到 WebRTC。你要么构建帧流水线（Pipecat），要么依赖平台（LiveKit）。

## 核心概念

### Pipecat（pipecat-ai/pipecat）

- Python 帧流水线框架。
- `Frame` → `FrameProcessor` 链。
- 两种流向：
  - **DOWNSTREAM** — 源 → 汇（音频进，TTS 出）。
  - **UPSTREAM** — 反馈和控制（取消、指标、打断）。
- `PipelineTask` 管理生命周期，包含事件（`on_pipeline_started`、`on_pipeline_finished`、`on_idle_timeout`）和用于指标/追踪/RTVI 的观察者。

典型流水线：

```
VAD (Silero) → STT → LLM（上下文在用户/助手之间交替）→ TTS → 传输层
```

传输层支持：Daily、LiveKit、SmallWebRTCTransport、FastAPI WebSocket、WhatsApp。

Pipecat Flows 提供结构化对话（状态机）。Pipecat Cloud 是托管运行时。

### LiveKit Agents（livekit/agents）

- 通过 WebRTC 将 AI 模型与用户连接。
- 核心概念：`Agent`、`AgentSession`、`entrypoint`、`AgentServer`。
- 两种语音 Agent 类：
  - **MultimodalAgent** — 通过 OpenAI Realtime 或同类接口直接处理音频。
  - **VoicePipelineAgent** — STT → LLM → TTS 级联；在文本层保持控制。
- 基于 Transformer 模型的语义打断检测。
- 原生 MCP 集成。
- 支持 SIP 电话。
- 通过 LiveKit Inference 免费接入 50+ 模型；通过插件接入 200+ 更多模型。

### 商业平台

Vapi（优化高端技术栈约 450–600ms）和 Retell（180 次测试通话端到端约 600ms）均构建在前两者之上。需要托管语音栈且没有 WebRTC 团队时，选平台。

### 这个模式的常见误区

- **没有打断处理。** 用户中断但 Agent 继续说话。需要 Pipecat 中的 UPSTREAM cancel 帧，LiveKit 中等效机制。
- **忽略 STT 置信度。** 低置信度转写被当作真理传给 LLM。设置信度门控或请求确认。
- **TTS 截断不通知。** 流水线的中间取消时，TTS 需要知道，否则音频被截断。
- **忽视延迟预算。** 每个组件增加 50–200ms。上线前累加你的链路延迟。

### 2026 年典型延迟数据

- VAD：20–60ms
- STT 部分结果：100–250ms
- LLM 首 token：150–400ms
- TTS 首段音频：100–200ms
- 传输层 RTT：30–80ms

端到端 450–600ms 是高端水准。800–1200ms 很常见。超过 1500ms 则体验崩溃。

## 动手实现

`code/main.py` 是一个基于帧的玩具流水线：

- `Frame` 类型（audio、transcript、text、tts_audio、control）。
- `Processor` 接口，带 `process(frame)` 方法。
- 五阶段流水线（VAD → STT → LLM → TTS → 传输层）以脚本化处理器实现。
- 一个 UPSTREAM cancel 帧演示打断处理。

运行：

```
python3 code/main.py
```

追踪记录展示正常流程，以及一个打断取消如何中途停止 TTS 语音输出。

## 用现成库

- **Pipecat** — 需要完全控制：自定义处理器、Python 优先、可插拔provider。
- **LiveKit Agents** — WebRTC 优先部署和电话场景。
- **Vapi / Retell** — 无 WebRTC 团队时的托管语音 Agent。
- **OpenAI Realtime / Gemini Live** — 直接音频进/音频出（MultimodalAgent）。

## 产出

`outputs/skill-voice-pipeline.md` 脚手架出一个 Pipecat 风格的语音流水线，含 VAD + STT + LLM + TTS + 传输层及打断处理。

## 练习

1. 给玩具流水线添加指标观察者：每秒每阶段帧数。延迟在哪里累积？
2. 实现置信度门控的 STT：低于阈值时请求"请再说一遍？"
3. 添加语义打断检测：简单规则——转写文本以"？"结尾则本轮结束。
4. 阅读 Pipecat 的传输层文档。将标准库传输层替换为 SmallWebRTCTransport 配置（桩）。
5. 对同一问题测量 OpenAI Realtime 与 STT+LLM+TTS 级联。文本层控制带来了多少延迟代价？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| Frame | "事件" | 流水线中的类型化数据单元（audio、transcript、text、control） |
| Processor | "流水线阶段" | 带 process(frame) 的处理程序 |
| DOWNSTREAM | "正向流" | 源到汇：音频进，语音出 |
| UPSTREAM | "反馈流" | 控制信号：取消、指标、打断 |
| VAD | "语音活动检测" | 检测用户是否在说话 |
| Semantic turn detection | "智能轮次结束判断" | 基于模型的决策，判断用户是否已说完 |
| MultimodalAgent | "直接音频 Agent" | 音频进，音频出，中间无文本 |
| VoicePipelineAgent | "级联 Agent" | STT + LLM + TTS；在文本层保持控制 |

## 延伸阅读

- [Pipecat 文档](https://docs.pipecat.ai/getting-started/introduction) — 帧流水线、处理器、传输层
- [LiveKit Agents 文档](https://docs.livekit.io/agents/) — WebRTC + 语音原语
- [Vapi](https://vapi.ai/) — 托管语音平台
- [Retell AI](https://www.retellai.com/) — 托管语音，延迟有基准数据