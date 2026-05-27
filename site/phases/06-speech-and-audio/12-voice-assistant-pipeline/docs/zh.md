# 构建语音助手流水线 — Phase 6 毕业项目

> 第01-11课的一切串联起来。构建一个听、思考、回话的语音助手。2026年这已是解决的工程问题，不是研究问题——但集成细节决定它能否发货。

**类型：** 动手实现
**语言：** Python
**前置知识：** Phase 6 · 04、05、06、07、11；Phase 11 · 09（函数调用）；Phase 14 · 01（Agent 循环）
**时长：** 约120分钟

## 问题

构建端到端助手：

1. 捕获麦克风输入（16kHz 单声道）。
2. 检测用户语音的开始/结束。
3. 转录流式语音。
4. 传给能调用工具（计时器、天气、日历）的 LLM。
5. 将 LLM 文本流式传到 TTS。
6. 播放音频给用户。
7. 用户中断时停止。

延迟目标：用户说完后 800ms 内笔记本 CPU 上输出第一个 TTS 音频字节。质量目标：无漏词，静默上无幻觉字幕，无声音克隆泄漏，无 prompt 注入成功。

## 核心概念

### 七个组件

1. **音频捕获。** 麦克风 → 16kHz 单声道 → 20ms 块。
2. **VAD（第11课）。** Silero VAD @ 阈值0.5，最短语音250ms，静默缓冲500ms。信号"开始"和"结束"。
3. **流式 STT（第4-5课）。** Whisper-streaming、Parakeet-TDT 或 Deepgram Nova-3（API）。部分 + 最终转录。
4. **带工具调用的 LLM。** GPT-4o / Claude 3.5 / Gemini 2.5 Flash。JSON schema 工具。流式 token。
5. **流式 TTS（第7课）。** Kokoro-82M（最快开源）或 Cartesia Sonic（商业）。20个 LLM token 后开始 TTS。
6. **播放。** 扬声器输出；opus 编码用于低带宽网络。
7. **中断处理器。** TTS 播放期间 VAD 触发时，停止播放，取消 LLM，重新开始 STT。

### 三个常见失败模式

1. **首词截断。** VAD 触发稍晚。"hey"被截掉。阈值设0.3而非0.5。
2. **中断后响应混乱。** 用户中断后 LLM 继续生成；助手在用户上说话。接入 VAD → 取消 LLM。
3. **静默幻觉。** Whisper 在静默预热帧上输出"Thanks for watching"。总是 VAD 门控。

## 动手实现

### 步骤 1：VAD 检测

```python
import torch
model, utils = torch.hub.load('snakers4/silero-vad', 'silero_vad')
vad_fn = utils[0]
# 检测语音开始/结束
```

### 步骤 2：流式 STT

```python
import whisper
model = whisper.load_model("large-v3-turbo")
# 流式转录...
```

### 步骤 3：LLM 工具调用

```python
# 带工具的流式对话
messages = [{"role": "user", "content": transcript}]
stream = client.messages.stream(model="claude-3-5-sonnet", messages=messages, tools=tools)
```

### 步骤 4：流式 TTS + 中断

```python
from kokoro import KPipeline
tts = KPipeline(voice="af_bella")
for text_chunk in streamed_tokens:
    audio = tts.run(text_chunk)
    play(audio)
```

## 陷阱

- **首词截断。** VAD 阈值调低。
- **中断响应混乱。** 接入 VAD 取消 LLM。
- **静默幻觉。** 总是先 VAD 门控再调 Whisper。

## 产出

保存为 `outputs/skill-voice-assistant.md`。设计给定延迟和质量目标的语音助手流水线。