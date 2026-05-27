# 文本转语音（TTS）— 从 Tacotron 到 F5 和 Kokoro

> ASR 将语音反转成文本；TTS 将文本反转成语音。2026年技术栈分三部分：文本 → token，token → mel，mel → 波形。每部分都有一个可在笔记本上运行的默认模型。

**类型：** 动手实现
**语言：** Python
**前置知识：** Phase 6 · 02（频谱图 & Mel），Phase 5 · 09（Seq2Seq），Phase 7 · 05（完整 Transformer）
**时长：** 约75分钟

## 问题

你有一个字符串："请在下午6点提醒我浇花。"你需要一个听起来自然、语调正确（停顿、重音）、正确发音"plants"、在 CPU 上实时语音助手300毫秒内运行的3秒音频片段。你还需要换声音、处理代码转换输入（"remind me at 6 pm, daijoubu?"），以及不在名字上出丑。

现代 TTS 流水线看起来像：

1. **文本前端。** 规范化文本（日期、数字、邮箱），转换为音素或子词 token，预测韵律特征。
2. **声学模型。** 文本 → mel 频谱图。Tacotron 2（2017）、FastSpeech 2（2020）、VITS（2021）、F5-TTS（2024）、Kokoro（2024）。
3. **声码器。** Mel → 波形。WaveNet（2016）、WaveRNN、HiFi-GAN（2020）、BigVGAN（2022）、2024+ 的神经编解码器声码器。

到2026年，声学 + 声码器分叉与端到端扩散和流匹配模型模糊。但三部分的思维模型对调试仍有效。

## 核心概念

**Tacotron 2（2017）。** Seq2seq：char 嵌入 → BiLSTM encoder → 位置敏感注意力 → 自回归 LSTM decoder 发出 mel 帧。慢（AR），长文本上不稳定。仍被引用为基线。

**FastSpeech 2（2020）。** 非自回归。持续预测器输出每个音素得到多少 mel 帧。1次通过，比 Tacotron 快10×。失去一些自然度（单调对齐）但广泛发货。

**VITS（2021）。** 端到端联合训练 encoder + 基于流的持续 + HiFi-GAN 声码器，带变分推理。高质量，单一模型。主导开源 TTS 2022–2024。变体：YourTTS（多说话人零样本）、XTTS v2（2024，Coqui）。

**F5-TTS（2024）。** 流匹配上的扩散 Transformer。自然韵律，5秒参考音频零样本声音克隆。2026年开源 TTS 排行榜顶部。335M 参数。

**Kokoro（2024）。** 小型（82M），CPU 可跑，实时使用的英语 TTS 最佳。封闭词汇仅英语，apache-2.0。

**OpenAI TTS-1-HD，ElevenLabs v2.5，Google Chirp-3。** 商业前沿。ElevenLabs v2.5 情感标签（"[whispered]"、"[laughing]"）和角色声音主导2026年有声书生产。

### 声码器演进

| 时代 | 声码器 | 延迟 | 质量 |
|-----|---------|---------|---------|
| 2016 | WaveNet | 仅离线 | 发布时 SOTA |
| 2018 | WaveRNN | ~实时 | 好 |
| 2020 | HiFi-GAN | 100×实时 | 接近人类 |
| 2022 | BigVGAN | 50×实时 | 跨说话人/语言泛化 |
| 2024 | SNAC、DAC（神经编解码器） | 与 AR 模型集成 | 离散 token，高比特效率 |

到2026年大多数"TTS"模型都是端到端从文本到波形；mel 频谱图是内部表示。

### 评估

- **MOS（平均意见分）。** 1–5 分制，众包。仍是黄金标准；慢得痛苦。
- **CMOS（比较 MOS）。** A vs B 偏好。每个标注置信区间更紧。
- **UTMOS、DNSMOS。** 无参考神经 MOS 预测器。用于排行榜。
- **通过 ASR 的 CER（字符错误率）。** 将 TTS 输出通过 Whisper，计算相对输入文本的 CER。可懂性的代理。
- **SECS（说话人嵌入余弦相似度）。** 声音克隆质量。

2026年 LibriTTS test-clean 数字：

| 模型 | UTMOS | CER（通过 Whisper） | 规模 |
|-------|-------|-------------------|------|
| Ground truth | 4.08 | 1.2% | — |
| F5-TTS | 3.95 | 2.1% | 335M |
| XTTS v2 | 3.81 | 3.5% | 470M |
| VITS | 3.62 | 3.1% | 25M |
| Kokoro v0.19 | 3.87 | 1.8% | 82M |
| Parler-TTS Large | 3.76 | 2.8% | 2.3B |

## 动手实现

### 步骤 1：音素化输入

```python
from phonemizer import phonemize
ph = phonemize("Hello world", language="en-us", backend="espeak")
# 'həloʊ wɜːld'
```

音素是通用桥梁。不要向低于 VITS 质量的任何东西输入原始文本。

### 步骤 2：跑 Kokoro（2026 CPU 默认）

```python
from kokoro import KPipeline
tts = KPipeline(lang_code="a")  # "a" = American English
audio, sr = tts("Please remind me to water the plants at 6 pm.", voice="af_bella")
# audio: float32 tensor, sr=24000
```

离线运行，单文件，82M 参数。

### 步骤 3：用声音克隆跑 F5-TTS

```python
from f5_tts.api import F5TTS
tts = F5TTS()
wav = tts.infer(
    ref_file="my_voice_5s.wav",
    ref_text="The quick brown fox jumps over the lazy dog.",
    gen_text="Please remind me to water the plants.",
)
```

传5秒参考片段 + 其转录；F5 克隆韵律和音色。

### 步骤 4：HiFi-GAN 声码器

太大不适合教程脚本，但形状是：

```python
class HiFiGAN(nn.Module):
    def __init__(self, mel_channels=80, upsample_rates=[8, 8, 2, 2]):
        super().__init__()
        ...
    def forward(self, mel):
        return self.blocks(mel)  # -> waveform
```

## 陷阱

- **无文本归一器。** "Dr. Smith" 读作"Doctor"还是"Drive"？"2026"读作"二十零二十六"还是"二洞洞二六"？在音素化器之前规范化。
- **OOV 专有名词。** 总是备一个字素到音素模型用于未知 token。
- **裁剪。** 声码器输出很少裁剪，但 mel 缩放不匹配会在推理时超出 ±1.0。总是 `np.clip(wav, -1, 1)`。
- **采样率不匹配。** Kokoro 输出 24kHz；下游流水线期望 16kHz → 重采样或产生混叠。

## 产出

保存为 `outputs/skill-tts-designer.md`。为给定声音、延迟和语言目标设计 TTS 流水线。

## 练习

1. **简单。** 运行 `code/main.py`。从玩具词汇表构建音素字典，估计每个音素的持续时间，打印假的"mel"调度。
2. **中等。** 安装 Kokoro，在 voice `af_bella` 和 `am_adam` 上合成相同句子。比较音频持续时间和主观质量。
3. **困难。** 录一段5秒自己的参考片段。用 F5-TTS 克隆。报告参考和克隆输出间的 SECS。

## 关键术语

| 术语 | 行话 | 实际含义 |
|------|-----------------|-----------------------|
| Phoneme | 音素 | 声音单位；英语中39个（ARPABET）。 |
| Duration predictor | 每个音素持续多久 | 非 AR 模型输出；每个音素的整数帧。 |
| Vocoder | Mel → 波形 | 将 mel-spec 映射到原始样本的神经网络。 |
| HiFi-GAN | 标准声码器 | 基于 GAN；主导2020–2024。 |
| MOS | 主观质量 | 来自人类评分员的 1–5 平均意见分。 |
| SECS | 声音克隆指标 | 目标和输出说话人嵌入间的余弦相似度。 |
| F5-TTS | 2026 开源 SOTA | 流匹配扩散；零样本克隆。 |
| Kokoro | CPU 英语领先 | 82M 参数模型，Apache 2.0。 |

## 扩展阅读

- [Shen et al. (2017). Tacotron 2](https://arxiv.org/abs/1712.05884) — seq2seq 基线。
- [Kim, Kong, Son (2021). VITS](https://arxiv.org/abs/2106.06103) — 端到端基于流的。
- [Chen et al. (2024). F5-TTS](https://arxiv.org/abs/2410.06885) — 当前开源 SOTA。
- [Kong, Kim, Bae (2020). HiFi-GAN](https://arxiv.org/abs/2010.05646) — 仍在2026年发货的声码器。
- [Kokoro-82M on HuggingFace](https://huggingface.co/hexgrad/Kokoro-82M) — 2024年 CPU 友好英语 TTS。