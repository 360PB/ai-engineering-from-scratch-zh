# 语音合成（TTS）

> ASR 把声音变成文字。TTS 把文字变成声音。管道完全相同，只是方向相反——编码器、解码器、自回归，但这次输出的是音频帧而非文本 token。

**类型：** 构建
**语言：** Python
**前置知识：** Phase 6 · 02（频谱图与梅尔）、Phase 6 · 05（Whisper 架构）
**时间：** 约 60 分钟

## 问题

你有一段文本。你想要一段匹配的音频：正确的词、正确的韵律、正确的说话人声音。2026 年的挑战不是让机器说话——而是让机器听起来不像机器。

三个层次的 TTS：

1. **级联 TTS。** 文本 → 音素 → 频谱图 → 波形。多个模型串联。
2. **端到端神经 TTS。** 文本 → 梅尔频谱图（或直接到离散 token） → 波形。一个模型。VALL-E、FastSpeech 2、YourTTS。
3. **扩散/自回归 LM over tokens。** 文本 → 语义 token → acoustic token → 波形。VALL-E 3、Moshi、Orpheus。

2026 年：声音克隆零镜头达到人类水平。Kokoro-v1（MIT）在 82 种语言上以 24 kHz 运行。VALL-E 2（微软）在 LibriSpeech 上达到 2.3% WER 语音质量。问题不再是"它能说话吗"，而是"它听起来像谁"。

## 概念

![TTS 三层：文本分析、mel/离散合成、声码器](../assets/tts-architecture.svg)

### 三种架构

**级联（2019 年前）。** 文本 → 文本分析（Grapheme-to-Phoneme，G2P）→ 韵律预测 → 梅尔频谱图生成器（如 Tacotron 2）→ 声码器（如 WaveRNN）。高质量但多阶段，每阶段独立训练。

**并行端到端（2019–2023）。** FastSpeech 2、Transformer-TTS：文本直接 → 梅尔，无自回归，用时长预测器控制语速。快（可并行），但韵律平淡。

**自回归 LM over discrete tokens（2023+）。** VALL-E 路线：文本 → 编码器 → 离散音频 token（来自神经声码器，如 EnCodec）→ 自回归 LM → 声码器解码。零镜头声音克隆：给定 3 秒参考音频，LM 在 token 空间生成听起来像该说话人的语音。

### 声码器：梅尔/离散 → 波形

最后一个阶段。将频谱图或 token 序列变回波形：

| 声码器 | 速度 | 质量 | 用于 |
|--------|------|------|------|
| WaveRNN | 慢（实时 ~0.1×） | 很高 | 旧 TTS |
| HiFi-GAN | 快（10–50× 实时） | 高 | 大多数现代 TTS |
| BigVGAN | 快（边缘优化） | 很高 | 设备端 |
| DAC 解码器 | 快 | 很高 | EnCodec 风格 |
| VALL-E 声码器 | 中 | 很高 | VALL-E 系列 |

### 2026 年模型图

| 模型 | 声音克隆 | 语言 | 速度 | 许可 |
|------|---------|------|------|------|
| Kokoro-v1 | 少量样本 | 82 | 24 kHz | MIT |
| XTTS-v2 | 零镜头 | 17 | 24 kHz | 商业 |
| VALL-E 2 | 零镜头 | 英语 | 16 kHz | 研究 |
| F5-TTS | 零镜头 | 中英 | 24 kHz | 开源 |
| Orpheus-3B | 零镜头 | 英语 | 24 kHz | Apache-2.0 |
| ElevenLabs v3 | 零镜头 | 29 | 44 kHz | 商业 |

### 评估

MOS（平均意见分）：人类评分员 1–5 分打语音质量。2026 年优质 TTS > 4.0。SECS（说话人相似度）：克隆声音与原始说话人的相似度。

## 构建

### 步骤 1：文本到 phoneme（G2P）

```python
from phonetics import g2p  # 或使用 `gruut` / `phonemizer`

text = "Hello world"
phonemes = g2p(text, lang="en-us")
# ['HH AH L OW', 'W ER L D']
```

级联系统的第一步。

### 步骤 2：梅尔频谱图生成（简化Tacotron风格）

```python
def mel_from_text(text, encoder, decoder, durations):
    phonemes = g2p(text)
    phoneme_ids = [vocab[p] for p in phonemes]
    phoneme_emb = encoder(torch.tensor(phoneme_ids))
    # 将 duration 扩展到帧长度
    expanded = phoneme_emb.repeat_interleave(
        torch.tensor(durations), dim=0
    )
    mel = decoder(expanded)
    return mel  # (n_frames, n_mels)
```

简化的两步。真实 Tacotron 使用注意力对齐机制自动学习 durations。

### 步骤 3：HiFi-GAN 声码器

```python
import soundfile as sf
from зв镇.hifigan import Hifigan

vocoder = Hifigan("checkpoints/hifigan")
mel = mel_from_text(text, encoder, decoder, durations)
waveform = vocoder(mel)  # (1, T)
sf.write("output.wav", waveform[0].numpy(), 24000)
```

步骤 3 声码器是独立模块。你可以替换任意梅尔生成器与任意声码器。

### 步骤 4：用 Coqui TTS 做零镜头声音克隆

```python
from TTS.api import TTS

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
# 克隆：提供参考音频
tts.tts_with_vc(
    text="This is how I sound.",
    speaker_wav="my_voice_3sec.wav",
    language="en"
)
```

3 秒参考。17 种语言。商业许可。

## 使用

2026 年栈：

| 场景 | 选择 |
|------|------|
| 开源、速度优先 | Kokoro-v1 或 F5-TTS |
| 商业质量 | ElevenLabs v3 或 XTTS-v2 |
| 边缘、实时 | BigVGAN + 小型 TTS 模型 |
| 多语言、研究 | VALL-E 2 或 YourTTS |
| 零镜头声音克隆 | XTTS-v2 或 Orpheus-3B |

决策规则：**声音克隆 > 级联 > 慢**。如果你需要保留说话人身份，零镜头克隆是唯一出路。如果你只需要"一个人说话"，F5-TTS 开源版本足够。

## 坑

- **机械韵律。** 级联 TTS 在语调上听起来像机器人。并行模型需要显式韵律控制。
- **长文本分段。** TTS 模型有最大输入长度。超过时需要分段并平滑拼接。
- **音频伪影在拼接处。** 段落边界声码器输出不连续。用 crossfade 或 overlap-add 修复。
- **声码器选择影响质量。** HiFi-GAN 在音乐上不如在语音上好；反之亦然。
- **声音克隆的同意问题。** 克隆某人声音需要明确同意——这是 2025–2026 年法律和平台的焦点。

## 发货

保存为 `outputs/skill-tts-picker.md`。为给定应用选择 TTS 系统、声码器和声音克隆策略。

## 练习

1. **简单。** 运行 `code/main.py`。它用 HiFi-GAN 声码器合成一个简化的梅尔频谱图。聆听并注意声码器伪影。
2. **中等。** 安装 `TTS`（Coqui）。用你自己的声音文件（3 秒）克隆。在不同文本上测试。报告语音相似度的主观印象。
3. **困难。** 用 YourTTS 或 VALL-E 风格的设置：预训练 TTS + 少量说话人适应数据。微调声码器在一个新说话人上（1 分钟录音）。评估 SECS（说话人相似度）和 WER（以评估可懂度）。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------|--------|
| G2P | 文本到音素 | Grapheme-to-Phoneme；将拼写转写为语音表示。 |
| 声码器 | 波形合成器 | 将频谱图或 token 序列转换为波形的神经网络。 |
| HiFi-GAN | 声码器 | 2020 年高保真 GAN 声码器；快速且质量高。 |
| 零镜头声音克隆 | 少量样本克隆 | 给定几秒参考音频，无需微调即可克隆声音。 |
| MOS | 语音质量评分 | 人类评 1–5 分；>4.0 为高质量。 |
| SECS | 说话人相似度 | 克隆声音与原始说话人的相似程度。 |
| VALL-E | 神经声码器 TTS | 2023 年；离散 token + 自回归 LM；零镜头。 |

## 延伸阅读

- [Shen et al. (2018). Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions](https://arxiv.org/abs/1712.05884) — Tacotron 2，经典级联 TTS。
- [Ren et al. (2021). FastSpeech 2: Fast and High-Quality End-to-End Text-to-Speech](https://arxiv.org/abs/2006.04558) — 并行端到端。
- [Wang et al. (2023). VALL-E](https://arxiv.org/abs/2301.02111) — 零镜头声音克隆。
- [Kokoro TTS repo](https://github.com/hexgrad/Kokoro) — 82 种语言，MIT 许可。
- [F5-TTS](https://github.com/SWivid/F5-TTS) — 开源零镜头 TTS，中英双语。
- [Kim et al. (2024). BigVGAN: GAN-based Neural Vocoder for Speech Synthesis at Scale](https://arxiv.org/abs/2309.02736) — 边缘优化声码器。