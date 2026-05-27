# 语音克隆与变声

> 声音克隆让你从几秒音频中重建一个说话人的声音。变声让你把一个声音的风格转换到另一个。两者都基于说话人嵌入或离散 token，但实现路径不同。

**类型：** 构建
**语言：** Python
**前置知识：** Phase 6 · 07（TTS）、Phase 5 · 22（嵌入模型）
**时间：** 约 45 分钟

## 问题

你有目标说话人 A 的几秒音频。你想让 B 的声音说出 A 的音色。或者：你有一段录音，想把它变成另一个人的声音而不改变内容。

两种不同的问题：

1. **语音克隆（Voice Clone）。** 在新文本上重现目标说话人的音色/韵律。零镜头 = 无需在目标说话人上微调。少量样本 = 1–5 分钟适应数据。
2. **变声（Voice Conversion）。** 保留源内容但改变音色/韵律到目标。"说这句话，但用那个人的声音"。

2026 年：XTTS-v2（Coqui）提供零镜头克隆，3 秒参考。VALL-E 3 在 2025 年将质量推向人类水平。变声模型如 FreeVC、RVC 在实时转换中广泛使用。

## 概念

### 语音克隆的三种方法

**说话人嵌入法。** 提取目标说话人的固定维度嵌入（来自 ECAPA-TDNN 或 WavLM），在合成时将其作为条件传入 TTS 模型。简单，但只捕获全局音色，不捕获韵律。

```
参考音频 → 说话人嵌入器 → 192d 嵌入 → TTS 条件 → 输出音频
```

**特征解耦法。** 将语音分解为内容（说什么）、韵律（怎么说）和说话人身份（谁在说）。分别建模后重新组合。YourTTS、VALL-E 3 使用此路线。

```
内容（文本/音素）+ 韵律（来自源）+ 说话人（来自目标）→ TTS
```

**离散 token 法（当前 SOTA）。** 语音被编码为离散 token（如 EnCodec）。说话人信息被编码为一个额外的 token 序列或条件向量。在 token 空间做克隆而非特征空间。

```
参考音频 → 声码器 encoder → 说话人 token 序列 → AR LM → 声码器 decoder → 输出音频
```

VALL-E 路线：给 AR LM 一个目标说话人的离散音频 token，它学习在这些 token 条件下生成。极强的零镜头能力。

### 变声架构

**并行方法（不需要参考音频时长匹配）。** CycleGAN-VC、StarGAN-VC：学习源和目标音色之间的映射，无需对齐。

**非并行方法（需要平行数据或外部对齐）。** FreeVC：在离散 token 空间做变声，解耦内容和说话人。RVC（Retrieval-Based Voice Conversion）：提取说话人嵌入，在推理时替换。

```
源音频 → 内容编码器 → 内容 token → 目标说话人嵌入 → 目标声音解码器 → 输出
```

### 2026 年模型

| 模型 | 类型 | 克隆方式 | 质量 | 许可 |
|------|------|---------|------|------|
| XTTS-v2 | TTS + 克隆 | 零镜头，3 秒 | 很高 | 商业 |
| VALL-E 3 | TTS + 克隆 | 零镜头，任意 | 人类水平 | 研究 |
| Coqui TTS | TTS + 克隆 | 少量样本 | 高 | 商业 |
| FreeVC | 变声 | 非并行 | 高 | 开源 |
| RVC | 变声 | 少量样本 | 中高 | 开源 |
| YourTTS | TTS + 克隆 | 零镜头 | 高 | 论文 |

## 构建

### 步骤 1：说话人嵌入提取

```python
from speechbrain.pretrained import EncoderClassifier

spk_encoder = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb"
)
embedding = spk_encoder.encode_batch(torch.tensor(audio_16k))
# shape: (1, 192) — ECAPA-TDNN 说话人嵌入
```

这是克隆管道的第一步。提取一次，重复使用。

### 步骤 2：说话人条件 TTS（YourTTS 风格）

```python
# 在 YourTTS 中，说话人嵌入在解码器中作为条件传入
def clone_voice(text, ref_audio, tts_model, spk_encoder):
    ref_emb = spk_encoder.encode_batch(torch.tensor(ref_audio))
    # 将说话人嵌入与文本嵌入连接
    mel = tts_model.forward(text_tokens, speaker_embedding=ref_emb)
    return vocoder(mel)
```

### 步骤 3：FreeVC 风格的变声

```python
# 源音频不变内容，改变说话人身份
def convert_voice(source_audio, target_ref, converter, vocoder):
    # 提取内容和说话人信息
    c_source = converter.encode_content(source_audio)  # 内容
    v_target = converter.encode_speaker(target_ref)    # 说话人身份

    # 重建：内容用目标说话人的音色
    converted_mel = converter.decode(c_source, v_target)
    return vocoder(converted_mel)
```

### 步骤 4：评估克隆质量

```python
from speechbrain.nn import params

# SECS（说话人嵌入余弦相似度）
def secs(emb1, emb2):
    return cosine(emb1, emb2)  # 越接近 1 越好

# WER（可懂度）
def evaluate_clone(clone_audio, reference_text):
    import whisper
    model = whisper.load_model("base")
    transcript = model.transcribe(clone_audio)["text"]
    return wer(reference_text, transcript)
```

## 使用

2026 年选择：

| 任务 | 选择 |
|------|------|
| 零镜头声音克隆 | XTTS-v2 或 VALL-E 3 |
| 少量样本适应 | Coqui TTS + 微调 |
| 实时变声 | RVC 或 FreeVC |
| 研究、开放 | YourTTS 或 VALL-E |
| 商业语音产品 | ElevenLabs + 自有克隆模型 |

## 坑

- **同意与身份。** 在 2025–2026 年，未经同意克隆某人声音在许多司法管辖区是违法或受监管的。
- **韵律泄露。** 零镜头克隆会保留参考音频的韵律风格。如果参考音频有强烈口音，克隆也会有。
- **变声的伪影。** 某些变声模型在高音调或快速语音处产生金属音。RVC 在快速语音上表现较好。
- **内容泄露。** 变声有时会轻微改变内容，特别是当说话人身份和内容高度纠缠时。

## 发货

保存为 `outputs/skill-voice-cloner.md`。为给定克隆/变声任务选择方法、模型和同意协议。

## 练习

1. **简单。** 运行 `code/main.py`。用两个不同的说话人嵌入驱动相同的 TTS 输出。注意音色差异。
2. **中等。** 使用 FreeVC 或 RVC 将你的声音转换为另一个说话人的音色。报告 SECS 和可懂度变化。
3. **困难。** 构建一个少量样本适应系统：用 1 分钟目标说话人数据微调 XTTS-v2 或 VALL-E。评估相对于零镜头基线的 SECS 提升。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------|--------|
| 语音克隆 | 零镜头说话人复制 | 给定几秒参考，重建目标说话人的声音。 |
| 变声 | 音色转换 | 改变声音的身份，保留内容。 |
| 说话人嵌入 | 身份向量 | 固定维度向量，编码说话人的音色特征。 |
| 零镜头 | 无适应 | 无需在目标说话人上微调即可克隆。 |
| 少量样本 | 少量适应 | 1–5 分钟目标说话人数据。 |
| SECS | 说话人相似度 | 两个声音嵌入之间的余弦相似度。 |

## 延伸阅读

- [Jia et al. (2023). VALL-E](https://arxiv.org/abs/2301.02111) — 零镜头语音克隆的突破。
- [Zhang et al. (2023). YourTTS](https://arxiv.org/abs/2110.08831) — 多说话人零镜头 TTS。
- [Li et al. (2023). FreeVC](https://arxiv.org/abs/2309.08331) — 高质量变声。
- [RVC repo](https://github.com/RVC-Project/Retrieval-Based-Voice-Conversion-WebUI) — 实时变声开源工具。
- [Coqui XTTS](https://github.com/coqui-ai/TTS) — 商业级零镜头克隆。