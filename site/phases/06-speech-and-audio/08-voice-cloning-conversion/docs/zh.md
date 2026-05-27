# 声音克隆与声音转换

> 声音克隆用别人的声音读你的文本。声音转换在保留你说了什么的同时把你的声音改写成别人的声音。两者都依赖于同一原语：从内容中分离说话人身份。

**类型：** 动手实现
**语言：** Python
**前置知识：** Phase 6 · 06（说话人识别），Phase 6 · 07（TTS）
**时长：** 约75分钟

## 问题

到2026年，5秒音频片段足以在消费者 GPU 上产生任何人的高质量声音克隆。ElevenLabs、F5-TTS、OpenVoice v2、VoiceBox 都发货零样本或少样本克隆。这项技术是祝福（无障碍 TTS、配音、辅助声音）也是武器（诈骗电话、政治深度伪造、IP 盗窃）。

两个密切相关任务：

- **声音克隆（TTS 侧）：** 文本 + 5秒参考声音 → 该声音的音频。
- **声音转换（语音侧）：** 源音频（A 人的 X）+ B 的参考声音 → B 说的 X 的音频。

两者都将波形分解为（内容，说话人，韵律）并重新组合来自一个来源的内容与来自另一个的说话人。

2026年你发货的关键约束：**水印和同意门控在欧盟（AI Act，2026年8月可执行）和加州（AB 2905，2025年生效）法律上必须。** 流水线必须发出听不见的水印并拒绝非同意克隆。

## 核心概念

**零样本克隆。** 将5秒片段传给在数千说话人上训练过的模型。说话人 encoder 将片段映射到说话人嵌入；TTS decoder 条件于该嵌入加文本。

**少样本微调。** 录制目标声音5-30分钟。对基座 LoRA 微调一小时。质量从"还行"飞跃到"无法区分"。

**声音转换（VC）。** 两大家族：

- **识别-合成。** 运行类 ASR 模型提取内容表示（如软音素后验概率、PPG），然后用目标说话人嵌入重新合成。鲁棒于语言和口音。
- **解耦。** 训练一个在瓶颈处分离内容、说话人、韵律的自编码器。在推理时交换说话人嵌入。

**基于神经编解码的克隆（2024+）。** VALL-E、VALL-E 2、NaturalSpeech 3、VoiceBox — 将音频视为 SoundStream / EnCodec 的离散 token，在编解码 token 上训练大型自回归或流匹配模型。

## 动手实现

### 步骤 1：零样本克隆

```python
from f5_tts.api import F5TTS
tts = F5TTS()
wav = tts.infer(
    ref_file="target_voice_5s.wav",
    ref_text="The quick brown fox jumps over the lazy dog.",
    gen_text="请在下午6点提醒我浇花。",
)
```

### 步骤 2：说话人验证（克隆质量）

```python
from speechbrain.pretrained import EncoderClassifier
spk_model = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb")
ref_emb = spk_model.encode_batch(ref_audio)
gen_emb = spk_model.encode_batch(cloned_audio)
secs = cosine_sim(ref_emb, gen_emb)  # >0.75 识别克隆
```

### 步骤 3：水印

```python
# AudioSeal 或 WavMark 将不可感知 ID 嵌入音频
from audioseal import AudioSeal
wm = AudioSeal()
watermarked = wm.watermark(audio, message="consent_id_12345")
```

## 陷阱

- **无同意记录。** 每个克隆输出必须附同意记录。
- **水印不鲁棒。** 验证抗 MP3/AAC 压缩、EQ、速度变换的水印。
- **身份欺骗。** 用 ASV 验证说话人身份与同意记录匹配。

## 产出

保存为 `outputs/skill-voice-cloning-architect.md`。为给定场景设计克隆或转换流水线，含同意门控和水印。

## 关键术语

| 术语 | 实际含义 |
|------|---|
| Zero-shot cloning | 5秒参考音频 + 文本 → 克隆声音 |
| Voice conversion | 将 A 的声音内容转换为 B 的声音 |
| Speaker embedding | 说话人的向量表示，用于条件 TTS |
| SECS | 克隆与参考的声音嵌入余弦相似度 |
| Consent gate | 每个克隆必须附同意记录的法律要求 |
| AudioSeal | Meta 2024 水印，16-32 bit ID，听感不可觉察 |
| ASVspoof 检测 | 检测合成语音的分类器 |