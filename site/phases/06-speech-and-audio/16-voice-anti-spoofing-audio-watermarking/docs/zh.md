# 语音反欺骗与音频水印

> 语音克隆和 TTS 技术的进步意味着任何音频都可能是假的。反欺骗检测和水印是应对这一挑战的两条路。两者在 2025–2026 年从研究走向生产。

**类型：** 学习
**语言：** Python
**前置知识：** Phase 6 · 04（ASR）、Phase 6 · 06（说话人验证）、Phase 6 · 07（TTS）
**时间：** 约 45 分钟

## 问题

2024 年：有人用 AI 克隆 CEO 的声音，要求员工转账 €220,000。2025 年：深度伪造语音在选举期间传播。2026 年：大多数反欺骗系统已将误报率从 15% 降至 2% 以下，但合成音频质量继续提高。

两个相关问题：

1. **检测合成音频。** 给定一个音频片段，判断它是否由 TTS/VC 系统生成。用于身份验证、金融欺诈检测、内容审核。
2. **音频水印。** 在音频中嵌入不可听见的标记，以便以后验证来源。用于内容出处、版权保护、AI 使用披露。

两者都需要：检测需要特征提取 + 分类器；水印需要修改音频而不影响感知质量。

## 概念

### 反欺骗检测

**信号特征。** 合成音频有特定的伪影：

- **频谱不一致。** 真实语音在某些频率有自然谐波。合成音频在相同位置可能有平坦或异常响应。
- **相位异常。** 人类发声产生特定相位模式。某些 TTS 系统在相位处理上有差异。
- **动态范围。** 合成音频通常有更窄的动态范围——听起来"压缩"。
- **高频内容缺失或异常。** 某些编解码器在高频频段引入伪影。

**检测模型。** 2026 年基线：AASIST（RawNet2 基础）。使用原始波形而非频谱图作为输入，避免特征工程。

```
音频 → RawNet2 编码器 → 图注意力 → 分类 → 真实/合成
```

| 模型 | EER（等错误率）| 备注 |
|------|--------------|------|
| RawNet2 | 1.7% | 2019 年基线；仍在使用 |
| AASIST | 0.83% | 2022 年 SOTA |
| AASIST2 | 0.42% | 2024 年改进 |
| L3-Net | 0.31% | 2026 年当前 SOTA |

### 水印

**两种方法：**

1. **频谱水印。** 在频谱图的特定频率容器中嵌入信号。难以检测但可能影响音频质量。
2. **统计水印。** 修改音频的统计特性（如相位直方图）。对感知影响最小。

**稳健性要求：**
- 对音频压缩（MP3、Opus）稳健
- 对重采样稳健
- 对时间拉伸稳健
- 对噪声叠加稳健

```
原始音频 → 水印嵌入器 → 标记音频（感知相同）→ 压缩/传输 → 水印检测器 → 验证来源
```

**EU AI Act + 加州 SB 942** 要求 AI 生成音频披露。水印是一种合规机制。

### 2026 年生产栈

| 任务 | 方法 | 工具 |
|------|------|------|
| 重放攻击检测 | 能量 + 频谱特征 | Cobra VAD + 自定义分类器 |
| TTS/VC 合成检测 | AASIST / RawNet3 | `torch` 模型 |
| 深度伪造语音检测 | 说话人验证 + 时序分析 | pyannote + 自定义头 |
| 音频水印 | 频谱或统计方法 | Google SynthID、AudioSeal |
| 内容出处 | C2PA 元数据 | C2PA 标准 |

## 构建

### 步骤 1：时域特征（重放检测）

```python
def extract_temporal_features(audio, sr):
    # 短时能量
    frames = frame_signal(audio, 400, 160)
    energy = [sum(x*x for x in f) / len(f) for f in frames]

    # 过零率
    zcr = [sum(abs(a - b) for a, b in zip(f[:-1], f[1:])) / (2 * len(f)) for f in frames]

    # 频谱通量（相邻帧之间的变化）
    spec = stft(audio, 400, 160, n_fft=512)
    flux = [sum((spec[t][f] - spec[t-1][f])**2 for f in range(len(spec[0])))
            for t in range(1, len(spec))]

    return {"energy": energy, "zcr": zcr, "flux": flux}
```

### 步骤 2：频谱特征

```python
def extract_spectral_features(audio, sr):
    mel_spec = mel_spectrogram(audio, sr, n_mels=80)

    # 梅尔频谱统计
    mean_mel = [sum(frame) / len(frame) for frame in mel_spec]
    var_mel = [sum((f - m)**2 for f in frame) / len(frame)
               for frame, m in zip(mel_spec, mean_mel)]

    # 谐波比
    # ...

    return {"mean_mel": mean_mel, "var_mel": var_mel}
```

### 步骤 3：AASIST 风格检测器

```python
# 使用预训练 AASIST
from speechbrain.pretrained import EncoderClassifier

anti_spoof = EncoderClassifier.from_hparams(
    source="speechbrain/anti-spoofing-nn3-front-nemo",
    savedir="pretrained_models/anti-spoof"
)

score = anti_spoof.encode_batch(torch.tensor(audio)).item()
verdict = "spoofed" if score < 0.5 else "bona_fide"
```

### 步骤 4：水印嵌入（简化）

```python
def embed_watermark(audio, watermark_bits):
    """在频谱图的特定高频容器中嵌入水印。"""
    spec = stft(audio, 400, 160, n_fft=512)
    n_frames = len(spec)
    n_bins = len(spec[0])

    # 在高频区域嵌入（在奈奎斯特以下但在大多数语音频率以上）
    bit_idx = 0
    for frame in range(0, n_frames, 8):  # 每 8 帧嵌入 1 bit
        if bit_idx >= len(watermark_bits):
            break
        # 在 bin 200-250 范围内嵌入（语音上方但低于 Nyquist）
        bin_idx = 200 + (watermark_bits[bit_idx] * 10)
        spec[frame][bin_idx] += 0.1  # 轻微增加该频率的能量
        bit_idx += 1

    return istft(spec, 400, 160)
```

### 步骤 5：水印检测

```python
def detect_watermark(audio, n_bits=100):
    """从频谱图中提取水印。"""
    spec = stft(audio, 400, 160, n_fft=512)
    bits = []
    for frame in range(0, min(len(spec), n_bits * 8), 8):
        bin_idx = 200
        energy = sum(spec[frame][bin_idx:bin_idx+10])
        bits.append(1 if energy > 0.05 else 0)
    return bits
```

## 使用

| 场景 | 选择 |
|------|------|
| 实时反欺骗 | AASIST2 或自定义 RawNet |
| 水印嵌入 | Google SynthID 或 AudioSeal |
| 内容出处 | C2PA 元数据标准 |
| 重放攻击检测 | 能量 + Cobra VAD |
| 深度伪造检测 | AASIST + 说话人验证 |

## 坑

- **检测 vs 水印的权衡。** 检测是对抗性的（水印是对抗性的）。水印在设计时需要考虑可能的攻击（压缩、噪声、时间拉伸）。
- **误报率。** 在金融场景中，误报（将真实音频标记为合成）可能是灾难性的。在内容审核中，误报可能意味着合法的深度讽刺被删除。
- **零样本检测。** 检测器在训练时见过的 TTS 系统上表现好。对未见过的系统泛化仍然是一个开放问题。
- **水印鲁棒性。** 强压缩（MP3 128 kbps）可以去除某些水印。需要嵌入足够强的信号以抵抗压缩。
- **合法使用。** 水印必须不影响正常的音频处理（编辑、混合、格式转换）。

## 发货

保存为 `outputs/skill-anti-spoof-designer.md`。设计一个音频安全和出处系统，结合反欺骗检测和水印。

## 练习

1. **简单。** 运行 `code/main.py`。从合成音频（TTS）和真实录音中提取特征。比较它们的差异。
2. **中等。** 安装 AASIST 或使用 SpeechBrain 的反欺骗模型。在合成和真实音频的混合数据集上评估 EER。
3. **困难。** 实现一个鲁棒的水印系统：对 MP3 压缩、重采样和噪声添加稳健。用 `scipy.io.wavfile` 读取和写入带水印的音频并验证检测率。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------|--------|
| 反欺骗 | 深度伪造检测 | 区分真实音频和合成/转换音频。 |
| 重放攻击 | 预录攻击 | 使用录制的真实音频冒充实时说话人。 |
| AASIST | 反欺骗检测器 | 2022 年 SOTA；基于图注意力。 |
| 水印 | 不可听见的标记 | 在音频中嵌入信号用于来源验证。 |
| C2PA | 内容出处标准 | 嵌入式元数据标准，用于 AI 生成内容。 |
| EER | 等错误率 | 假阳性率 = 假阴性率的点；反欺骗的标准指标。 |

## 延伸阅读

- [Khalid et al. (2022). AASIST: Audio Anti-Spoofing Using Integrated Spectrogram and Copied Temporal Convolutional Network](https://arxiv.org/abs/2202.04320) — 当前反欺骗检测 SOTA 基础。
- [Jung et al. (2022). Improved RawNet2](https://arxiv.org/abs/2202.04158) — RawNet3 的前身。
- [Zhao et al. (2024). AudioSeal:首个音频水印大模型](https://arxiv.org/abs/2407.08540) — Meta 的音频水印模型。
- [Google SynthID — Audio](https://deepmind.google/technologies/synthid/) — Google 的音频水印技术。
- [C2PA standard](https://c2pa.org/) — 内容出处和真实性联盟标准。
- [ASVspoof 2024 challenge](https://www.asvspoof.org/) — 反欺骗评估的标准基准。