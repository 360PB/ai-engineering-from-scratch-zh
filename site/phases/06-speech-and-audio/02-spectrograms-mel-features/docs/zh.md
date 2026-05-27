# 频谱图、Mel 尺度与音频特征

> 神经网络不能很好地消费原始波形。它们更擅长消费频谱图。更准确地消费 mel 频谱图。每个 2026 年的 ASR、TTS 和音频分类器都因这个单一预处理选择而兴衰。

**类型：** 动手实现
**语言：** Python
**前置知识：** Phase 6 · 01（音频基础）
**时长：** 约45分钟

## 问题

取一个10秒16kHz片段。那是160,000个浮点数，都在 `[-1, 1]` 中，几乎与标签"狗叫"或"单词 cat"完全不相关。原始波形有信息，但以模型无法轻易提取的形式存在。两个相同音素在100毫秒前后说出来有完全不同的原始样本。

频谱图修复了这一点。它在人类感知忽略的时序细节（微秒抖动）处折叠，并在感知attend的结构处保留（哪些频率有能量，在约10–25ms的时间窗口上）。

Mel 频谱图更进一步。人类感知音高是对数的：100 Hz vs 200 Hz 听起来"距离相同"如同 1000 Hz vs 2000 Hz。Mel 尺度弯曲频率轴以匹配感知。Mel 尺度频谱图是2010年到2026年语音 ML 中最重要的单一特征。

## 核心概念

**STFT（短时傅里叶变换）。** 将波形切成重叠帧（典型：25ms窗口，10ms跳 = 16000Hz 下400样本 / 160样本）。每帧乘以窗口函数（Hann 是默认值；Hamming 权衡略有不同）。对每帧做 FFT。将幅度频谱堆叠成形状 `(n_frames, n_freq_bins)` 的矩阵。那就是你的频谱图。

**对数幅度。** 原始幅度跨越5-6个数量级。取 `log(|X| + 1e-6)` 或 `20 * log10(|X|)` 压缩动态范围。每个生产流水线都用对数幅度，不用原始幅度。

**Mel 尺度。** 频率 `f` Hz 映射到 mel `m`：`m = 2595 * log10(1 + f / 700)`。映射在1kHz 以下大致线性，在以上大致对数。覆盖 0–8kHz 的80个 mel bins 是标准 ASR 输入。

**Mel 滤波器组。** 一组在 mel 尺度上等距间隔的三角滤波器。每个滤波器是相邻 FFT bins 的加权和。将 STFT 幅度乘以滤波器组矩阵得到 mel 频谱图（一次 matmul）。

**对数 mel 频谱图。** `log(mel_spec + 1e-10)`。Whisper 的输入。Parakeets 的输入。SeamlessM4T 的输入。通用的2026年音频前端。

**MFCC。** 取对数 mel 频谱图，应用 DCT（II型），保留前13个系数。去相关特征并进一步压缩。直到2015年主导特征，当时 CNN/Transformer 在原始 log-mels 上赶上。仍在说话人识别中使用（x-vectors、ECAPA）。

**分辨率权衡。** 更大 FFT = 更好频率分辨率但更差时序分辨率。25ms / 10ms 是音频-ML 默认；音乐用50ms / 12.5ms；瞬态检测（鼓点、爆破音）用5ms / 2ms。

## 动手实现

### 步骤 1：分帧波形

```python
def frame(signal, frame_len, hop):
    n = 1 + (len(signal) - frame_len) // hop
    return [signal[i * hop : i * hop + frame_len] for i in range(n)]
```

10秒16kHz clip，frame_len=400, hop=160，产出998帧。

### 步骤 2：Hann 窗口

```python
import math

def hann(N):
    return [0.5 * (1 - math.cos(2 * math.pi * n / (N - 1))) for n in range(N)]
```

在 FFT 前逐元素相乘。消除在非零端点截断引起的光谱泄漏。

### 步骤 3：STFT 幅度

```python
def stft_magnitude(signal, frame_len=400, hop=160):
    win = hann(frame_len)
    frames = frame(signal, frame_len, hop)
    return [magnitudes(dft([w * s for w, s in zip(win, f)])) for f in frames]
```

### 步骤 4：Mel 滤波器组

```python
def hz_to_mel(f):
    return 2595.0 * math.log10(1.0 + f / 700.0)

def mel_to_hz(m):
    return 700.0 * (10 ** (m / 2595.0) - 1.0)

def mel_filterbank(n_mels, n_fft, sr, fmin=0, fmax=None):
    fmax = fmax or sr / 2
    mels = [hz_to_mel(fmin) + (hz_to_mel(fmax) - hz_to_mel(fmin)) * i / (n_mels + 1)
            for i in range(n_mels + 2)]
    hzs = [mel_to_hz(m) for m in mels]
    bins = [int(h * n_fft / sr) for h in hzs]
    fb = [[0.0] * (n_fft // 2 + 1) for _ in range(n_mels)]
    for m in range(n_mels):
        for k in range(bins[m], bins[m + 1]):
            fb[m][k] = (k - bins[m]) / max(1, bins[m + 1] - bins[m])
        for k in range(bins[m + 1], bins[m + 2]):
            fb[m][k] = (bins[m + 2] - k) / max(1, bins[m + 2] - bins[m + 1])
    return fb
```

80 mels 覆盖 0–8kHz，n_fft=400，产出 `(80, 201)` 矩阵。

### 步骤 5：对数 mel

```python
def log_mel(mel_spec, eps=1e-10):
    return [[math.log(max(v, eps)) for v in frame] for frame in mel_spec]
```

### 步骤 6：MFCC

```python
def dct_ii(x, n_coeffs):
    N = len(x)
    return [
        sum(x[n] * math.cos(math.pi * k * (2 * n + 1) / (2 * N)) for n in range(N))
        for k in range(n_coeffs)
    ]
```

## 用现成库

2026年技术栈：

| 任务 | 特征 |
|------|----------|
| ASR（Whisper、Parakeet、SeamlessM4T） | 80 log-mels，10ms 跳，25ms 窗口 |
| TTS 声学模型（VITS、F5-TTS、Kokoro） | 80 mels，5–12ms 跳以精细时序控制 |
| 音频分类（AST、PANNs、BEATs） | 128 log-mels，10ms 跳 |
| 说话人嵌入（ECAPA-TDNN、WavLM） | 80 log-mels 或原始波形 SSL |
| 音乐（MusicGen、Stable Audio 2） | EnCodec 离散 token（不是 mels） |
| 关键词检测 | 40 MFCCs 用于小型设备 |

## 陷阱

- **Mel 数量不匹配。** 训练用80 mels，推理用128 mels。静默失败。在两端记录特征形状。
- **上游采样率不匹配。** 在22.05kHz 计算的 mels 看起来与16kHz 不同。在特征化*之前*修复 SR。
- **dB vs log。** Whisper 期望 log-mel，不是 dB-mel。
- **归一化漂移。** 训练时每话语归一化，推理时全局归一化。生产 bug，使 WER 加倍。
- **填充泄漏。** 在片段末尾补零产生平坦频谱。均匀填充或复制。

## 产出

保存为 `outputs/skill-feature-extractor.md`。该 skill 为给定模型目标选取特征类型、mel 数量、帧/跳和归一化。

## 练习

1. **简单。** 运行 `code/main.py`。综合一个啁啾（200 → 4000 Hz 频率扫描）并打印每帧的 argmax mel bin。绘制（可选）并确认匹配扫描。
2. **中等。** 用 `n_mels` ∈ `{40, 80, 128}` 和 `frame_len` ∈ `{200, 400, 800}` 重跑。测量时间轴上尖锐峰带宽。哪个组合解析啁啾最好？
3. **困难。** 实现 `power_to_db`，并用 (a) 原始 log-mel、(b) dB-mel 用 `ref=max`、(c) MFCC-13 + delta + delta-delta，在 AudioMNIST 上比较 tiny CNN 分类器的 ASR 准确率。

## 关键术语

| 术语 | 行话 | 实际含义 |
|------|-----------------|-----------------------|
| Frame | 一片 | 馈入一次 FFT 的 25ms 波形块。 |
| Hop | 步幅 | 连续帧之间的样本；10ms 是 ASR 默认。 |
| Window | 窗口 | 点态乘法器，将帧边缘渐变到零。 |
| STFT | 频谱图生成器 | 分帧 + 加窗 FFT；产出时间 × 频率矩阵。 |
| Mel | 弯曲频率 | 对数感知尺度；`m = 2595·log10(1 + f/700)`。 |
| Filterbank | 矩阵 | 将 STFT 投影到 mel bins 的三角滤波器。 |
| Log-mel | Whisper 的输入 | `log(mel_spec + eps)`；2026年标准化。 |
| MFCC | 老派特征 | log-mel 的 DCT；13系数，去相关。 |

## 扩展阅读

- [Davis, Mermelstein (1980). Comparison of parametric representations for monosyllabic word recognition](https://ieeexplore.ieee.org/document/1163420) — MFCC 论文。
- [Stevens, Volkmann, Newman (1937). A Scale for the Measurement of the Psychological Magnitude Pitch](https://pubs.aip.org/asa/jasa/article-abstract/8/3/185/735757/) — 原始 mel 尺度。
- [OpenAI — Whisper source, log_mel_spectrogram](https://github.com/openai/whisper/blob/main/whisper/audio.py) — 参考实现。
- [librosa feature extraction docs](https://librosa.org/doc/main/feature.html) — `mfcc`、`melspectrogram` 和 hop/窗口参考。
- [NVIDIA NeMo — audio preprocessing](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/asr_all.html#featurizers) — Parakeet + Canary 模型的生产规模流水线。