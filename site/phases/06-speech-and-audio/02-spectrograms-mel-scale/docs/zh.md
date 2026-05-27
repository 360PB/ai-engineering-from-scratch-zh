# 频谱图与梅尔尺度

> 音频在时间上变化。频谱图在频率上变化。梅尔尺度将 Hz 映射到人类听觉。每一个音频深度学习系统都在某个地方计算频谱图。

**类型：** 构建
**语言：** Python
**前置知识：** Phase 6 · 01（音频基础）、Phase 1 · 10（线性代数）
**时间：** 约 60 分钟

## 问题

一个 16 kHz 的 10 秒片段是 160,000 个浮点数。神经网络不能直接处理这些。它们需要一种表示，既压缩信号又保留相关信息。

频谱图将波形转换为"频率随时间变化"的图像。梅尔尺度进一步将 Hz 映射到人类感知——人类对低频比对高频敏感得多，梅尔频谱图将更多容器放在低频区域。

这是连接原始采样和深度学习的桥梁。每个音频任务——ASR、TTS、音乐生成、说话人识别——都在频谱图上运行（或隐式地通过使用频谱图作为中间表示的编解码器）。

## 概念

![从波形到梅尔频谱图的完整管道](../assets/spectrogram-mel.svg)

### 从波形到频谱图

```
波形 → 分帧（25ms，hop 10ms）→ 加窗（Hann）→ FFT → 幅度谱 → 对数压缩 → 频谱图
```

**分帧。** 将 160,000 个采样切成重叠的 25 ms 块（16 kHz 时 400 采样），相邻块间隔 10 ms（160 采样）。10 秒 → ~988 帧。

**加窗。** 每帧乘以 Hann 窗：`w[n] = 0.5 * (1 - cos(2πn/N))`。消除边缘不连续，否则 FFT 会在帧边界引入高频伪影。

**FFT。** 每帧 → 复数频谱。丢弃负频率（因为实信号是对称的），保留 N/2+1 个频率容器。对于 400 采样帧，25 ms 零填充到 512，FFT 给出 256 个频率容器（0–8 kHz）。

**幅度。** 取复数的模。你不关心相位，只关心每个频率的能量。

**对数。** 幅度范围从 0 到数千。取 `log(1 + amplitude)` 压缩动态范围，使特征更易于神经网络处理。

### 频谱图矩阵

结果是一个 `(n_frames, n_fft_bins)` 矩阵：
- 行 = 时间帧（约 988 行 @ 16 kHz，10 秒）
- 列 = 频率容器（128、256 或 512）
- 值 = 该时间该频率的能量（对数尺度）

可以绘制为热图——这就是你看到的"语音频谱图"：水平轴是时间，垂直轴是频率，亮度是能量。

### 梅尔尺度

人类对频率的感知是对数的。500 Hz 和 1000 Hz 之间的差异听起来比 5000 Hz 和 5500 Hz 之间的差异大。梅尔尺度近似感知：

```
m = 2595 * log10(1 + f / 700)
f = 700 * (10^(m/2595) - 1)
```

将 Hz 映射到梅尔。梅尔频谱图在低频有更多容器（更细的频率分辨率），在高频有更少（人类不敏感的区域）。

常见设置：

| n_mels | 频率分辨率 | 典型用途 |
|--------|-----------|---------|
| 40 | 低 | 语音识别、说话人识别 |
| 80 | 中 | Whisper 使用的 |
| 128 | 高 | 音乐生成、音频分类 |
| 256 | 很高 | 声码器、细粒度分析 |

**为什么是梅尔而非线性频谱图？** 语音的大部分信息在低频（共振峰在 300–3000 Hz）。梅尔尺度将更多特征桶分配到该区域。在相同容器数下，梅尔频谱图对语音任务有更多信息。

### 梅尔滤波器组

将线性频率容器映射到梅尔容器的加权矩阵。每个梅尔容器的响应是附近几个线性容器的加权和。滤波器形状：三角形或仿射。

```python
import numpy as np

def mel_filterbank(n_mels, n_fft, sr, fmin=0, fmax=None):
    if fmax is None:
        fmax = sr / 2
    # 将频率转换为梅尔
    def hz_to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)
    def mel_to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    # 在梅尔空间中均匀间隔
    mel_points = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_points = mel_to_hz(mel_points)  # 转换回 Hz

    # 构建三角滤波器
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(1, n_mels + 1):
        left = bin_points[i - 1]
        center = bin_points[i]
        right = bin_points[i + 1]
        for j in range(left, center):
            fb[i - 1, j] = (j - left) / (center - left)
        for j in range(center, right):
            fb[i - 1, j] = (right - j) / (right - center)
    return fb  # shape: (n_mels, n_fft // 2 + 1)
```

### 完整流程

```
波形 (160,000) → 分帧 → 加窗 → FFT(512) → 幅度 → 梅尔滤波器组(80) → log → log-mel
                   988帧        × Hann        256 bins   × 80          → (988, 80)
```

这就是 Whisper 看到的输入。

## 构建

### 步骤 1：分帧

```python
def frame_signal(signal, frame_size, hop):
    frames = []
    for start in range(0, len(signal) - frame_size + 1, hop):
        frames.append(signal[start:start + frame_size])
    return frames
```

`frame_size=400, hop=160` @ 16 kHz → 25 ms 帧，10 ms 步长。

### 步骤 2：Hann 窗

```python
def hann_window(n):
    return [0.5 * (1 - math.cos(2 * math.pi * i / (n - 1))) for i in range(n)]
```

### 步骤 3：STFT

```python
def stft(signal, frame_size, hop, n_fft=512):
    window = hann_window(frame_size)
    frames = frame_signal(signal, frame_size, hop)
    spectrogram = []
    for frame in frames:
        # 零填充到 n_fft
        padded = list(frame) + [0] * (n_fft - frame_size)
        windowed = [padded[i] * window[i] for i in range(n_fft)]
        # DFT（简化；生产用 numpy.fft.rfft）
        out = dft(windowed)  # 使用 Lesson 01 的 dft
        mag = [math.sqrt(r*r + i*i) for r, i in out[:n_fft//2+1]]
        spectrogram.append(mag)
    return spectrogram  # (n_frames, n_fft//2+1)
```

### 步骤 4：梅尔频谱图

```python
def mel_spectrogram(signal, sr, n_mels=80, frame_size=400, hop=160, n_fft=512):
    spec = stft(signal, frame_size, hop, n_fft)
    fb = mel_filterbank(n_mels, n_fft, sr)
    melspec = []
    for frame in spec:
        mel_frame = []
        for m in range(n_mels):
            val = sum(frame[f] * fb[m, f] for f in range(len(frame)))
            mel_frame.append(val)
        melspec.append(mel_frame)
    return melspec  # (n_frames, n_mels)
```

### 步骤 5：对数压缩

```python
def log_mel(melspec, floor=1e-5):
    return [[math.log(max(v, floor)) for v in frame] for frame in melspec]
```

这就是 Whisper 消费的东西。

## 使用

| 任务 | n_mels | 为什么 |
|------|--------|--------|
| ASR（Whisper） | 80 | 论文验证 |
| 说话人验证 | 40 | 足够；减少计算 |
| 音频分类 | 128 | 更多频率信息 |
| 音乐生成 | 128 | 乐器音色需要更高分辨率 |

**librosa 和 torchaudio 都实现了所有这些。** 从零构建是为了理解，而不是为了生产：

```python
import librosa
import numpy as np

# librosa：单行
mel_spec = librosa.feature.melspectrogram(
    y=audio,
    sr=16000,
    n_mels=80,
    n_fft=512,
    hop_length=160,  # 10 ms
)
log_mel = librosa.power_to_db(mel_spec)
```

## 坑

- **帧大小 vs 频率分辨率。** 更大的帧 → 更多的 FFT 容器 → 更好的频率分辨率，但更差的时间分辨率。语音用 25 ms（平衡）。敲击声检测可能需要 10 ms。
- **对数 vs 功率。** 频谱图有两种：幅度和功率。`melspectrogram` 默认返回功率。取对数时用 `librosa.power_to_db` 而非 `log`。
- **填充。** 短于帧大小的音频需要零填充或截断；始终处理整个信号。
- **标准化。** log-mel 值通常很大（负数，因为对数）。训练前减去均值除以标准差。

## 发货

保存为 `outputs/skill-mel-spec-designer.md`。为给定的音频任务选择 n_mels、帧大小、hop 和标准化策略。

## 练习

1. **简单。** 运行 `code/main.py`。从正弦波合成一个 440 Hz 音调。计算频谱图并验证峰值在预期容器中。绘制热图。
2. **中等。** 比较 40、80、128 n_mels 在同一语音片段上。用 `librosa` 验证你的手写代码与库输出在数值上一致。
3. **困难。** 从头构建完整的 STFT → Mel → log 管道（仅使用 `math`）。绘制 Speech Commands 数据集中三个不同词的 log-mel 频谱图。识别每个词的频谱特征。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------|--------|
| 频谱图 | 时间-频率表示 | 每个时间帧的频率能量热图。 |
| STFT | 短时傅里叶变换 | 分帧 + 加窗 + FFT 的组合。 |
| 梅尔尺度 | 感知频率映射 | 从 Hz 到梅尔的非线性映射；更好反映人类听觉。 |
| 梅尔频谱图 | 音频的"图像" | 将频谱图通过梅尔滤波器组压缩。 |
| 帧大小 | FFT 窗口 | 每个 FFT 的采样数；控制频率分辨率。 |
| Hop | 帧移 | 相邻帧之间移动的采样数；控制时间分辨率。 |
| Hann 窗 | 边缘平滑 | 减少帧边界不连续的锥形加权。 |

## 延伸阅读

- [librosa docs — Mel-frequency cepstral coefficients (MFCCs)](https://librosa.org/doc/latest/core.html#common-transformations) — 完整的频谱分析教程。
- [Smith — DSP Guide, Chapter 8: The Discrete Fourier Transform](https://www.dspguide.com/ch8.htm) — 频谱图的物理直觉。
- [Whisper paper — Mel spectrogram specs](https://arxiv.org/abs/2212.04356) — 80 mel bins、10ms hop 的来源。
- [Auditory filterbank design — mel and bark scales](https://asa.scitation.org/doi/10.1121/1.1835507) — 梅尔尺度的感知基础。