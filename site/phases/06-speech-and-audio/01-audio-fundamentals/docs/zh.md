# 音频基础 — 波形、采样、傅里叶变换

> 波形是原始信号。频谱图是表示形式。梅尔特征是机器学习友好的形态。每个现代 ASR 和 TTS 管道都走这条路，第一步是理解采样和傅里叶。

**类型：** 学习
**语言：** Python
**前置知识：** Phase 1 · 06（向量与矩阵）、Phase 1 · 14（概率分布）
**时间：** 约 45 分钟

## 问题

麦克风产生一个随时间变化的压力信号。神经网络消费的是张量。两者之间有一套约定，违反它们会产生静默的 bug：模型训练正常但 WER 加倍，或者 TTS 发出嘶嘶声，或者语音克隆系统记住了麦克风而不是说话人。

语音系统中的每一个 bug 都可以追溯到三个问题：

1. 数据是以什么采样率录制的，模型期望的又是什么？
2. 信号是否发生了混叠？
3. 处理的是原始采样还是频率表示？

把这三个问题搞对，Phase 6 的其余部分就是可理解的。搞错了，即使 Whisper-Large-v4 也会输出垃圾。

## 概念

![波形、采样、DFT、频率容器可视化](../assets/audio-fundamentals.svg)

**波形。** 一个一维浮点数数组，范围在 `[-1.0, 1.0]`。以采样序号为索引。转换为秒：除以采样率：`t = n / sr`。一个 10 秒的片段，16 kHz 采样，是 160,000 个浮点数的数组。

**采样率（sr）。** 每秒多少个采样点。2026 年常见采样率：

| 采样率 | 用途 |
|--------|------|
| 8 kHz | 电话、传统 VOIP。奈奎斯特 4 kHz 会吃掉辅音。ASR 避免使用。 |
| 16 kHz | ASR 标准。Whisper、Parakeet、SeamlessM4T v2 都消耗 16 kHz。 |
| 22.05 kHz | 旧模型的 TTS 声码器训练。 |
| 24 kHz | 现代 TTS（Kokoro、F5-TTS、xTTS v2）。 |
| 44.1 kHz | CD 音频、音乐。 |
| 48 kHz | 电影、专业音频、高保真 TTS（VALL-E 2、NaturalSpeech 3）。 |

**奈奎斯特-香农定理。** 采样率 `sr` 可以无歧义地表示最高 `sr/2` 的频率。`sr/2` 边界是**奈奎斯特频率**。高于奈奎斯特的能量会发生**混叠**——折回低频——从而破坏信号。降采样前务必进行低通滤波。

**位深度。** 16 位 PCM（有符号 int16，范围 ±32,767）是通用交换格式。音乐用 24 位，内部 DSP 用 32 位浮点。像 `soundfile` 这样的库读取 int16，但暴露为 `[-1, 1]` 范围内的 float32 数组。

**傅里叶变换。** 任何有限信号都是不同频率正弦波的叠加。离散傅里叶变换（DFT）对于 `N` 个采样，计算出 `N` 个复系数——每个频率容器一个。`bin k` 对应频率 `k · sr / N` Hz。幅度是该频率的能量，角度是相位。

**FFT。** 快速傅里叶变换：当 `N` 为 2 的幂时，DFT 的 `O(N log N)` 算法。每个音频库底层都使用 FFT。1024 采样点的 FFT 在 16 kHz 下产生 512 个可用频率容器，覆盖 0–8 kHz，分辨率 15.6 Hz。

**分帧 + 窗口。** 不对整个片段做 FFT。将其切成重叠的**帧**（通常 25 ms，步长 10 ms），每帧乘以一个窗函数（Hann、Hamming）以消除边缘不连续，然后对每帧做 FFT。这就是短时傅里叶变换（STFT）。Lesson 02 在此基础上展开。

## 构建

### 步骤 1：读取片段并绘制波形

`code/main.py` 只使用 stdlib 的 `wave` 模块来保持 demo 无依赖。在生产中你会使用 `soundfile` 或 `torchaudio.load`（都返回 `(waveform, sr)` 元组）：

```python
import soundfile as sf
waveform, sr = sf.read("clip.wav", dtype="float32")  # shape (T,), sr=int
```

### 步骤 2：从第一性原理合成正弦波

```python
import math

def sine(freq_hz, sr, seconds, amp=0.5):
    n = int(sr * seconds)
    return [amp * math.sin(2 * math.pi * freq_hz * i / sr) for i in range(n)]
```

440 Hz 正弦波（音乐标准音 A）以 16 kHz 采样 1 秒，产生 16,000 个浮点数。用 `wave.open(..., "wb")` 和 16 位 PCM 编码写入。

### 步骤 3：手算 DFT

```python
def dft(x):
    N = len(x)
    out = []
    for k in range(N):
        re = sum(x[n] * math.cos(-2 * math.pi * k * n / N) for n in range(N))
        im = sum(x[n] * math.sin(-2 * math.pi * k * n / N) for n in range(N))
        out.append((re, im))
    return out
```

`O(N²)` — 对 `N=256` 验证正确性没问题，对真实音频无用。真实代码调用 `numpy.fft.rfft` 或 `torch.fft.rfft`。

### 步骤 4：找主频率

幅度峰值索引 `k_star` 对应频率 `k_star * sr / N`。在 440 Hz 正弦波上运行应返回一个在 `440 * N / sr` 容器的峰值。

### 步骤 5：演示混叠

在 10 kHz 采样 7 kHz 正弦波（奈奎斯特 = 5 kHz）。7 kHz 音调高于奈奎斯特，折回 `10 − 7 = 3 kHz`。FFT 峰值出现在 3 kHz。这是经典的混叠演示，也是每个 DAC/ADC 都配备砖墙低通滤波器的原因。

## 使用

2026 年你实际发货的栈：

| 任务 | 库 | 为什么 |
|------|-----|--------|
| 读/写 WAV/FLAC/OGG | `soundfile`（libsndfile 封装） | 最快，稳定，返回 float32。 |
| 重采样 | `torchaudio.transforms.Resample` 或 `librosa.resample` | 内置正确的抗混叠。 |
| STFT / Mel | `torchaudio` 或 `librosa` | GPU 友好；PyTorch 生态系统。 |
| 实时流 | `sounddevice` 或 `pyaudio` | 跨平台 PortAudio 绑定。 |
| 检查文件 | `ffprobe` 或 `soxi` | CLI，快，报告 sr/声道/编解码器。 |

决策规则：**先匹配采样率，再匹配其他任何东西**。Whisper 期望 16 kHz 单声道 float32。传入 44.1 kHz 立体声，你会得到看起来像模型 bug 的垃圾。

## 发货

保存为 `outputs/skill-audio-loader.md`。该 skill 帮助检查音频输入是否符合下游模型的要求，并在不匹配时正确重采样。

## 练习

1. **简单。** 合成 220 Hz + 440 Hz + 880 Hz 的 1 秒混音，16 kHz 采样。运行 DFT。确认三个峰值出现在预期的容器中。
2. **中等。** 以 48 kHz 录制你的声音的 3 秒 WAV。降至 16 kHz 使用 `torchaudio.transforms.Resample`（带抗混叠），再用朴素抽取（每隔三个采样取一个）。对两者做 FFT。混叠出现在哪里？
3. **困难。** 仅使用 `math` 和步骤 3 的 DFT 从头构建 STFT。帧大小 400，步长 160，Hann 窗。用 `matplotlib.pyplot.imshow` 绘制幅度。这就是 Lesson 02 的频谱图。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------|--------|
| 采样率 | 每秒多少个采样 | ADC 测量信号的频率（Hz）。 |
| 奈奎斯特 | 可表示的最高频率 | `sr/2`；高于它的能量会混叠回低频。 |
| 位深度 | 每个采样的分辨率 | `int16` = 65,536 级；`float32` = `[-1, 1]` 中 24 位精度。 |
| DFT | 序列的傅里叶变换 | `N` 个采样 → `N` 个复频率系数。 |
| FFT | 快速 DFT | `O(N log N)` 算法，需要 `N` 为 2 的幂。 |
| 容器 | 频率列 | `k · sr / N` Hz；分辨率 = `sr / N`。 |
| STFT | 频谱图的底层 | 对时间分帧 + 加窗的 FFT。 |
| 混叠 | 奇怪的频率鬼影 | 高于奈奎斯特的能量镜像到低频容器。 |

## 延伸阅读

- [Shannon (1949). Communication in the Presence of Noise](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf) — 采样定理背后的论文。
- [Smith — The Scientist and Engineer's Guide to Digital Signal Processing](https://www.dspguide.com/ch8.htm) — 免费，权威的 DSP 教材。
- [librosa docs — audio primer](https://librosa.org/doc/latest/tutorial.html) — 带代码的实用教程。
- [Heinrich Kuttruff — Room Acoustics (6th ed.)](https://www.routledge.com/Room-Acoustics/Kuttruff/p/book/9781482260434) — 现实世界音频不是干净正弦波的原因。
- [Steve Eddins — FFT Interpretation notebook](https://blogs.mathworks.com/steve/2020/03/30/fft-spectrum-and-spectral-densities/) — 10 分钟内理清频率容器直觉。