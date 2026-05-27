# 音乐生成 — MusicGen、Stable Audio、Suno，以及版权地震

> 2026 年音乐生成：Suno v5 和 Udio v4 主导商业领域；MusicGen、Stable Audio Open 和 ACE-Step 领跑开源。技术问题基本解决了。法律问题（Warner Music 5亿美元和解，UMG 和解）重塑了 2025–2026 年的格局。

**类型：** 构建
**语言：** Python
**前置知识：** Phase 6 · 02（频谱图）、Phase 4 · 10（扩散模型）
**时间：** 约 75 分钟

## 问题

文本 → 30 秒到 4 分钟的音乐片段，包含歌词、人声和结构。三个子问题：

1. **器乐生成。** 文本如"lo-fi hip-hop drums with warm keys" → 音频。MusicGen、Stable Audio、AudioLDM。
2. **歌曲生成（有人声 + 歌词）。** "关于德克萨斯雨夜的乡村歌曲" → 完整歌曲。Suno、Udio、YuE、ACE-Step。
3. **条件/可控生成。** 扩展现有片段、重生成桥段、转换风格、分轨分离或修补。Udio 的修补 + 分轨分离是 2026 年需要匹配的功能。

## 概念

![音乐生成：token LM vs 扩散，2026 年模型图](../assets/music-generation.svg)

### 基于神经编解码器 token 的 Token LM

Meta 的 **MusicGen**（2023，MIT）及其衍生：以文本/旋律嵌入为条件，自回归预测 EnCodec token（32 kHz，4 个码本），用 EnCodec 解码。3 亿至 33 亿参数。强基线；超过 30 秒困难。

**ACE-Step**（开源，2026 年 4 月发布 4B XL）扩展为全曲歌词条件生成。开源社区最接近 Suno 的产品。

### 扩散 over 梅尔或隐空间

**Stable Audio（2023）** 和 **Stable Audio Open（2024）**：在压缩音频上做隐扩散。擅长循环、音效设计、环境纹理。不擅长结构完整的歌曲。

**AudioLDM / AudioLDM2**：通过 T2I 风格的隐扩散实现文本转音频，泛化到音乐、音效、语音。

### 混合（生产）— Suno、Udio、Lyria

闭源权重。可能为 AR 码本 LM + 基于扩散的声码器，带专用语音/鼓/旋律头。Suno v5（2026）是 ELO 1293 质量领先者。Udio v4 增加修补 + 分轨分离（贝斯、鼓、人声分离下载）。

### 评估

- **FAD（Fréchet 音频距离）。** 使用 VGGish 或 PANNs 特征在嵌入层面计算生成音频 vs 真实音频分布之间的距离。越低越好。MusicGen small：MusicCaps 上 4.5 FAD；SOTA 约 3.0。
- **音乐性（主观）。** 人类偏好。Suno v5 ELO 1293 领先。
- **文本-音频对齐。** CLAP 分数在 prompt 和输出之间。
- **音乐性伪影。** 节拍转换偏移、人声短语漂移、超过 30 秒结构丢失。

## 2026 年模型图

| 模型 | 参数 | 长度 | 人声 | 许可 |
|------|------|------|------|------|
| MusicGen-large | 3.3B | 30 秒 | 无 | MIT |
| Stable Audio Open | 1.2B | 47 秒 | 无 | Stability 非商业 |
| ACE-Step XL（2026 年 4 月） | 4B | >2 分钟 | 有 | Apache-2.0 |
| YuE | 7B | >2 分钟 | 有，多语言 | Apache-2.0 |
| Suno v5（闭源） | ? | 4 分钟 | 有，ELO 1293 | 商业 |
| Udio v4（闭源） | ? | 4 分钟 | 有 + 分轨 | 商业 |
| Google Lyria 3（闭源） | ? | 实时 | 有 | 商业 |
| MiniMax Music 2.5 | ? | 4 分钟 | 有 | 商业 API |

## 法律格局（2025–2026）

- **Warner Music vs Suno 和解。** 5 亿美元。WMG 现在对 AI 相似度、音乐版权和用户生成曲目有监督权。UMG 与 Udio 的类似和解。
- **EU AI Act** + **加州 SB 942**：AI 生成音乐必须披露。
- **Riffusion / MusicGen** 在 MIT 下没有合规包袱，但也无商业人声。

安全发货模式：

1. 仅生成器乐（MusicGen、Stable Audio Open、MIT/CC0 输出）。
2. 使用商业 API（Suno、Udio、ElevenLabs Music）带每次生成许可。
3. 在自有或授权目录上训练（大多数企业最终走到这里）。
4. 用水印 + 元数据标记生成物。

## 构建

### 步骤 1：用 MusicGen 生成

```python
from audiocraft.models import MusicGen
import torchaudio

model = MusicGen.get_pretrained("facebook/musicgen-small")
model.set_generation_params(duration=10)
wav = model.generate(["upbeat synthwave with driving drums, 128 BPM"])
torchaudio.save("out.wav", wav[0].cpu(), 32000)
```

三种规模：`small`（300M，快）、`medium`（1.5B）、`large`（3.3B）。Small 足以验证"这个想法是否可行"。

### 步骤 2：旋律条件

```python
melody, sr = torchaudio.load("humming.wav")
wav = model.generate_with_chroma(
    ["jazz piano cover"],
    melody.squeeze(),
    sr,
)
```

MusicGen-melody 获取色度图，在变换音色同时保留旋律。用于"把这首旋律给我变成弦乐四重奏"。

### 步骤 3：FAD 评估

```python
from frechet_audio_distance import FrechetAudioDistance
fad = FrechetAudioDistance()

fad.get_fad_score("generated_folder/", "reference_folder/")
```

计算 VGGish 嵌入距离。用于类型级回归测试；不能替代人类听众。

### 步骤 4：加入 LLM-音乐工作流

结合 Lesson 7-8 的思路：

```python
prompt = "Write a 30-second jazz loop. Describe the drums, bass, and piano voicing."
description = llm.complete(prompt)
music = musicgen.generate([description], duration=30)
```

## 使用

| 目标 | 栈 |
|------|------|
| 器乐音效设计 | Stable Audio Open |
| 游戏/自适应音乐 | Google Lyria RealTime（闭源） |
| 全歌曲有人声（商业） | Suno v5 或 Udio v4 带明确许可 |
| 全歌曲有人声（开源） | ACE-Step XL 或 YuE |
| 短广告 jingle | MusicGen 旋律条件于哼唱参考 |
| 音乐视频背景 | MusicGen + Stable Video Diffusion |

## 2026 年仍在发货的坑

- **版权洗白 prompt。** "Taylor Swift 风格的歌曲"——商业 Suno/Udio 现在过滤这些，开源模型不会。添加你自己的过滤列表。
- **超过 30 秒的重复/漂移。** AR 模型会循环。用交叉渐变拼接多个生成，或用 ACE-Step 做结构连贯性。
- **节拍漂移。** 模型会偏离 BPM。在 prompt 中使用 BPM 标签，用 librosa 的 `beat_track` 后处理过滤。
- **人声清晰度。** Suno 优秀；开源模型在人声词语上通常模糊。如果歌词重要，使用商业 API 或微调。
- **单声道输出。** 开源模型生成单声道或假立体声。用合适的立体声重建升级（ezst、Cartesia 的立体声扩散）。

## 发货

保存为 `outputs/skill-music-designer.md`。为音乐生成部署选择模型、许可策略、长度/结构计划和披露元数据。

## 练习

1. **简单。** 运行 `code/main.py`。它产生一个"生成"和弦进行 + 鼓点作为 ASCII 符号——音乐生成的卡通。通过任意 MIDI 渲染器回放。
2. **中等。** 安装 `audiocraft`，跨 4 种类型 prompt 用 MusicGen-small 生成 10 秒片段，用 FAD 测量相对于参考类型集的差异。
3. **困难。** 使用 ACE-Step（或 MusicGen-melody），用不同音色 prompt 生成同一旋律的三个变体。计算 CLAP 相似度验证对齐。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------|--------|
| FAD | 音频 FID | 真实 vs 生成音频嵌入分布之间的 Fréchet 距离。 |
| 色度图 | 旋律即音高 | 每帧 12 维向量；旋律条件的输入。 |
| 分轨 | 乐器轨道 | 分离的贝斯/鼓/人声/旋律为 WAV。 |
| 修补 | 重生成一个段落 | 遮罩一个时间窗口；模型仅重生成那部分。 |
| CLAP | 文本-音频 CLIP | 对比音频-文本嵌入；评估文本-音频对齐。 |
| EnCodec | 音乐编解码器 | Meta 的神经编解码器，用于 MusicGen；32 kHz，4 个码本。 |

## 延伸阅读

- [Copet et al. (2023). MusicGen](https://arxiv.org/abs/2306.05284) — 开放自回归基准。
- [Evans et al. (2024). Stable Audio Open](https://arxiv.org/abs/2407.14358) — 音效设计默认选择。
- [ACE-Step](https://github.com/ace-step/ACE-Step) — 开放 4B 全曲生成器，2026 年 4 月。
- [Suno v5 平台文档](https://suno.com) — 商业质量领先者。
- [AudioLDM2](https://arxiv.org/abs/2308.05734) — 音乐 + 音效的隐扩散。
- [WMG-Suno 和解报道](https://www.musicbusinessworldwide.com/suno-warner-music-settlement/) — 2025 年 11 月先例。