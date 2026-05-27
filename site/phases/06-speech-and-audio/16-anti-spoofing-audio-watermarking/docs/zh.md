# 语音反欺骗与音频水印 — ASVspoof 5、AudioSeal、WaveVerify

> 声音克隆发货快过防御。2026年生产语音系统需要两件事：一个分类真实语音 vs 合成语音的检测器（AASIST、RawNet2），和一个能抵抗压缩和编辑的水印（AudioSeal）。发货两者，否则不发货声音克隆。

**类型：** 动手实现
**语言：** Python
**前置知识：** Phase 6 · 06（说话人识别），Phase 6 · 08（声音克隆）
**时长：** 约75分钟

## 问题

三个相关防御：

1. **反欺骗 / 深度伪造检测。** 给定音频片段，是合成的还是真实的？ASVspoof 基准（ASVspoof 2019 → 2021 → 5）是黄金标准。
2. **音频水印。** 在生成音频中嵌入听感不可觉察的信号，供后续检测器提取。AudioSeal（Meta）和 WavMark 是开源选项。
3. **认证溯源。** 音频文件和元数据的加密签名。C2PA / Content Authenticity Initiative。

检测处理不合作的对手。水印处理合规——AI 生成的音频应可识别。两者在2026年都是必需的。

## 核心概念

### ASVspoof 5 — 2024-2025 基准

ASVspoof 5 相比之前版本的最大变化：

- **众包数据**（非录音棚干净）——现实条件。
- **约2000说话人**（之前约100人）。
- **32种攻击算法。** TTS + 声音转换 + 对抗扰动。
- **两个赛道。** CM 独立检测；SASV 生物识别系统。

### AASIST 和 RawNet2 — 检测模型家族

**AASIST（2021，通过2026年更新）。** 频谱特征上的图注意力。在 ASVspoof 5 CM 任务上当前 SOTA。

**RawNet2。** 原始波形卷积前端 + TDNN 主干。更简单基线；通过微调仍有竞争力。

### AudioSeal — 2024 水印默认

Meta 的 **AudioSeal**（2024年1月，v0.2 2024年12月）。关键设计：

- **局部化。** 以 16kHz 采样率逐帧检测水印（1/16000 s）。
- **生成器和检测器联合训练。** 生成器学习嵌入听感不可觉察信号；检测器学习通过增强找到它。
- **鲁棒。** 抗 MP3/AAC 压缩、EQ、速度变换 ±10%、噪声混合 +10dB SNR。
- **快速。** 检测器以 485× 实时运行；比 WavMark 快 1000×。

## 动手实现

### 步骤 1：反欺骗检测

```python
from speechbrain.inference.speaker import SpeakerRecognizer
# AASIST 或 RawNet2 模型
detector = load_model("aasist")
score = detector.classify_audio(audio)  # 0=真实，1=合成
```

### 步骤 2：AudioSeal 水印

```python
from audioseal import AudioSeal
wm_model = AudioSeal.load_watermark()
# 嵌入
watermarked = wm_model.apply_watermark(audio, message="consent_abc123")
# 检测
detected = wm_model.detect_watermark(watermarked)
```

### 步骤 3：C2PA 元数据签名

```python
# 音频文件的加密签名 + 元数据
sign(audio, private_key, metadata={"speaker_consent": "hash", "generated_at": timestamp})
```

## 陷阱

- **水印不鲁棒。** 测试抗 MP3/AAC 压缩、EQ、速度变换。
- **合成检测器漂移。** 新克隆方法可能绕过旧检测器。持续用 ASVspoof 最新版本评估。
- **同意记录缺失。** 每个克隆必须附同意 ID 可验证。

## 产出

保存为 `outputs/skill-audio-security-architect.md`。设计语音安全流水线，含检测 + 水印 + 溯源。

## 关键术语

| 术语 | 实际含义 |
|------|---|
| ASVspoof | 合成语音检测基准，2019/2021/2025/2025 版 |
| EER | 等错误率，检测器性能 |
| AudioSeal | Meta 2024 水印，16-32 bit，听感不可觉察 |
| C2PA | 内容出处和真实性协议 |
| AASIST | 图注意力反欺骗检测器，当前 SOTA |
| RawNet2 | 原始波形 TDNN 检测基线 |