# 音频评估 — WER、MOS、UTMOS、MMAU、FAD 和开源排行榜

> 你无法发货无法测量的东西。本课命名2026年每个音频任务的指标：ASR（WER、CER、RTFx）、TTS（MOS、UTMOS、SECS、WER 往返）、音频语言（MMAU、LongAudioBench）、音乐（FAD、CLAP）和说话人（EER）。以及你比较的排行榜。

**类型：** Learn
**语言：** Python
**前置知识：** Phase 6 · 04、06、07、09、10；Phase 2 · 09（模型评估）
**时长：** 约60分钟

## 问题

每个音频任务有多个指标，每个测量不同轴。用错误的指标是你在仪表盘上看起来很好但在生产中很糟糕地发货模型的原因。

## 指标总览

| 任务 | 主要指标 | 次要指标 |
|------|---------|---------|
| ASR | WER | CER · RTFx · 首个 token 延迟 |
| TTS | MOS / UTMOS | SECS · ASR 往返 WER · TTFA |
| 声音克隆 | SECS（ECAPA 余弦） | MOS · CER |
| 说话人验证 | EER | minDCF · FAR / FRR |
| 说话人分离 | DER | JER · 说话人混淆 |
| 音频分类 | top-1 · mAP | macro F1 · 每类召回 |
| 音乐生成 | FAD | CLAP · 听感 MOS 面板 |
| 音频语言模型 | MMAU-Pro | LongAudioBench · AudioCaps FENSE |
| 流式 S2S | 延迟 P50/P95 | WER · MOS |

## 核心概念

### ASR 指标

**WER（词错误率）。** `(S + D + I) / N`。小写，评分前去除标点，规范数字。< 5% = 朗读语音的人类水平。

**CER（字符错误率）。** 相同公式，字符级。用于词边界模糊的语调语言（普通话、粤语）。

**RTFx（逆实时因子）。** 每墙上时钟秒处理的音频秒。越高越好。Parakeet-TDT 达到 3380×。

**首个 token 延迟。** 音频输入到首个转录 token 的墙上时钟。Deepgram Nova-3：约 150ms。

### TTS 指标

**MOS（平均意见分）。** 1-5 人类评分。黄金标准但慢。每样本 20+ 听众，模型 100+ 样本。

**UTMOS（2022-2026）。** 学习的 MOS 预测器。在标准基准上与人类 MOS 约 0.9 相关。F5-TTS：UTMOS 3.95；真实录音：4.08。

**SECS（说话人嵌入余弦相似度）。** 用于声音克隆。参考与克隆输出的 ECAPA 嵌入余弦。>0.75 = 可识别克隆。

**ASR 往返 WER。** 在 TTS 输出上跑 Whisper，计算相对输入文本的 WER。捕获可懂度回归。2026年 SOTA：< 2% CER。

**TTFA（首个音频时间）。** 墙上时钟延迟。Kokoro-82M：约 100ms；F5-TTS：约 1s。

## 产出

保存为 `outputs/skill-audio-eval-architect.md`。设计给定任务的评估流水线，含指标和排行榜。

## 关键术语

| 术语 | 实际含义 |
|------|---|
| WER | 词错误率，越低越好 |
| UTMOS | 学习的 MOS 预测器，相关约 0.9 vs 人类 |
| SECS | 克隆与参考的说话人嵌入余弦相似度 |
| DER |  diarization 错误率 |
| FAD | Frechet 音频距离，音乐生成质量 |
| MMAU | 多模态音频理解基准 |