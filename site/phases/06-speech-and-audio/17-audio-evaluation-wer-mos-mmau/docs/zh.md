# 音频评估 — WER、MOS、MMAU

> ASR 用 WER。TTS 用 MOS。音频理解用 MMAU。每种指标对应不同任务，用错了指标会误导整个开发周期。

**类型：** 学习
**语言：** Python
**前置知识：** Phase 6 · 04（ASR）、Phase 6 · 07（TTS）、Phase 6 · 10（音频语言模型）
**时间：** 约 45 分钟

## 问题

你在开发一个语音助手。你跑了 Whisper-large-v3-turbo 在测试集上，得到 WER = 1.6%。好还是坏？训练了 Kokoro-v1 TTS，MOS = 4.2。够了吗？你开发了一个音频问答模型，MMAU-Pro 整体 = 52%。多音频子集 = 22%。这意味着什么？

每个音频任务有正确的指标。用错了会让你以为系统比实际好（或差）。

## 概念

### ASR：WER

**词错误率（Word Error Rate）。** 编辑距离在词级别：

```
WER = (S + D + I) / N
S = 替换（wrong word）
D = 删除（missing word）
I = 插入（extra word）
N = 参考词数
```

```python
def wer(ref, hyp):
    r, h = ref.split(), hyp.split()
    dp = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        dp[i][0] = i
    for j in range(len(h) + 1):
        dp[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i-1] == h[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)
    return dp[len(r)][len(h)] / max(1, len(r))
```

**2026 年基准：**

| 场景 | WER 含义 |
|------|---------|
| < 3% | 人类水平（LibriSpeech 等干净语音） |
| 3–10% | 高质量，适合大多数应用 |
| 10–20% | 尚可，需要改进 |
| > 20% | 不可用 |

**WER 的坑：**
- 不报告字符 vs 词 vs 子词级别
- 不对大小写和标点归一化
- 不告诉你错误的类型（是删除还是替换更常见）

### ASR：其他指标

| 指标 | 用途 | 公式 |
|------|------|------|
| CER（字符错误率） | 中文等字符语言 | 类似 WER，但按字符计算 |
| TER（令牌错误率） | 子词/音素 | 类似 WER，但按 token 计算 |
| SER（句子错误率） | 完整句子 | 至少一个错误的句子比例 |
| Accuracy | 分类（语言 ID 等） | 正确数 / 总数 |
| AUC-ROC | 开放集检测 | 阈值无关的质量 |

### TTS：MOS

**平均意见分（Mean Opinion Score）。** 人类评分员 1–5 分：

```
MOS = sum(score_i) / N
N = 评分员数量
```

**2026 年质量标准：**

| MOS | 质量描述 |
|-----|---------|
| 5.0 | 完美，无法区分 |
| 4.0–4.5 | 高质量，商业可用 |
| 3.5–4.0 | 可接受，某些方面有瑕疵 |
| 3.0–3.5 | 明显不自然 |
| < 3.0 | 不可用 |

**MOS 的坑：**
- 主观性：不同评分员有不同的标准
- 上下文：相同音频在安静 vs 嘈杂环境中得分不同
- 无可懂度信息：MOS 5 的音频可能是混乱但听起来"自然"

### TTS：其他指标

| 指标 | 测什么 | 如何计算 |
|------|--------|---------|
| SECS | 说话人相似度 | 克隆声音 vs 原始的嵌入余弦相似度 |
| PESQ | 感知质量 | 客观，范围 -0.5 到 4.5 |
| ViSQOL | 主观质量 | 客观，与 MOS 强相关 |
| STOI | 短时目标清晰度 | 语音清晰度，0–1 |
| WER | 可懂度 | ASR 在 TTS 输出上的 WER |

### 音频分类：精度/召回/F1

```python
from sklearn.metrics import classification_report

report = classification_report(y_true, y_pred, target_names=classes)
print(report)
```

### 音频理解：MMAU

**多模态音频理解评估（MultiModal Audio Understanding Benchmark）。**

MMAU 评估音频语言模型在四个类别上的理解：

| 子集 | 内容 | 说明 |
|------|------|------|
| 语音 | 转录、QA、说话人 | ASR 能力的延伸 |
| 声音 | 事件检测、场景分类 | 非语音音频 |
| 音乐 | 流派、乐器、情感 | 音乐理解 |
| 多音频 | 比较两个片段 | 推理跨片段；最难 |

**解读 MMAU 分数：**

| 分数 | 含义 |
|------|------|
| > 70% | 强；在该类别上接近人类 |
| 50–70% | 中等；某些任务困难 |
| 30–50% | 弱；需要大量改进 |
| ~25% | 随机（四选项） |
| < 25% | 比随机差 |

**多音频的 22% 意味着什么？** 仅比随机（25%）略低。当前模型难以做跨片段比较和推理。

### 综合评估框架

```python
def comprehensive_audio_eval(model_type, audio, reference, predictions):
    if model_type == "asr":
        return {
            "wer": wer(reference, predictions["text"]),
            "cer": cer(reference, predictions["text"]),
            "accuracy": accuracy(reference, predictions["text"])
        }
    elif model_type == "tts":
        return {
            "mos": predictions["mos"],
            "secs": secs(predictions["embedding"], reference["embedding"]),
            "wer": wer(reference["text"], transcribe(predictions["audio"]))
        }
    elif model_type == "alm":
        return {
            "overall": predictions["mmau_score"],
            "speech": predictions["speech_score"],
            "sound": predictions["sound_score"],
            "music": predictions["music_score"],
            "multi_audio": predictions["multi_audio_score"]
        }
```

## 坑

- **WER 归一化。** 始终在计算 WER 前对参考和预测进行归一化（去标点、小写、Unicode 归一化）。否则你会得到假的高 WER。
- **MOS 评估的众包偏差。** 众包评分员可能对某些语言或口音有偏见。使用本地评分员或针对目标人群的评分员。
- **MMAU 子集分离。** 报告整体分数同时报告每类分数。多音频 = 22% 会被 60% 的语音分数掩盖。
- **时间戳质量。** ASR 的 WER 不反映时间戳准确性。如果需要时间戳，单独评估。
- **长音频评估。** MMAU 是 5 秒片段。对于长音频（播客、会议），需要不同的评估策略（摘要质量、说话人识别等）。

## 发货

保存为 `outputs/skill-audio-eval-designer.md`。为给定的音频任务选择正确的评估指标组合，并设置评估管道。

## 练习

1. **简单。** 运行 `code/main.py`。计算模拟 ASR 输出的 WER、CER、WER（归一化后）。对比差异。
2. **中等。** 在 Speech Commands v2 数据集上评估 Whisper-turbo。报告 WER per 类别（yes/no/stop/go 等）。哪些类别最难？
3. **困难。** 在 MMAU-Pro 上评估 Qwen2.5-Omni-7B（或类似模型）。分别报告四个子集分数，并解释多音频分数的含义。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------|--------|
| WER | 词错误率 | 词级编辑距离：`(S+D+I)/N`。ASR 标准指标。 |
| CER | 字符错误率 | 字符级 WER。用于中文等字符语言。 |
| SER | 句子错误率 | 至少一个词错误的句子比例。 |
| MOS | 平均意见分 | 人类评分员 1–5 分的均值。TTS 标准指标。 |
| SECS | 说话人嵌入相似度 | 克隆 vs 原始的嵌入余弦相似度。 |
| PESQ | 感知质量 | 客观语音质量指标，范围 -0.5 到 4.5。 |
| MMAU | 多模态音频理解 | 音频语言模型的综合评估基准。 |
| mAP | 平均精度均值 | 多标签分类的标准指标。 |
| AUC-ROC | 曲线下面积 | 阈值无关的分类质量。 |

## 延伸阅读

- [Park et al. (2024). MMAU: A Unified Benchmark for Audio Understanding](https://arxiv.org/abs/2407.10952) — MMAU 基准。
- [MMAU-Pro leaderboard](https://mmaubenchmark.github.io/) — 实时分数。
- [Graves (2006). Connectionist Temporal Classification](https://www.cs.toronto.edu/~graves/icml_2006.pdf) — WER 的背景。
- [LibriSpeech leaderboard](https://www.openslr.org/12) — ASR 标准基准。
- [Cosign MOS leaderboard](https://paperswithcode.com/sota/text-to-speech-on-ljspeech) — TTS MOS 基准。
- [ASVspoof 2024](https://www.asvspoof.org/) — 反欺骗评估。