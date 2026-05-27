# Capstone 12 — 视频理解流水线（场景、问答、搜索）

> Twelve Labs 将 Marengo + Pegasus 产品化。VideoDB 发版了视频 CRUD API。AI2 的 Molmo 2 发布了开源 VLM 检查点。Gemini 长上下文原生处理数小时视频。TimeLens-100K 定义了大规模时序定位。2026年的流水线已确定：场景分割、每场景字幕 + 嵌入、对齐转录稿、多向量索引，以及用 (start, end) 时间戳加帧预览回答的查询。毕业项目是摄取100小时视频，在公开基准上达标，并测量计数和动作类问题的幻觉率。

**类型：** 毕业项目
**语言：** Python（流水线），TypeScript（UI）
**前置知识：** Phase 4（计算机视觉）、Phase 6（语音）、Phase 7（Transformers）、Phase 11（LLM工程）、Phase 12（多模态）、Phase 17（基础设施）
**涉及阶段：** P4 · P6 · P7 · P11 · P12 · P17
**时长：** 30小时

## 问题

长视频问答是2026年规模下带宽需求最大的多模态问题。Gemini 2.5 Pro 能原生读取2小时视频，但将100小时视频摄取到可查询语料库仍需要场景级索引。生产形态结合场景分割（TransNetV2 或 PySceneDetect）、用 VLM 做每场景字幕（Gemini 2.5、Qwen3-VL-Max 或 Molmo 2）、转录稿对齐（带词级时间戳的 Whisper-v3-turbo）以及多向量索引，在同一处存储字幕、帧嵌入和转录。查询流水线用 (start, end) 时间戳加帧预览回答。

基准是公开的（ActivityNet-QA、NeXT-GQA）加上你自己的100题自定义集。计数和动作类问题的幻觉是已知困难的失败类；毕业项目显式测量它。

## 核心概念

摄取时三条流水线并行运行。**场景分割**将视频切成场景。**VLM 字幕**为每个场景生成字幕，并从关键帧生成帧嵌入。**ASR 对齐**产生词级时间戳。三条流通过 (scene_id, time range) 连接。每个场景在多向量索引（Qdrant）中有三种向量类型：字幕嵌入、关键帧嵌入、转录嵌入。

查询时，自然语言问题同时查三种向量；结果用 RRF 合并；时序定位适配器（TimeLens 风格）在 top 场景内细化 (start, end) 窗口。VLM 合成器（Gemini 2.5 Pro 或 Qwen3-VL-Max）取查询 + top 场景 + 裁剪帧，回答时附引用时间戳和帧预览。

幻觉测量很重要。计数（"有多少人进入房间？"）和动作类（"厨师在搅拌前倒了吗？"）问题历来不可靠。将准确率与描述性问题分开报告。

## 架构

```
视频文件 / URL
      |
      v
PySceneDetect / TransNetV2  （场景分割）
      |
      +--- 每场景关键帧 --- VLM 字幕 + 帧嵌入
      |                            （Gemini 2.5 Pro / Qwen3-VL-Max / Molmo 2）
      |
      +--- 音频通道 --- Whisper-v3-turbo ASR + 词级时间戳
      |
      v
多向量 Qdrant：{caption_emb, keyframe_emb, transcript_emb}
      |
查询:
   三路密集查询 -> RRF 合并 -> top-k 场景
      |
      v
TimeLens / VideoITG 时序定位（在场景内细化 start/end）
      |
      v
VLM 合成：查询 + top 场景 + 帧预览
      |
      v
答案 + (start, end) 时间戳 + 帧缩略图 + 引用
```

## 技术栈

- 场景分割：TransNetV2（2024-26 年前沿）或 PySceneDetect
- ASR：通过 faster-whisper 用 Whisper-v3-turbo，带词级时间戳
- VLM 字幕 + 回答：Gemini 2.5 Pro 或 Qwen3-VL-Max 或 Molmo 2
- 时序定位：TimeLens-100K 训练的适配器或 VideoITG
- 索引：Qdrant 带多向量支持（字幕 / 帧 / 转录）
- UI：Next.js 15 带 HTML5 视频播放器和场景缩略图
- 评估：ActivityNet-QA、NeXT-GQA、100题自定义人工标注集
- 幻觉基准：计数和动作类子集，带人工标签

## 动手实现

1. **摄取遍历器。** 接受 YouTube URL 或本地 MP4。如需要缩放到 720p。持久化 `{video_id, file_path}`。

2. **场景分割。** 运行 TransNetV2 或 PySceneDetect 产出 `[{scene_id, start_ms, end_ms, keyframe_path}]`。目标100小时：约6k-8k 场景。

3. **ASR 通过。** 在音频上运行 Whisper-v3-turbo；导出词级时间戳；切分为每场景转录切片。

4. **VLM 字幕。** 每场景，用关键帧和短字幕模板调用 Gemini 2.5 Pro（或 Qwen3-VL-Max）。产出字幕 + 帧嵌入。

5. **多向量索引。** Qdrant collection 三个命名向量。Payload：`{video_id, scene_id, start_ms, end_ms, keyframe_url}`。

6. **查询。** 自然语言问题发三路密集查询；RRF 合并；top-k=5 场景。

7. **时序定位。** 在 top 场景上运行 TimeLens 风格适配器，细化场景内的 (start, end) 窗口。

8. **VLM 合成。** 用查询 + top-3 场景片段（作图像或短视频）+ 转录调用 Gemini 2.5 Pro。要求 `(video_id, start_ms, end_ms)` 引用。

9. **评估。** 运行 ActivityNet-QA 和 NeXT-GQA。构建100题自定义集。报告总体准确率和分类明细（计数、动作、描述）。

## 用现成库

```bash
$ video-qa ask --url=https://youtube.com/watch?v=X "第一分钟内有多少辆车经过这个路口？"
[场景]    检测到23个场景
[ASR]      转录完成，4m12s
[索引]     写入69个向量（23场景 x 3）
[查询]     top 场景：场景3 [01:32-01:54]，置信度 0.84
[定位]     细化窗口：[00:12-00:58]
[合成]     gemini 2.5 pro，1.4s
答案:      5辆车在00:12到00:58之间经过了这个路口。
引用:      [场景3：00:12-00:58]
          [帧预览 at 00:14, 00:27, 00:44, 00:51, 00:57]
```

## 产出

`outputs/skill-video-qa.md` 是交付物。给定 YouTube URL 或上传视频，流水线索引场景并用带时间戳引用的方式回答问题。

| 权重 | 指标 | 衡量方式 |
|:-:|---|---|
| 25 | 时序定位 IoU | 在 held-out 定位集上的交并比 |
| 20 | 问答准确率 | NeXT-GQA 和自定义100题 |
| 20 | 摄取吞吐 | 每美元处理的视频小时数 |
| 20 | UI 和引用 UX | 时间戳链接、缩略图条、跳转到帧 |
| 15 | 幻觉率 | 计数和动作类准确率分开 |
| **100** | | |

## 练习

1. 在字幕通过上将 Gemini 2.5 Pro 换成 Qwen3-VL-Max。在人工评分的50场景样本上报告字幕质量差值。

2. 将每场景帧嵌入缩减为单一池化向量而非多向量。测量检索退化。

3. 构建"严格计数"模式：合成器提取每个计数的实例并附时间戳，用户点击验证。测量用户验证是否减少幻觉。

4. 基准摄取成本：三个 VLM 选项的每美元视频小时数。选择最佳性价比。

5. 添加说话人分离转录：在音频上运行 pyannote 说话人分离并嵌入每说话人转录。演示"Alice 关于 X 说了什么？"查询。

## 关键术语

| 术语 | 行话 | 实际含义 |
|------|-----------------|------------------------|
| Scene segmentation | "镜头检测" | 在镜头边界处将视频切成场景 |
| Multi-vector index | "字幕 + 帧 + 转录" | 每种表示在 Qdrant collection 中有命名向量 |
| Temporal grounding | "具体什么时候发生的" | 在查询答案的 (start, end) 窗口内细化 |
| Frame embedding | "视觉表示" | 关键帧的向量嵌入；用于场景视觉相似性 |
| RRF fusion | "倒数排名融合" | 多排名列表的合并策略；经典的混合检索技巧 |
| Counting hallucination | "数错" | VLM 在"有多少 X"问题上的已知失败模式 |
| ActivityNet-QA | "视频问答基准" | 长视频问答准确率基准 |

## 扩展阅读

- [AI2 Molmo 2](https://allenai.org/blog/molmo2) — 开源 VLM 检查点
- [TimeLens（CVPR 2026）](https://github.com/TencentARC/TimeLens) — 大规模时序定位
- [Gemini Video 长上下文](https://deepmind.google/technologies/gemini) — 托管参考
- [VideoDB](https://videodb.io) — 视频 CRUD API 参考
- [Twelve Labs Marengo + Pegasus](https://www.twelvelabs.io) — 商业参考
- [TransNetV2](https://github.com/soCzech/TransNetV2) — 场景分割模型
- [PySceneDetect](https://github.com/Breakthrough/PySceneDetect) — 经典开源替代
- [ActivityNet-QA](https://arxiv.org/abs/1906.02467) — 参考评估基准