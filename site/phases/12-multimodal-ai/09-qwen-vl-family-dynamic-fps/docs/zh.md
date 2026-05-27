# Qwen-VL 家族与动态-FPS 视频

> Qwen-VL 家族——Qwen-VL（2023）、Qwen2-VL（2024）、Qwen2.5-VL（2025）、Qwen3-VL（2025）——是 2026 年最有影响力的开源视觉-语言模型血脉。每一代都做出一个决定性的架构赌注，开源生态在十二个月内复制：原生动态分辨率通过 M-RoPE、动态-FPS 采样与绝对时间对齐、ViT 中的窗口注意力，以及结构化 agent 输出格式。到 Qwen3-VL，配方已经稳定：带原生宽高比输入的 2D-RoPE-ViT 编码器、到大型 Qwen3 语言基座的 MLP 投影器，以及将 OCR、接地和 agent 行为作为一等公民的训练阶段。本节按时间顺序解读该家族，让你理解为什么每个旋钮在那里。

**类型：** Learn
**语言：** Python（标准库，M-RoPE 编码器 + 动态-FPS 采样器）
**前置知识：** Phase 12 · 06（patch-n'-pack）
**时间：** 约 120 分钟

## 学习目标

- 计算 M-RoPE 的三轴旋转（时间、高度、宽度）并解释为什么三个都必需。
- 为视频选择动态-FPS 采样策略，并推理每分钟 token 数 vs 事件检测准确率。
- 按顺序说出 Qwen-VL 的四个代际升级及每个启用了什么。
- 连接 Qwen2.5-VL 风格的 JSON agent 输出格式并从 VLM 响应中解析结构化工具调用。

## 问题背景

Qwen-VL 于 2023 年 8 月作为 LLaVA-1.5 和 BLIP-2 的直接对手出货。Qwen 团队瞄准的差距是三维的：分辨率、视频和结构化输出。

分辨率：LLaVA-1.5 在 336x336 运行。对照片可以，对中文发票或密集表格截图无效。Qwen-VL 的第一个创新是 448x448 和带 grounding 的边界框输出，让模型指向物体。

视频：Video-LLaMA 堆叠逐帧编码器并喂给 LLM。对短视频片段有效，对多分钟视频（时间轴是信号所在）无效。Qwen 团队想要一个理解时间的单一编码器。

结构化输出：LLaVA 发出自由格式文本。Agent 需要 JSON。Qwen-VL 在显式 JSON 输出格式上训练，包括边界框坐标作为文本。

每个 Qwen-VL 代际都延伸了这三个轴中的一个。

## 核心概念

### Qwen-VL（2023 年 8 月）

第一代：OpenCLIP ViT-bigG/14 作为编码器（25 亿参数），与 Llama 兼容的 Q-Former（1 步，256 个 query），Qwen-7B 基座。贡献：

- 448x448 分辨率（当时开源 VLM 的 SOTA）。
- Grounding：在带显式坐标 token 输出的图文对上训练。"The cat is at `<box>(112, 204), (280, 344)</box>`"。
- 从一开始就是中英双语训练。

当时基准：在英语上与 GPT-4V 竞争，在中文上占主导。Grounding 监督是真正的头条。

### Qwen2-VL（2024 年 9 月）— M-RoPE 和原生分辨率

Qwen2-VL 用原生动态分辨率 ViT 编码器替换了固定分辨率 + Q-Former 堆栈。关键变化：

- 原生动态分辨率。ViT 接受任意可被 28（patch 14 加 2x 空间合并）整除的 HxW。1120x672（40x24 合并 patches）的图像产生 960 视觉 token。无 resize，无平铺，无缩略图。
- M-RoPE（多模态 RoPE）。每个 token 携带 3D 位置（t, h, w）而不是 1D。对图像 t=0，对视频 t = 帧索引。RoPE 按每轴频率旋转 query/key 向量。无位置编码表。
- MLP 投影器。去掉 Q-Former；在合并的 patch token 上使用 2 层 MLP。
- 带动态 FPS 的视频。视频默认 1-2 FPS 采样，但模型接受任意帧数。

结果：Qwen2-VL-7B 在几个多模态基准上与 GPT-4o 匹配，在 DocVQA 上击败它（94.5 vs 88.4）。架构变化是决定性的一步。

### Qwen2.5-VL（2025 年 2 月）— 动态 FPS + 绝对时间

Qwen2.5-VL 的大转变是视频。动态 FPS 不只是"需要时采样更多帧"。论文正式化了：

- 绝对时间 token。不是位置索引（帧 0, 1, 2...），而是实际时间戳。"At 0:04, the cat jumps." 模型看到与帧 token 交错的 `<time>0.04</time>` token。
- 动态 FPS。慢镜头 1 FPS，动作 4+ FPS。用户或训练者选择；M-RoPE 适应。
- ViT 中的窗口注意力。空间注意力在块内是窗口化的（局部）；每隔几层加全局注意力以提高吞吐量。
- 显式 JSON 输出格式。明确训练工具调用数据：`{"tool": "click", "coords": [380, 220]}`。开箱即用的 agent。
- MRoPE-v2 缩放。位置随最大输入大小缩放，因此 10 分钟视频不会用尽频率范围。

基准：Qwen2.5-VL-72B 在大多数视频基准上击败 GPT-4o，在文档上匹配 Gemini 2.0，并在 GUI grounding 上设置开源模型 SOTA（ScreenSpot：84% vs GPT-4o 的 38%）。

### Qwen3-VL（2025 年 11 月）

Qwen3-VL 是巩固而非重塑的增量升级：更大的 LLM 骨干（Qwen3-72B）、扩展训练数据、通过 Qwen3 "思考模式"改进 OCR 和更强推理。ViT 和 M-RoPE 保持不变。论文重点是数据和训练改进而非架构。

传承要点：到 2025 年，Qwen-VL 架构已经稳定。后续代际扩展计算和数据，不改原语。

### M-RoPE 数学

经典 RoPE 用配对坐标旋转维度 `d` 的 query `q` 在位置 `m` 处：

```
q_rot[2i]   = q[2i]   * cos(m * theta_i) - q[2i+1] * sin(m * theta_i)
q_rot[2i+1] = q[2i]   * sin(m * theta_i) + q[2i+1] * cos(m * theta_i)
theta_i     = 10000^(-2i/d)
```

M-RoPE 将隐藏维度分为三段。假设 `d = 96`。分配 32 维给时间，32 给高度，32 给宽度。每段按自己的轴位置旋转。在 (t=5, h=10, w=20) 处的 patch 获得对其三段应用的旋转 `R_t(5)`、`R_h(10)`、`R_w(20)`。

文本 token 使用 `t = text_index, h = 0, w = 0`（或归一化选择），保持兼容性。视频帧使用 `t = frame_time, h = row, w = col`。单图像使用 `t = 0`。

好处：一个位置编码处理文本、图像和视频，无需分支代码或不同位置表。

### 动态-FPS 采样逻辑

给定持续时间 `T` 秒的视频和目标 token 预算 `B`：

1. 计算你能承受的最大 FPS：`fps_max = B / (T * tokens_per_frame)`。
2. 从 `{1, 2, 4, 8}` 中选择满足 `fps <= fps_max` 的目标 FPS。
3. 如果运动剧烈（光流启发式或显式用户请求），选更高 FPS。如果运动平缓，选更低 FPS。
4. 在选定的 FPS 上均匀采样；在帧之间插入 `<time>t</time>` token。

Qwen2.5-VL 隐式训练此逻辑；推理时用户通过 `fps` 参数控制。60 秒动作序列在 4 FPS、每帧 81 token = 19440 token，在 32k 上下文中可控。

### 结构化 agent 输出

Qwen2.5-VL 的 agent 训练明确针对结构化工具调用：

```
{
  "tool": "mouse_click",
  "coords": [1024, 512],
  "button": "left",
  "modifier": null
}
```

解析是确定性的：对模型输出做 JSON.parse。与自由格式 "click at (1024, 512)" 对比，后者需要正则表达式和歧义处理。这一转变是 Qwen2.5-VL 的 ScreenSpot 分数从 Qwen2-VL 的 55% 跳到 84% 的原因。

## 使用方法

`code/main.py` 实现：

- 为混合文本、图像 patch 和视频帧的打包序列计算 M-RoPE 位置。
- 动态-FPS 采样器：给定（持续时间、预算、运动级别），选 FPS 并发出帧时间戳。
- 一个玩具 Qwen2.5-VL JSON 输出解析器，处理带坐标字段的工具调用响应。

运行它，然后在 5 分钟视频上用动态 FPS 替换固定 FPS，感受差异。

## 输出作品

本节生成 `outputs/skill-qwen-vl-pipeline-designer.md`。给定视频任务（监控、agent、动作识别、无障碍），发出 Qwen2.5-VL 配置（帧预算、FPS 策略、窗口注意力标志、agent 输出模式）和延迟估算。每当部署 Qwen-VL 家族模型做视频产品时使用。

## 练习

1. 计算在 hidden 48（每段 16，基础 theta 10000）下 patch 在 (t=3, h=5, w=7) 处的 M-RoPE 旋转。显示每段前三对的旋转角度。

2. 10 分钟监控摄像头录像在 1 FPS 下产生多少帧？在 384 分辨率、3x 池化下，总共多少 token？Qwen2.5-VL 默认 32k 上下文能处理吗？

3. 为 30 秒网球回合 vs 30 秒食谱演示 vs 30 秒 UI-agent 录像选 FPS。用动态-FPS 逻辑为每个提供理由。

4. Qwen2.5-VL 完全放弃了 Q-Former。为什么 2025 年简单的 MLP 可以工作而 2023 年不行？（提示：数据规模和编码器质量。）

5. 将三个 Qwen2.5-VL JSON 工具调用输出解析为 Python 字典。格式错误的 JSON 什么情况下会失败？Qwen 食谱推荐什么恢复策略？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| M-RoPE | "多模态 RoPE" | 隐藏维度中带时间、高度和宽度波段的 3D 旋转位置 embedding |
| Dynamic FPS | "智能采样" | 基于运动、持续时间和 token 预算为每个视频选择的帧采样率 |
| Absolute time token | "时间戳 token" | 序列中交错的 `<time>t</time>`，让模型看到实际秒数而非帧索引 |
| Window attention | "局部注意力" | 空间自注意力限制在小窗口内以提高速度；定期加全局注意力 |
| Structured agent output | "JSON 模式" | 训练数据监督教 VLM 发出带坐标和工具名称的可解析 JSON |
| min_pixels / max_pixels | "分辨率边界" | 每请求 Qwen2.5-VL 控制总像素数因而控制 token 数的边界 |
| Grounding | "指向它" | 输出边界框坐标作为文本 token；从 Qwen-VL v1 开始使用 |

## 延伸阅读

- [Bai 等 — Qwen-VL (arXiv:2308.12966)](https://arxiv.org/abs/2308.12966)
- [Wang 等 — Qwen2-VL (arXiv:2409.12191)](https://arxiv.org/abs/2409.12191)
- [Qwen Team — Qwen2.5-VL Technical Report (arXiv:2502.13923)](https://arxiv.org/abs/2502.13923)
- [Qwen Team — Qwen3-VL (arXiv:2511.21631)](https://arxiv.org/abs/2511.21631)
- [Zhu 等 — InternVL3 (arXiv:2504.10479)](https://arxiv.org/abs/2504.10479)