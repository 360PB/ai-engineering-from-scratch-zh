# 基准测试：WebArena 与 OSWorld

> WebArena 在四个自托管应用上测试 Web Agent 能力。OSWorld 在 Ubuntu、Windows、macOS 上测试桌面 Agent 能力。在发布时（2023–2024），两者都显示了最佳 Agent 与人类之间的巨大差距。差距在缩小；失败模式没有改变。

**类型：** 学习
**语言：** Python（标准库）
**前置知识：** Phase 14 · 19（SWE-bench, GAIA）
**时间：** 约 60 分钟

## 学习目标

- 描述 WebArena 的四个自托管应用以及为何基于执行结果的评估很重要。
- 解释为何 OSWorld 使用真实操作系统截图而非无障碍 API。
- 说出 OSWorld 两个主要失败模式：GUI 定位和操作知识。
- 总结 OSWorld-G 和 OSWorld-Human 在基础基准上添加了什么。
- 理解 WebArena 和 OSWorld 如何塑造了计算机使用 Agent 的训练工作负载。

## 问题背景

通用 Agent 能调用工具。它们能驱动浏览器完成 20 次点击的购物结账吗？它们能仅用键盘和鼠标配置一台 Linux 服务器吗？WebArena 和 OSWorld 回答这些问题。

## 核心概念

### WebArena（Zhou et al., ICLR 2024）

- 跨越四个自托管 Web 应用：购物网站、论坛、类 GitLab 开发工具、企业 CMS 的 812 个长期任务。
- 辅助工具：地图、计算器、便签本。
- 评估基于 gym API 执行——订单是否下了、issue 是否关闭、CMS 页面是否更新？
- 发布时：最佳 GPT-4 Agent 14.41% vs 人类 78.24%。

自托管框架很重要——基准不 flak，因为目标应用是固定的且可复现。

### 扩展

- **VisualWebArena** — 视觉落地任务，成功取决于对图像（截图作为一等观察）的解读。
- **TheAgentCompany**（2024 年 12 月）— 增加终端 + 编码；更像真实的远程工作环境。

### OSWorld（Xie et al., NeurIPS 2024）

- 跨越 Ubuntu、Windows、macOS 的 369 个真实计算机任务。
- 通过自由形式键盘和鼠标控制真实应用。
- 1920×1080 截图作为观察。
- 发布时：最佳模型 12.24% vs 人类 72.36%。

### 主要失败模式

1. **GUI 定位。** 像素→元素映射。模型在 1920×1080 中难以可靠地定位 UI 元素。
2. **操作知识。** 哪个菜单有这个设置，哪个键盘快捷键，哪个偏好面板。这是人类经过多年积累的知识尾。

### 后续工作

- **OSWorld-G** — 564 样本的定位套件 + Jedi 训练集。将定位从规划中分解出来，可以分别测量。
- **OSWorld-Human** — 人工策划的金牌动作轨迹。显示最佳 Agent 比必要多使用 1.4–2.7 倍步数（轨迹效率差距）。

### 为何这很重要

Claude computer use、OpenAI CUA、Gemini 2.5 Computer Use（第 21 课）都在受 WebArena 和 OSWorld 塑造的工作负载上训练。基准是目标；生产模型是交付的答案。

### 基准测试会出错的地方

- **仅截图评估。** OSWorld 是截图驱动的；对使用 DOM 或无障碍 API 的 Agent 在 OSWorld 上评估会错过定位挑战。
- **忽略轨迹长度。** 只评成功率会遗漏 OSWorld-Human 揭示的 1.4–2.7 倍步数低效。
- **自托管应用版本过时。** WebArena 的应用固定了特定版本；更新而不重新策划会破坏可比性。

## 动手实现

`code/main.py` 实现了一个 toy Web Agent 测试框架：

- 一个最小的"购物应用"状态机：list_items、add_to_cart、checkout。
- 3 个任务的金牌轨迹。
- 一个脚本 Agent 尝试每个任务。
- 基于执行结果的评估器（状态检查）和轨迹效率指标（步数 vs 金牌）。

运行：

```
python3 code/main.py
```

输出：每个任务的成功率和轨迹效率，镜像 OSWorld-Human 的方法论。

## 用现成库

- **WebArena Verified** 在内部集群上自托管用于持续评估。
- **OSWorld** 在 VM 集群上用于桌面 Agent。
- **计算机使用 Agent**（第 21 课）——Claude、OpenAI CUA、Gemini——都在类似这些的工作负载上训练。
- **你自己的产品流程** — 为前 20 个任务捕获金牌轨迹；每周用 Agent 运行它们。

## 产出

`outputs/skill-web-desktop-harness.md` 构建一个 Web/桌面 Agent 测试框架，含基于执行结果的评估和轨迹效率指标。

## 练习

1. 用第二个应用（论坛）扩展玩具框架。写 3 个任务加金牌轨迹。
2. 添加每个任务的轨迹效率报告。在你的玩具上，Agent 是 1 倍、2 倍还是 3 倍金牌？
3. 实现一个"干扰"工具——金牌轨迹从不使用它。脚本 Agent 会受到诱惑吗？
4. 读取 OSWorld-G。你如何在你的评估中分离定位失败和规划失败？
5. 读取 WebArena 的应用 README。升级其中一个固定应用版本会出什么错？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| WebArena | "Web Agent 基准" | 812 个任务，跨 4 个自托管应用；gym 风格评估 |
| VisualWebArena | "视觉 WebArena" | 视觉落地的 WebArena；截图是一等观察 |
| OSWorld | "桌面 Agent 基准" | 369 个任务，在真实 Ubuntu/Windows/macOS 上 |
| GUI grounding | "像素到元素映射" | 模型在 1920×1080 中定位 UI 元素 |
| Operational knowledge | "操作系统知识" | 哪个菜单、哪个快捷键、哪个偏好面板 |
| OSWorld-G | "定位套件" | 564 个纯定位样本 + 训练集 |
| OSWorld-Human | "金牌轨迹" | 人工策划的专家动作序列，用于测量效率 |
| Trajectory efficiency | "步数相对金牌" | Agent 步数除以人类最小步数 |

## 延伸阅读

- [Zhou et al., WebArena (arXiv:2307.13854)](https://arxiv.org/abs/2307.13854) — 四应用 Web 基准
- [Xie et al., OSWorld (arXiv:2404.07972)](https://arxiv.org/abs/2404.07972) — 跨 OS 桌面基准
- [Anthropic, Introducing computer use](https://www.anthropic.com/news/3-5-models-and-computer-use) — Claude 由基准塑造的能力
- [OpenAI, Computer-Using Agent](https://openai.com/index/computer-using-agent/) — OSWorld 和 WebArena 数字