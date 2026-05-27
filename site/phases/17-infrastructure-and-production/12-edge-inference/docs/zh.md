# 边缘推理 — Apple Neural Engine、Qualcomm Hexagon、WebGPU/WebLLM、Jetson

> 边缘推理的核心约束是内存带宽，不是算力。移动端 DRAM 50-90 GB/s；数据中心 HBM3 达 2-3 TB/s——差距 30-50 倍。Decode 是内存受限的，所以差距是决定性的。2026 年格局分为四路。Apple M4/A18 Neural Engine 峰值 38 TOPS，统一内存（CPU↔NPU 无拷贝）。Qualcomm Snapdragon X Elite / 8 Gen 4 Hexagon 达 45 TOPS。WebGPU + WebLLM 在 M3 Max 上跑 Llama 3.1 8B（Q4）约 41 tok/s（原生约 70-80%）；17.6k GitHub stars，OpenAI 兼容 API，约 70-75% 移动端覆盖。NVIDIA Jetson Orin Nano Super（8GB）装 Llama 3.2 3B / Phi-3；AGX Orin 通过 vLLM 跑 gpt-oss-20b 约 40 tok/s；Jetson T4000（JetPack 7.1）是 AGX Orin 的 2 倍性能。TensorRT Edge-LLM 支持 EAGLE-3、NVFP4、分块 prefill——2026 年 CES 由 Bosch、ThunderSoft、MediaTek 展示。

**类型：** 精读
**语言：** Python（标准库，玩具级带宽受限 decode 模拟器）
**前置要求：** Phase 17 · 04（vLLM 推理内部原理）、Phase 17 · 09（生产量化）
**时长：** 约 60 分钟

## 学习目标

- 解释为什么移动端 LLM 推理是内存带宽受限，算力是次要的。
- 枚举四个边缘目标（Apple ANE、Qualcomm Hexagon、WebGPU/WebLLM、NVIDIA Jetson）并匹配各场景。
- 说出 2026 年 WebGPU 覆盖缺口（Firefox Android 仍在追赶）和 Safari iOS 26 的落地情况。
- 为每个目标选定量化格式（ANE 用 Core ML INT4 + FP16、Hexagon 用 QNN INT8/INT4、浏览器用 WebGPU Q4、Jetson Thor 用 NVFP4）。

## 背景问题

客户想要一个设备端聊天机器人：语音优先、默认私密、离线可用。在 MacBook Pro M3 Max 上，Llama 3.1 8B Q4 跑 55 tok/s——可以。在 iPhone 16 Pro 上，同一模型跑 3 tok/s——不行。在骁龙 8 Gen 3 中端 Android 上，7 tok/s。在 Chrome Android v121+ 上通过 WebGPU，4-8 tok/s 取决于设备。

吞吐差异不是移植问题。是带宽差距乘以量化格式再乘以 NPU 是否能从用户空间访问的问题。2026 年边缘推理是四个不同问题，需要四种不同解法。

## 核心概念

### 带宽是真正的上限

Decode 每生成一个 token 都要读取完整权重集。一个 7B Q4 模型 3.5 GB。在 50 GB/s 下读取 3.5 GB 需 70 ms——理论上限约 14 tok/s。带宽低于这个数字，再多算力也帮不上忙。

数据中心 HBM3 在 3 TB/s 下读取同样 3.5 GB 只需 1.2 ms——上限 830 tok/s。同模型、同权重，不同内存子系统。

### Apple Neural Engine（M4 / A18）

- 最高 38 TOPS。统一内存（CPU 和 ANE 共享同一池）——无拷贝开销。
- 通过 Core ML + 编译的 `.mlmodel` 模型或通过 PyTorch 的 Metal Performance Shaders（MPS）访问。
- llama.cpp Metal 后端用 MPS，不直接用 ANE；原生 ANE 访问需要 Core ML 转换。
- 2026 年 iOS 应用最佳路径：Core ML + INT4 权重 + FP16 激活。

### Qualcomm Hexagon（骁龙 X Elite / 8 Gen 4）

- 最高 45 TOPS。集成在 SoC 中与 CPU、GPU 共处，但内存域分离。
- QNN（Qualcomm Neural Network）SDK 和 AI Hub 提供从 PyTorch/ONNX 转换的路径。
- 聊天模板、Llama 3.2、Phi-3 都以一等公民形式在 AI Hub 上发布。

### Intel / AMD NPU（Lunar Lake、Ryzen AI 300）

- 40-50 TOPS。软件落后于 Apple/Qualcomm；OpenVINO 在改进但仍小众。
- 最适合 Windows ARM 副驾驶应用；在 AMD/Intel 台式机上用于本地优先场景。

### WebGPU + WebLLM

- 通过 WebGPU 计算 shader 在浏览器内运行模型；无需安装。
- Llama 3.1 8B Q4 在 M3 Max 上约 41 tok/s——约通过同一后端的原生 70-80%。
- WebLLM 有 17.6k GitHub stars；OpenAI 兼容 JS API；Apache 2.0。
- 2026 年覆盖：Chrome Android v121+、Safari iOS 26 GA、Firefox Android 仍在追赶。总体约 70-75% 移动端覆盖。

### NVIDIA Jetson 系列

- Orin Nano Super（8GB）：装 Llama 3.2 3B、Phi-3 效果良好。
- AGX Orin：通过 vLLM 跑 gpt-oss-20b 约 40 tok/s。
- Thor / T4000（JetPack 7.1）：性能是 AGX Orin 的 2 倍，支持 EAGLE-3 和 NVFP4。
- TensorRT Edge-LLM（2026）支持 EAGLE-3 推测解码、NVFP4 权重、分块 prefill——数据中心优化移植到边缘。

### 各目标量化格式选择

| 目标 | 格式 | 说明 |
|------|------|------|
| Apple ANE | INT4 权重 + FP16 激活 | Core ML 转换路径 |
| Qualcomm Hexagon | QNN INT8 / INT4 | AI Hub 转换器 |
| WebGPU / WebLLM | Q4 MLC（q4f16_1） | 用 `mlc_llm convert_weight` + 编译 `.wasm`；不支持 GGUF |
| Jetson Orin Nano | Q4 GGUF 或 TRT-LLM INT4 | 内存带宽受限 |
| Jetson AGX / Thor | NVFP4 + FP8 KV | Edge-LLM 路径 |

### 边缘上的长上下文陷阱

Llama 3.1 的 128K 上下文是数据中心功能。在只有 8 GB RAM 的手机上，4 GB 模型 + 2 GB KV 缓存（32K token）+ 系统开销 = OOM。边缘部署保持上下文 4K-8K，除非接受激进的 KV 量化（Q4 KV）。

### 语音是杀手级应用

语音 Agent 对延迟敏感（首个 token < 500 ms）。本地推理完全消除网络延迟。结合语音转文字（Whisper Turbo 变体可在边缘运行），边缘推理成为生产质量语音环。

### 必须记住的数字

- Apple M4 / A18 ANE：38 TOPS。
- Qualcomm Hexagon SD X Elite：45 TOPS。
- WebLLM M3 Max：Llama 3.1 8B Q4 约 41 tok/s。
- AGX Orin：通过 vLLM 跑 gpt-oss-20b 约 40 tok/s。
- 数据中心-边缘带宽差距：30-50 倍。
- WebGPU 移动端覆盖：约 70-75%（Firefox Android 落后）。

## 用现成库

`code/main.py` 从带宽受限数学计算跨边缘目标的理论 decode 吞吐上限。与实测基准对比，并高亮带宽而非算力是瓶颈的地方。

## 产出

本课产出 `outputs/skill-edge-target-picker.md`。给定平台（iOS/Android/浏览器/Jetson）、模型和延迟/内存预算，选出量化格式和转换流水线。

## 练习

1. 运行 `code/main.py`。对于 7B 模型 Q4 在骁龙 8 Gen 3（约 77 GB/s 带宽）上，计算 decode 上限。与实测 6-8 tok/s 比较——运行时效率如何？
2. WebGPU 在 Android 上需要 Chrome v121+。为更老浏览器设计降级方案——通过同一 OpenAI 兼容 API 走服务端。
3. 你的 iOS 应用需要 4K 上下文流式输出。哪个模型/格式组合让你在 iPhone 16 上活动内存保持在 4 GB 以下？
4. Jetson AGX Orin 跑 gpt-oss-20b 达 40 tok/s。Jetson Nano 只能装 3B。如果你的产品同时覆盖两者，如何统一推理栈？
5. 论证"WebLLM 在 2026 年已可用于生产"。引用覆盖、性能和 Firefox Android 差距。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| ANE | "Apple 神经引擎" | M 系列和 A 系列的设备端 NPU；统一内存 |
| Hexagon | "Qualcomm NPU" | 骁龙 NPU；通过 QNN SDK 访问 |
| WebGPU | "浏览器 GPU" | W3C 标准浏览器 GPU API；Chrome/Safari 2026 |
| WebLLM | "浏览器 LLM 运行时" | MLC-LLM 项目；Apache 2.0；OpenAI 兼容 JS |
| Jetson | "NVIDIA 边缘端" | Orin Nano / AGX / Thor / T4000 系列 |
| TRT Edge-LLM | "边缘 TensorRT" | 2026 年 TensorRT-LLM 边缘移植；EAGLE-3 + NVFP4 |
| 统一内存 | "共享池" | CPU 和 NPU 看到同一 RAM；无拷贝开销 |
| 带宽受限 | "内存受限" | Decode 受限于每秒读取权重的字节数 |
| Core ML | "Apple 转换" | ANE 原生模型的 Apple 框架 |
| QNN | "Qualcomm 栈" | Qualcomm Neural Network SDK |

## 扩展阅读

- [On-Device LLMs State of the Union 2026](https://v-chandra.github.io/on-device-llms/) — 格局和基准测试。
- [NVIDIA Jetson Edge AI](https://developer.nvidia.com/blog/getting-started-with-edge-ai-on-nvidia-jetson-llms-vlms-and-foundation-models-for-robotics/) — Orin / AGX / Thor。
- [NVIDIA TensorRT Edge-LLM](https://developer.nvidia.com/blog/accelerating-llm-and-vlm-inference-for-automotive-and-robotics-with-nvidia-tensorrt-edge-llm/) — 2026 年边缘移植公告。
- [WebLLM (arXiv:2412.15803)](https://arxiv.org/html/2412.15803v2) — 设计和基准测试。
- [Apple Core ML](https://developer.apple.com/documentation/coreml) — ANE 原生转换。
- [Qualcomm AI Hub](https://aihub.qualcomm.com/) — Hexagon 预转换模型。