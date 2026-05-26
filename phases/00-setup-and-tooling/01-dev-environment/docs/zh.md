# 开发环境

> 你的工具塑造你的思维。一次性配好，配到位。

**类型：** 动手搭建
**语言：** Python、Node.js、Rust
**前置条件：** 无
**时间：** ~45 分钟

## 学习目标

- 从零安装 Python 3.11+、Node.js 20+ 和 Rust 工具链
- 配置虚拟环境和包管理器，确保构建可复现
- 用 CUDA/MPS 验证 GPU 可用性并运行张量测试
- 理解四层栈：系统层、包管理层、运行时层、AI 库层

## 问题所在

你即将用 Python、TypeScript、Rust 和 Julia 学习 200 多节 AI 工程课。如果环境有问题，每节课都会变成与工具搏斗，而不是学习。

大多数人跳过环境配置。然后花数小时调试导入错误、版本冲突和缺失的 CUDA 驱动。我们要一次性、正确地做好这件事。

## 核心概念

AI 工程环境有四层：

```mermaid
graph TD
    A["4. AI/ML 库\nPyTorch、JAX、transformers 等"] --> B["3. 语言运行时\nPython 3.11+、Node 20+、Rust、Julia"]
    B --> C["2. 包管理器\nuv、pnpm、cargo、juliaup"]
    C --> D["1. 系统基础\n操作系统、shell、git、编辑器、GPU 驱动"]
```

自下而上安装。每一层都依赖下面一层。

## 动手搭建

### 步骤 1：系统基础

检查你的系统并安装基本工具。

```bash
# macOS
xcode-select --install
brew install git curl wget

# Ubuntu/Debian
sudo apt update && sudo apt install -y build-essential git curl wget

# Windows（使用 WSL2）
wsl --install -d Ubuntu-24.04
```

### 步骤 2：Python 与 uv

我们使用 `uv` —— 比 pip 快 10-100 倍，自动处理虚拟环境。

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

uv python install 3.12

uv venv
source .venv/bin/activate  # Windows 用 .venv\Scripts\activate

uv pip install numpy matplotlib jupyter
```

验证：

```python
import sys
print(f"Python {sys.version}")

import numpy as np
print(f"NumPy {np.__version__}")
a = np.array([1, 2, 3])
print(f"向量: {a}, 自点积: {np.dot(a, a)}")
```

### 步骤 3：Node.js 与 pnpm

用于 TypeScript 课程（智能体、MCP 服务器、Web 应用）。

```bash
curl -fsSL https://fnm.vercel.app/install | bash
fnm install 22
fnm use 22

npm install -g pnpm

node -e "console.log('Node', process.version)"
```

### 步骤 4：Rust

用于性能关键课程（推理、系统层）。

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

rustc --version
cargo --version
```

### 步骤 5：Julia（可选）

用于 Julia 擅长的数学密集型课程。

```bash
curl -fsSL https://install.julialang.org | sh

julia -e 'println("Julia ", VERSION)'
```

### 步骤 6：GPU 配置（如果你有的话）

```bash
# NVIDIA
nvidia-smi

# 安装带 CUDA 的 PyTorch
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

```python
import torch
print(f"CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

没有 GPU？没问题。大多数课程在 CPU 上也能运行。对于训练密集型课程，使用 Google Colab 或云端 GPU。

### 步骤 7：全面验证

运行验证脚本：

```bash
python phases/00-setup-and-tooling/01-dev-environment/code/verify.py
```

## 学以致用

你的环境现在已为本课程的每节课做好准备。各语言使用场景如下：

| 语言 | 使用阶段 | 包管理器 |
|------|---------|---------|
| Python | 阶段 1-12（ML、DL、NLP、视觉、音频、LLM） | uv |
| TypeScript | 阶段 13-17（工具、智能体、群体、基础设施） | pnpm |
| Rust | 阶段 12、15-17（性能关键系统） | cargo |
| Julia | 阶段 1（数学基础） | Pkg |

## 交付物

本节课产出一个验证脚本，任何人都可以运行它来检查自己的环境配置。

参见 `outputs/prompt-env-check.md` 获取一个帮助 AI 助手诊断环境问题的提示词。

## 练习

1. 运行验证脚本并修复任何报错
2. 为本课程创建一个 Python 虚拟环境并安装 PyTorch
3. 用四种语言各写一个 "hello world" 并分别运行
