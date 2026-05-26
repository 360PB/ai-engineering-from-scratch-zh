#!/usr/bin/env bash
set -euo pipefail

PYTHON_MIN_MAJOR=3
PYTHON_MIN_MINOR=11
VENV_DIR=".venv"
CORE_PACKAGES="numpy matplotlib jupyter scikit-learn pandas"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

pass() { echo -e "  ${GREEN}[通过]${NC} $1"; }
fail() { echo -e "  ${RED}[失败]${NC} $1"; }
warn() { echo -e "  ${YELLOW}[警告]${NC} $1"; }

REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$REPO_ROOT"

echo ""
echo "=== AI Engineering from Scratch: Python 环境设置 ==="
echo ""
echo "仓库根目录: $REPO_ROOT"
echo ""

HAS_UV=false
if command -v uv &> /dev/null; then
    HAS_UV=true
    pass "已找到 uv: $(uv --version)"
else
    warn "未找到 uv。安装命令: curl -LsSf https://astral.sh/uv/install.sh | sh"
    warn "回退到 python3 -m venv + pip"
fi

PYTHON_CMD=""
for cmd in python3 python; do
    if command -v "$cmd" &> /dev/null; then
        version=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || true)
        if [ -n "$version" ]; then
            major=$(echo "$version" | cut -d. -f1)
            minor=$(echo "$version" | cut -d. -f2)
            if [ "$major" -ge "$PYTHON_MIN_MAJOR" ] && [ "$minor" -ge "$PYTHON_MIN_MINOR" ]; then
                PYTHON_CMD="$cmd"
                break
            fi
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    fail "未找到 Python ${PYTHON_MIN_MAJOR}.${PYTHON_MIN_MINOR}+"
    echo ""
    echo "安装 Python ${PYTHON_MIN_MAJOR}.${PYTHON_MIN_MINOR}+:"
    echo "  uv:    uv python install 3.12"
    echo "  macOS: brew install python@3.12"
    echo "  Linux: sudo apt install python3.12 python3.12-venv"
    exit 1
fi

pass "Python: $($PYTHON_CMD --version)"

echo ""
echo "--- 创建虚拟环境 ---"
echo ""

if [ -d "$VENV_DIR" ]; then
    warn "已存在 $VENV_DIR。复用它。"
else
    if $HAS_UV; then
        uv venv "$VENV_DIR"
    else
        "$PYTHON_CMD" -m venv "$VENV_DIR"
    fi
    pass "已创建 $VENV_DIR"
fi

if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
elif [ -f "$VENV_DIR/Scripts/activate" ]; then
    source "$VENV_DIR/Scripts/activate"
else
    fail "在 $VENV_DIR 中找不到激活脚本"
    exit 1
fi

pass "已激活虚拟环境"

VENV_PYTHON="$(which python)"
if [[ "$VENV_PYTHON" != *"$VENV_DIR"* ]]; then
    fail "Python 未从 venv 运行: $VENV_PYTHON"
    exit 1
fi
pass "Python 路径: $VENV_PYTHON"

echo ""
echo "--- 安装核心包 ---"
echo ""

if $HAS_UV; then
    uv pip install $CORE_PACKAGES
else
    pip install --upgrade pip
    pip install $CORE_PACKAGES
fi

pass "已安装: $CORE_PACKAGES"

echo ""
echo "--- 验证安装 ---"
echo ""

FAILURES=0

verify_package() {
    local pkg=$1
    local import_name=${2:-$1}
    if python -c "import $import_name; print(f'  $pkg: {${import_name}.__version__}')" 2>/dev/null; then
        return 0
    else
        fail "$pkg"
        FAILURES=$((FAILURES + 1))
        return 1
    fi
}

verify_package "numpy" "numpy"
verify_package "matplotlib" "matplotlib"
verify_package "scikit-learn" "sklearn"
verify_package "pandas" "pandas"
verify_package "jupyter" "jupyter_core"

echo ""
python -c "
import numpy as np
a = np.random.randn(3, 3)
b = np.random.randn(3, 3)
c = a @ b
print(f'  矩阵乘法检查: ({a.shape}) @ ({b.shape}) = ({c.shape})')
"
pass "NumPy 运算正常"

echo ""
if python -c "import torch" 2>/dev/null; then
    TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
    CUDA_AVAIL=$(python -c "import torch; print(torch.cuda.is_available())")
    pass "PyTorch $TORCH_VERSION (CUDA: $CUDA_AVAIL)"
else
    warn "未安装 PyTorch（需要时稍后安装）:"
    echo "    uv pip install torch torchvision torchaudio"
fi

echo ""
echo "=== 摘要 ==="
echo ""
echo "  仓库根目录:    $REPO_ROOT"
echo "  虚拟环境:         $REPO_ROOT/$VENV_DIR"
echo "  Python:       $(python --version)"
echo "  包:     $CORE_PACKAGES"
echo ""

if [ "$FAILURES" -gt 0 ]; then
    fail "$FAILURES 个包验证失败"
    exit 1
else
    pass "所有检查通过"
    echo ""
    echo "在以后的会话中激活此环境:"
    echo ""
    echo "  source $REPO_ROOT/$VENV_DIR/bin/activate"
    echo ""
fi
