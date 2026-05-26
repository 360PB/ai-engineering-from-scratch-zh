#!/usr/bin/env bash
#
# AI 开发的 Shell 别名和函数。
# 从你的 ~/.bashrc 或 ~/.zshrc 中 source 本文件：
#   source /path/to/shell_aliases.sh

# --- GPU ---

alias gpu='nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader'
alias gpuwatch='watch -n1 nvidia-smi'
alias gpumem='nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader'
alias gpuprocs='nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv'

# --- 训练控制 ---

alias killtraining='pkill -f "python.*train"'

killtrain() {
    if [ -z "$1" ]; then
        pkill -f "python.*train"
        echo "已终止所有 Python 训练进程"
    else
        pkill -f "$1"
        echo "已终止匹配进程: $1"
    fi
}

# --- 虚拟环境 ---

alias ae='source .venv/bin/activate'
alias de='deactivate'
alias mkvenv='python -m venv .venv && source .venv/bin/activate'
alias uvvenv='uv venv && source .venv/bin/activate'

# --- 日志监视 ---

alias watchloss='tail -f logs/*.log | grep --line-buffered "loss"'
alias watchacc='tail -f logs/*.log | grep --line-buffered "accuracy\|acc"'
alias watcherr='tail -f logs/*.log | grep --line-buffered "ERROR\|error\|Exception"'

taillog() {
    local pattern="${1:-loss}"
    tail -f logs/*.log 2>/dev/null | grep --line-buffered "$pattern"
}

# --- 磁盘空间（训练数据很快填满磁盘） ---

alias diskuse='df -h .'
alias bigfiles='find . -type f -size +100M | xargs du -h 2>/dev/null | sort -rh | head -20'
alias bigmodels='find . \( -name "*.pt" -o -name "*.pth" -o -name "*.safetensors" -o -name "*.ckpt" -o -name "*.bin" \) | xargs du -h 2>/dev/null | sort -rh | head -20'

# --- 快速环境检查 ---

alias checkgpu='python -c "import torch; print(f\"CUDA: {torch.cuda.is_available()}\"); print(f\"Device: {torch.cuda.get_device_name(0)}\") if torch.cuda.is_available() else None"'
alias checkcuda='env | grep -i cuda'
alias checkenv='python --version && pip --version && python -c "import torch; print(f\"PyTorch {torch.__version__}, CUDA {torch.cuda.is_available()}\")" 2>/dev/null'

# --- tmux 快捷键 ---

alias ta='tmux attach -t'
alias tls='tmux ls'
alias tn='tmux new -s'
alias tk='tmux kill-session -t'

trainenv() {
    local name="${1:-train}"
    tmux new-session -d -s "$name"
    tmux split-window -h -t "$name"
    tmux split-window -v -t "$name"
    tmux send-keys -t "$name:0.1" 'watch -n1 nvidia-smi' C-m
    tmux send-keys -t "$name:0.2" 'htop' C-m
    tmux select-pane -t "$name:0.0"
    tmux attach -t "$name"
}

# --- SSH 助手 ---

syncto() {
    if [ -z "$1" ] || [ -z "$2" ]; then
        echo "用法: syncto <host> <remote_path> [local_path]"
        echo "示例: syncto gpu ~/data ./data"
        return 1
    fi
    local host="$1"
    local remote="$2"
    local local_path="${3:-.}"
    rsync -avz --progress "$local_path" "${host}:${remote}"
}

syncfrom() {
    if [ -z "$1" ] || [ -z "$2" ]; then
        echo "用法: syncfrom <host> <remote_path> [local_path]"
        echo "示例: syncfrom gpu ~/results ./results"
        return 1
    fi
    local host="$1"
    local remote="$2"
    local local_path="${3:-.}"
    rsync -avz --progress "${host}:${remote}" "$local_path"
}

# --- 实验管理 ---

newexp() {
    local name="${1:-experiment}"
    local dir="experiments/${name}_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$dir/logs" "$dir/checkpoints" "$dir/configs"
    echo "已创建实验目录: $dir"
    echo "$dir"
}

lastexp() {
    ls -dt experiments/*/ 2>/dev/null | head -1
}

# --- 模型下载助手 ---

hfdownload() {
    if [ -z "$1" ]; then
        echo "用法: hfdownload <model_id> [filename]"
        echo "示例: hfdownload meta-llama/Llama-2-7b config.json"
        return 1
    fi
    local model="$1"
    local file="${2:-}"
    if [ -n "$file" ]; then
        wget "https://huggingface.co/${model}/resolve/main/${file}"
    else
        echo "正在克隆完整仓库（使用 git-lfs）..."
        git lfs install
        git clone "https://huggingface.co/${model}"
    fi
}

# --- 进程管理 ---

memhogs() {
    ps aux --sort=-%mem 2>/dev/null | head -11 || ps aux -m | head -11
}

psg() {
    ps aux | grep -v grep | grep -i "$1"
}
