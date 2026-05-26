import time
import sys


def check_gpu():
    try:
        import torch
    except ImportError:
        print("PyTorch 未安装。运行: pip install torch")
        return

    print("=== GPU 检查 ===\n")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("\n未检测到 GPU。大部分课程不受影响。")
        print("对于 GPU 密集型课程，请使用 Google Colab (免费)。")
        return

    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    props = torch.cuda.get_device_properties(0)
    print(f"显存: {props.total_memory / 1e9:.1f} GB")
    print(f"计算能力: {props.major}.{props.minor}")

    print("\n=== CPU vs GPU 基准测试 ===\n")
    size = 4000

    a = torch.randn(size, size)
    b = torch.randn(size, size)

    start = time.time()
    _ = a @ b
    cpu_time = time.time() - start
    print(f"CPU 矩阵乘法 ({size}x{size}): {cpu_time:.3f}s")

    a_gpu = a.to("cuda")
    b_gpu = b.to("cuda")
    torch.cuda.synchronize()

    start = time.time()
    _ = a_gpu @ b_gpu
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    print(f"GPU 矩阵乘法 ({size}x{size}): {gpu_time:.3f}s")
    print(f"加速比: {cpu_time / gpu_time:.0f}x")

    vram_gb = props.total_memory / 1e9
    params_fp16 = vram_gb * 1e9 / 2
    params_billions = params_fp16 / 1e9
    print(f"\n估算最大模型大小 (fp16): ~{params_billions:.0f}B 参数")


if __name__ == "__main__":
    check_gpu()
