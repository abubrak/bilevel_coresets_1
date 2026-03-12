"""
GPU 配置自动检测模块
用于适配 Colab 单 GPU 环境和本地多 GPU 环境

本模块提供自动 GPU 检测功能，解决了原项目硬编码多 GPU 配置的问题。
在 Colab 环境中，通常只有 1 个 GPU；在本地环境中可能有多个 GPU。
此模块会自动检测并适配不同的环境。
"""
import subprocess
import os
import torch


def detect_gpu_config():
    """自动检测可用 GPU 配置

    Returns:
        dict: 包含 'NR_GPUS' (GPU数量) 和 'NR_PROCESSES' (建议进程数)
    """
    try:
        # 使用 PyTorch 检测 GPU
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            # Colab 通常只有 1 个 GPU
            if gpu_count == 1:
                return {'NR_GPUS': 1, 'NR_PROCESSES': 2}
            else:
                # 多 GPU 环境
                return {'NR_GPUS': gpu_count, 'NR_PROCESSES': min(gpu_count * 8, 32)}
        else:
            # CPU 模式
            return {'NR_GPUS': 0, 'NR_PROCESSES': 1}
    except Exception as e:
        print(f"GPU 检测失败: {e}")
        return {'NR_GPUS': 0, 'NR_PROCESSES': 1}


def get_cuda_visible_devices(process_id, total_gpus):
    """为进程分配 GPU

    Args:
        process_id (int): 进程 ID
        total_gpus (int): 总 GPU 数量

    Returns:
        str: GPU ID 字符串，如果是 CPU 模式则返回空字符串
    """
    if total_gpus == 0:
        return ''  # CPU 模式
    return str(process_id % total_gpus)


def print_gpu_info():
    """打印 GPU 信息用于调试"""
    if torch.cuda.is_available():
        print(f"可用 GPU 数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("未检测到 CUDA GPU，使用 CPU 模式")


if __name__ == "__main__":
    print_gpu_info()
    config = detect_gpu_config()
    print(f"检测到的配置: {config}")
