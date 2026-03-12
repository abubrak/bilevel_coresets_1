# Google Colab 使用指南

## 环境设置

### 1. 选择 GPU 运行时
- 菜单: Runtime → Change runtime type → Hardware accelerator → GPU

### 2. 安装依赖
运行 demo.ipynb 的前几个单元格，会自动：
- 安装 PyTorch
- 检测 CUDA 版本
- 安装匹配的 JAX

### 3. 验证安装
```python
import torch
import jax
print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
print(f"JAX: {jax.__version__}, Devices: {jax.devices()}")
```

## 运行示例

### 回归任务
直接运行 demo.ipynb 的回归部分，无需 GPU。

### MNIST 分类
运行 demo.ipynb 的 MNIST 部分，需要 GPU：
- limit=2500 (快速测试)
- 修改为 limit=60000 (完整训练)

## 故障排除

### JAX 安装失败
确保 CUDA 版本匹配，手动安装：
```bash
# CUDA 12.x
!pip install jax jaxlib[cuda12]

# CUDA 11.x
!pip install jax jaxlib[cuda11]
```

### CUDA OOM
减小 batch size 或数据集大小：
```python
limit = 1000  # 从 2500 减小
candidate_batch_size = 500  # 从 1000 减小
```

### 常见错误及解决方案

#### 错误 1: `ImportError: cannot import name 'jit' from 'jax.api'`
**原因**: JAX 0.3+ 的 API 变化
**解决**: 已通过修改导入语句修复 (`from jax import jit`)

#### 错误 2: `CUDA SETUP failed`
**原因**: JAX CUDA 版本不匹配
**解决**: 已通过自动检测 CUDA 版本修复

#### 错误 3: `RuntimeError: CUDA out of memory`
**原因**: NTK 计算消耗大量内存
**解决**: 减小 `limit` 或 `candidate_batch_size`

## GPU 配置说明

项目已自动适配 Colab 单 GPU 环境：
- 自动检测 GPU 数量
- 自动调整进程数
- 支持 CPU 降级模式

详情参考 `gpu_config.py` 模块。
