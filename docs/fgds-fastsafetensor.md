# FGDS + FastSafeTensors：高性能模型加载指南

## 1. 概述

[fastsafetensors](https://github.com/foundation-model-stack/fastsafetensors) 是一个针对 `safetensors` 格式优化的高性能模型加载库，利用 GPU 直访存储（GPUDirect Storage）技术实现**从 NVMe SSD 到 GPU 显存的零拷贝数据通路**，极大缩短大型 AI 模型（尤其是 LLM）的加载时间。

FGDS 提供了对 fastsafetensors 的完整支持。通过在 fastsafetensors 中集成 FGDS Copier，模型权重可以直接从 SSD 加载到 GPU 显存，无需经过 CPU 内存中转。

### 1.1 工作原理

```
传统路径（CPU 中转）：
NVMe SSD ──DMA──▶ CPU 内存 ──PCIe──▶ GPU 显存
                    ↑
                    两次拷贝 + CPU 内存瓶颈

FGDS 加速路径（GPU 直访存储）：
NVMe SSD ──GPUDirect Storage──▶ GPU 显存
                                 ↑
                                 零拷贝 + GPU 直接读取
```

### 1.2 核心组件

| 组件                 | 说明                                             |
| ------------------ | ---------------------------------------------- |
| `FgdsFileCopier`   | fastsafetensors Python 层的 FGDS 拷贝器，负责文件 I/O 调度 |
| `fgds_file_reader` | C++ 扩展层，封装 `fgds_read`/`fgds_write` 调用         |
| `FgdsDriver`       | Python 驱动管理类，封装 `fgds_open`/`fgds_close`       |
| `Fgds`             | Python 文件操作类，提供 `read`/`write`/`regmem` 等 API  |

***

## 2. 前置条件

### 2.1 系统要求

- **操作系统**：Linux (openEuler / Ubuntu / CentOS)
- **CUDA Toolkit**：>= 12.4
- **NVIDIA 驱动**：>= 535（需带内核模块源码）
- **Python**：>= 3.9
- **Python 依赖**：`torch`、`safetensors`

### 2.2 FGDS 内核模块

加载 FGDS 内核模块并验证设备就绪：

```shell
# 加载模块
bash scripts/load_fgds.sh

# 验证模块加载
lsmod | grep fgdsfs

# 验证设备节点
ls /dev/fgds_dev*
# 期望输出：/dev/fgds_dev0  /dev/fgds_dev1  ...

```
详细构建步骤请参考 [Getting Started](getting-started.md)。

## 3. 部署 fastsafetensors

### 3.1 环境准备

```shell
# 克隆一个干净的 conda 环境
conda create --name fastsafetensor python=3.10 -y
conda activate fastsafetensor

# 安装 PyTorch 和 CUDA 依赖
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install safetensors numpy
```

### 3.2 获取 fastsafetensors 源码

```shell
# 克隆 fastsafetensors 仓库
git clone https://github.com/foundation-model-stack/fastsafetensors
cd fastsafetensors
```

### 3.3 应用 FGDS 补丁

FGDS 提供了一个 patch，为 fastsafetensors 添加 FGDS Copier 支持。patch 文件位于 FGDS 项目根目录：

```shell
# 进入 fastsafetensors 项目根目录
cd /path/to/fastsafetensors

# 应用 FGDS patch（从 FGDS 项目根目录获取）, 当前仅为示例用法，上游 fastsafetensors PR正在推进中
git am /path/to/fgds/0001-add-a-new-copier-of-fgds-to-use-gpu-direct-storage-a.patch
```

该 patch 会新增以下文件：

| 文件                                 | 说明                      |
| ---------------------------------- | ----------------------- |
| `fastsafetensors/copier/fgds.py`   | FGDS Python 层 Copier 实现 |
| `fastsafetensors/cpp/fgds_ext.cpp` | C++ 扩展层，封装 libfgds 调用   |
| `fastsafetensors/cpp/fgds_ext.pyi` | C++ 扩展的 Python 类型存根     |
| `examples/test_fgds.py`            | 功能测试脚本                  |

同时会修改 `fastsafetensors/loader.py`，在 `SafeTensorsFileLoader` 中加入 `USE_FGDS` 环境变量的自动检测逻辑。

### 3.4 安装 FGDS Python 绑定

在 fastsafetensors 仓库外，先安装 FGDS 的 Python 包：

```shell
cd /path/to/fgds/python
pip install -e .
```

### 3.5 编译 C++ 扩展并安装

```shell
cd /path/to/fastsafetensors

# 编译 C++ 扩展（需要 libfgds.so 在 LD_LIBRARY_PATH 中）
export LD_LIBRARY_PATH=/path/to/fgds/build/lib:$LD_LIBRARY_PATH 或者cp /path/to/fgds/build/libfgds.so /usr/lib64/
python setup.py build_ext --inplace

# 安装 fastsafetensors
pip install .
```

***

## 4. 使用方法

### 4.1 基本使用

下面的示例展示了如何使用 FGDS 通过 fastsafetensors 加载模型：

```python
import os
import torch
from fastsafetensors import fastsafe_open

# 准备 safetensors 文件列表
model_dir = "/data/models/llama-7b"
filenames = sorted([
    os.path.join(model_dir, f)
    for f in os.listdir(model_dir)
    if f.endswith(".safetensors")
])

# 使用 FGDS 加速加载
# 设置环境变量 USE_FGDS=1 来启用 FGDS Copier
# 该变量由 fastsafetensors 的 SafeTensorsFileLoader 读取，
# 当 USE_FGDS 为 'true'/'1'/'yes' 且设备非 CPU 时，自动选择 "fgds" copier
os.environ['USE_FGDS'] = '1'

# 通过 context manager 打开并加载
with fastsafe_open(filenames=filenames, device="cuda:0") as f:
    keys = f.keys()
    print(f"Found {len(keys)} tensors")

    tensors = {}
    for key in keys:
        tensors[key] = f.get_tensor(key)

print("Model loaded successfully!")
```

### 4.2 Copier 选择机制

fastsafetensors 的 `SafeTensorsFileLoader` 通过 `USE_FGDS` 环境变量自动选择 Copier：

```python
# loader.py 中的选择逻辑（简化）
use_fgds = os.environ.get('USE_FGDS', '').lower() in ('true', '1', 'yes') and self.device != "cpu"
copier_type = "fgds" if use_fgds else "gds"  # FGDS 优先，否则回退到标准 GDS
```

选择规则：

| 条件                    | Copier 类型    | 说明                                          |
| --------------------- | ------------ | ------------------------------------------- |
| `USE_FGDS=1` 且设备非 CPU | `fgds`       | 使用 FGDS 加速路径                                |
| `USE_FGDS` 未设置        | `gds`        | 使用标准 GPUDirect Storage                      |
| `nogds=True`          | `nogds`      | 强制跳过 GDS/FGDS，走 CPU 中转                      |
| FGDS 初始化失败            | 自动回退 `nogds` | `new_fgds_file_copier` 内部检测 FGDS 可用性，不可用时回退 |

### 4.3 对比测试

使用 patch 中附带的测试脚本进行性能对比：

```shell
# 不使用 FGDS（走标准 GDS 路径）
python3 examples/test_fgds.py

# 使用 FGDS（GPU 直访存储加速路径）
USE_FGDS=1 python3 examples/test_fgds.py
```

### 4.4 关键参数说明

| 参数            | 默认值        | 说明                                   |
| ------------- | ---------- | ------------------------------------ |
| `filenames`   | -          | safetensors 文件路径列表                   |
| `device`      | `"cuda:0"` | 目标 GPU 设备                            |
| `nogds`       | `False`    | 设为 `True` 则强制跳过 GDS/FGDS，使用标准 CPU 路径 |
| `max_threads` | `16`       | 并发读取的线程数（由 `FgdsFileCopier` 使用）      |

### 4.5 设备 ID 映射

`FgdsFileCopier` 会通过以下逻辑自动推导 FGDS `device_id`：

```python
device_list = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
idx = device.index if device.index is not None else 0
self.device_id = int(device_list.split(",")[idx])
```

如果 `CUDA_VISIBLE_DEVICES` 未设置，`device_id` 默认为 `0`（第一块 GPU）。

***

## 5. 运行原理详解

### 5.1 加载流程

```
fastsafe_open(filenames, device="cuda:0")
    │
    ▼
SafeTensorsFileLoader.__init__()
    │
    ├── 读取 USE_FGDS 环境变量
    ├── 根据 copier_type="fgds" 创建 new_fgds_file_copier()
    │
    ▼
new_fgds_file_copier() (注册为 "fgds")
    │
    ├── init_fgds()                 # 加载 libfgds.so
    ├── 检测 FGDS 可用性（临时文件测试 fgds_open/fgds_close）
    ├── 创建 fgds_file_reader(max_threads, device_id)
    ├── 返回 construct_copier 工厂函数
    │
    ▼
FgdsFileCopier.__init__()  (每个文件的拷贝器)
    │
    ├── fgds_file_handle(path, o_direct, device_id)
    │
    ▼
FgdsFileCopier.submit_io()  (提交异步读取)
    │
    ├── 64KB 对齐计算
    ├── alloc_tensor_memory()           # 分配 GPU 显存
    ├── fgds_regmem()                   # 注册显存（建立 GPU↔SSD 直访映射）
    ├── 按 max_copy_block_size 切分请求
    ├── fgds_file_reader.submit_read()  # 提交读请求（异步）
    │
    ▼
[其他文件并行加载 / 模型计算]           # IO 与计算重叠
    │
    ▼
FgdsFileCopier.wait_io()  (等待并回收)
    │
    ├── fgds_file_reader.wait_read()    # 等待所有读请求完成
    ├── fgds_deregmem()                 # 注销显存
    └── 返回张量数据
```

### 5.2 io\_uring 异步流水线

对于大 IO（≥512KB），FGDS 使用 io\_uring 实现多子 IO 并发流水线：

```
┌──────────────────────────────────────────────────────┐
│ fgds_read (nbyte ≥ 512KB)                            │
│                                                      │
│  外层 while (done < nbyte)                           │
│  ├── 内层 while 填充 ≤256 个 SQE                      │
│  │   └── 每个子 IO = 256KB                           │
│  ├── io_uring_submit() 提交整批                       │
│  └── io_uring_wait_cqe() 逐个等待完成                │
│                                                      │
│  小 IO (<512KB) 回退:                                │
│  └── pread() 单次读取（避免 io_uring 开销）            │
└──────────────────────────────────────────────────────┘
```

### 5.3 fgds\_file\_reader 异步接口

`fgds_file_reader` 是 C++ 扩展层提供的异步读取器，封装了 `fgds_read` 的调用：

```python
# C++ 扩展层 API
class fgds_file_reader:
    def __init__(self, max_threads: int, device_id: int):
        """初始化读取器，启动 max_threads 个工作线程"""

    def submit_read(self, fgds_file_handle, dev_buffer, offset, length, req_id):
        """提交异步读请求，立即返回 req_id"""

    def wait_read(self, req_id):
        """等待指定请求完成，返回 0 表示成功"""
```

### 5.4 显存管理

FGDS 通过 `fgds_regmem` 将 GPU 显存映射为可被 SSD 直接访问的虚拟地址：

```python
# 分配 GPU 显存
gbuf = self.framework.alloc_tensor_memory(aligned_length, device)
gbuf_ptr = gbuf.get_base_address()

# 注册显存（建立直访映射）
fgds_regmem(device_id, gbuf_ptr, size, &target_addr)
# target_addr 即为 SSD 可直接访问的地址

# 使用完成后注销
fgds_deregmem(device_id, gbuf_ptr, size)
```

***

## 6. 故障排查

### 6.1 `fgds_open` 失败

```
RuntimeError: open fgds error -1
```

- **原因**：FGDS 内核模块未加载，或设备节点不存在
- **排查**：

```shell
lsmod | grep fgdsfs
ls /dev/fgds_dev*
```

- **解决**：执行 `bash scripts/load_fgds.sh` 加载模块

### 6.2 `io_uring_get_sqe failed`

```
fgds_read: io_uring_get_sqe failed
```

- **原因**：io\_uring submission queue 已满，通常发生在多线程高并发场景
- **解决方案**：
  - **代码层面已加入修复机制**：最新版本在 `libfgds/fgds.cc` 的 `fgds_read`/`fgds_write` 中加入了互斥锁保护和 SQE 溢出自动重试逻辑（需重新编译 libfgds.so 生效）
  - 如果问题仍然存在，可降低 `max_threads` 或 `max_copy_block_size`
  - 作为临时规避，可设置 `USE_FGDS=0` 回退到标准 GDS 路径

### 6.3 `fgds_regmem` 失败

```
fgds_regmem error: Cannot allocate memory
```

- **原因**：GPU 显存不足，或注册地址未 64KB 对齐
- **排查**：

```shell
nvidia-smi                            # 查看显存使用情况
# 确保分配的大小是 64KB (65536) 的整数倍
```

### 6.4 性能异常低下

- **检查存储介质**：确保使用 NVMe SSD，不支持机械硬盘
- **检查文件系统**：需要支持 `O_DIRECT`（推荐 ext4 / xfs）
- **检查对齐**：模型文件需要 64KB 对齐读取，`safetensors` 格式天然对齐良好
- **关闭系统缓存干扰**：

```shell
sync && echo 3 > /proc/sys/vm/drop_caches
```

### 6.5 CUDA 设备 ID 不匹配

- **问题**：`CUDA_VISIBLE_DEVICES` 设置导致 `device_id` 映射错误
- **解决方案**：确保 `CUDA_VISIBLE_DEVICES` 与实际使用的 GPU 对应

```shell
# 如果只使用 GPU 0
export CUDA_VISIBLE_DEVICES=0

# 如果使用 GPU 2
export CUDA_VISIBLE_DEVICES=2
```

***

## 7. 性能验证

### 7.1 测试环境

| 项目   | 配置             |
| ---- | -------------- |
| GPU  | NVIDIA H100    |
| CPU  | Hygon C86 7490 |
| 存储   | NVMe SSD       |
| 模型   | Qwen-32B       |
| 模型大小 | \~64GB (BF16)  |

### 7.2 测试结果

使用 FGDS 通过 fastsafetensors 加载 Qwen-32B 模型，相比传统 CPU 中转路径获得 **2.6 倍** 性能提升：

| 路径                 | 加载延迟    | 说明                         |
| ------------------ | ------- | -------------------------- |
| CPU 中转（无 GDS/FGDS） | **31s** | SSD → CPU 内存 → GPU 显存，两次拷贝 |
| FGDS 加速（GPU 直访存储）  | **12s** | SSD 直通 GPU 显存，零拷贝          |

### 7.3 性能提升来源

- **零拷贝数据通路**：模型权重从 NVMe SSD 直接写入 H100 显存，跳过 CPU 内存中转
- **异步 IO 流水线**：`io_uring` 将大文件切分为 256KB 子 IO 并行提交，充分利用 SSD 并发能力

***

## 8. 性能调优建议

### 8.1 线程数调整

`max_threads` 控制并发读取线程数。默认 16，建议根据存储 IOPS 能力调整：

```python
# NVMe Gen4 (高性能 SSD): max_threads=16~32
# NVMe Gen3 (标准 SSD):  max_threads=8~16
# SATA SSD (低速):       max_threads=4~8
```

### 8.2 文件预对齐

`safetensors` 格式按张量存储，天然适合并行读取。建议：

- 单个 safetensors 文件大小 ≥ 128MB 时效果最佳
- 避免大量小文件（< 1MB），合并为大文件

### 8.3 系统调优

```shell
# 设置 CPU 调度策略（避免中断）
bash scripts/set_cpu_freq.sh

# 使用 irqbalance 或手动绑定中断到特定 CPU 核心
# 确保 NVMe SSD 中断和 GPU 中断不冲突
```

***

## 9. 参考链接

| 资源                   | 链接                                                                                                             |
| -------------------- | -------------------------------------------------------------------------------------------------------------- |
| fastsafetensors 上游   | [github.com/foundation-model-stack/fastsafetensors](https://github.com/foundation-model-stack/fastsafetensors) |
| FGDS Getting Started | [getting-started.md](getting-started.md)                                                                       |
| FGDS Python API      | [python/README.md](../python/README.md)                                                                        |
| 故障排查                 | [troubleshooting.md](troubleshooting.md)                                                                       |
| libfgds API          | [libfgds.md](libfgds.md)                                                                                       |

