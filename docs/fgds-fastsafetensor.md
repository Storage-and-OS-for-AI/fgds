# FGDS + FastSafeTensors：高性能模型加载指南

## 1. 概述

[fastsafetensors](https://github.com/foundation-model-stack/fastsafetensors) 是一个针对 `safetensors` 格式优化的高性能模型加载库，vLLM和SGLang可以通过 `--load-format fastsafetensors` 来加速大模型文件的加载。

FGDS 为 fastsafetensors 提供了完整支持。通过在 fastsafetensors 中集成 FGDS Copier，模型权重可直接从 SSD 加载到 GPU 显存，无需经过 CPU 内存中转，实现极致的加载性能。FGDS 同时支持 **NVIDIA CUDA** 和 **AMD ROCm/HIP** 两大GPU平台。

### 1.1 工作原理

```
传统路径（CPU 中转）：
NVMe SSD ──DMA──▶ CPU 内存 ──PCIe──▶ GPU 显存
                    ↑
                    两次数据拷贝 + CPU 内存带宽瓶颈

FGDS 加速路径（GPU 直访存储）：
NVMe SSD ──GPUDirect Storage──▶ GPU 显存
                                 ↑
                                 零拷贝 + GPU 直接读取
                                 （支持 NVIDIA CUDA / AMD HIP）
```

### 1.2 核心组件

| 组件                 | 说明                                             |
| ------------------ | ---------------------------------------------- |
| `FgdsFileCopier`   | fastsafetensors Python 层的 FGDS 拷贝器，负责文件 I/O 调度与管理 |
| `fgds_file_reader` | C++ 扩展层，封装 `fgds_read`/`fgds_write` 底层调用，支持多线程异步读取  |
| `fgds_ext`         | pybind11 C++ 扩展模块，动态加载 `libfgds.so` 并提供 Python 绑定 |
| `SafeTensorsFileLoader` | fastsafetensors 主加载器，根据参数自动选择合适的 Copier |

***

## 2. 前置条件

### 2.1 系统要求

- **操作系统**：Linux（openEuler / Ubuntu / CentOS）
- **GPU**:
  - NVIDIA: H100/A100/L40 等支持 GPUDirect Storage 的 GPU
  - AMD: Radeon 7950/Instinct MI 系列等支持 ROCm 的 GPU
- **存储**: NVMe SSD（PCIe 接口，支持 O_DIRECT）
- **Python**：≥ 3.9
- **Python 依赖**：`torch`、`safetensors`、`pybind11`

### 2.2 平台支持矩阵

| 平台 | GPU 运行时 | 支持状态 | 自动回退 |
| ---- | -------- | ---- | ---- |
| Linux + NVIDIA | libcudart.so (CUDA) | ✅ 完全支持 | 回退到 nogds |
| Linux + AMD | libamdhip64.so (ROCm/HIP) | ✅ 完全支持 | 回退到 nogds |
| Linux + CPU | - | ⚠️ 自动回退到 nogds | 回退到 nogds |
| Windows | - | ❌ 不支持 | 回退到 nogds/dstorage |

### 2.3 FGDS 内核模块

加载 FGDS 内核模块并验证设备就绪状态：

```shell
# 加载内核模块
bash scripts/load_fgds.sh

# 验证模块是否成功加载
lsmod | grep fgdsfs

# 验证设备节点是否存在
ls /dev/fgds_dev*
# 期望输出：/dev/fgds_dev0  /dev/fgds_dev1  ...
```

详细构建与安装步骤请参考 [Getting Started](getting-started.md)。

## 3. 部署 fastsafetensors

### 3.1 环境准备

#### 3.1.1 NVIDIA CUDA 环境

```shell
# 创建独立的 conda 环境
conda create --name fastsafetensor python=3.10 -y
conda activate fastsafetensor

# 安装 PyTorch 及 CUDA 相关依赖
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install safetensors numpy pybind11
```

#### 3.1.2 AMD ROCm/HIP 环境

```shell
# 创建独立的 conda 环境
conda create --name fastsafetensor python=3.10 -y
conda activate fastsafetensor

# 安装 PyTorch 及 ROCm 相关依赖
pip install torch --index-url https://download.pytorch.org/whl/rocm7.2
pip install safetensors numpy pybind11
```

### 3.2 获取 fastsafetensors 源码

```shell
# 克隆 fastsafetensors 仓库
git clone https://github.com/foundation-model-stack/fastsafetensors
cd fastsafetensors
```

### 3.3 应用 FGDS 补丁

FGDS 提供了一个补丁文件，为 fastsafetensors 添加 FGDS Copier 支持。补丁文件位于 FGDS 项目的 `python/` 目录下：

```shell
# 进入 fastsafetensors 项目根目录
cd /path/to/fastsafetensors

# 应用 FGDS 补丁（从 FGDS 项目 python 目录获取）
git am /path/to/fgds/python/0001-Add-FGDS-an-alternative-solution-of-GDS-copier.patch
```
该补丁当前正在提交到 fastsafetensors 项目的 PR #101 中。

该补丁会新增以下文件：

| 文件                                 | 说明                      |
| ---------------------------------- | ----------------------- |
| `fastsafetensors/copier/fgds.py`   | FGDS Python 层 Copier 实现 |
| `fastsafetensors/cpp/fgds_ext.cpp` | C++ 扩展层，封装 libfgds 调用   |
| `fastsafetensors/fgds_ext.pyi`     | C++ 扩展的 Python 类型存根文件     |

同时会修改以下文件：
- `fastsafetensors/loader.py`: 添加 `use_fgds` 参数支持
- `fastsafetensors/parallel_loader.py`: 透传 `use_fgds` 参数
- `setup.py`: 添加 `fgds_ext` 编译目标
- `fastsafetensors/copier/__init__.py`: 注册 `FgdsFileCopier`
- `docs/configuration.md`: 添加 FGDS 配置说明
- `tests/unit/`: 添加 FGDS 单元测试

### 3.4 编译并安装 libfgds

确保已按照 FGDS 文档编译安装 `libfgds.so`：

```shell
cd /path/to/fgds
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
# libfgds.so 将生成在 build/lib/ 目录下
```

### 3.5 安装 FGDS Python 绑定

```shell
cd /path/to/fgds/python
pip install -e .
```

### 3.6 编译 C++ 扩展并安装 fastsafetensors

```shell
cd /path/to/fastsafetensors

# 设置 LD_LIBRARY_PATH 环境变量指向 libfgds.so
export LD_LIBRARY_PATH=/path/to/fgds/build/lib:$LD_LIBRARY_PATH

# 对于 ROCm 环境，确保 ROCm 库路径也在 LD_LIBRARY_PATH 中
# export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH

# 编译 C++ 扩展（fgds_ext 会自动检测平台并编译）
python setup.py build_ext --inplace

# 安装 fastsafetensors
pip install .
```

***

## 4. 使用方法

### 4.1 API 使用方式

最新版本使用 `use_fgds=True` 参数显式启用 FGDS，不再依赖环境变量：

#### 4.1.1 使用 fastsafe_open 接口

```python
from fastsafetensors import fastsafe_open

# 使用 FGDS 加速加载
with fastsafe_open(
    filenames=["model-00001-of-00004.safetensors"],
    device="cuda:0",  # NVIDIA 和 AMD GPU 均使用 "cuda:0" 格式（PyTorch 兼容）
    use_fgds=True,   # 启用 FGDS
    nogds=False,
) as f:
    for key in f.keys():
        tensor = f.get_tensor(key)
        # 处理 tensor
```

#### 4.1.2 使用 SafeTensorsFileLoader 接口

```python
from fastsafetensors import SafeTensorsFileLoader

loader = SafeTensorsFileLoader(
    pg=None,
    device="cuda:0",
    use_fgds=True,       # 启用 FGDS
    max_threads=16,
)
loader.add_filenames({0: ["model-00001-of-00004.safetensors"]})
bufs = loader.copy_files_to_device()

# 获取张量
tensor = bufs.get_tensor_wrapped("model.layers.0.self_attn.q_proj.weight")
```

#### 4.1.3 使用配置文件选择 FGDS

```json
{
  "loader": "base",
  "base": {
    "copier_type": "fgds",
    "max_threads": 16
  }
}
```

### 4.2 命令行测试脚本

FGDS 提供了测试脚本 `python/test/test_fastsafetensors_copier.py` 用于性能对比测试：

```shell
# 进入 FGDS Python 测试目录
cd /path/to/fgds/python/test

# 单模式测试 - 使用 FGDS
python test_fastsafetensors_copier.py \
    --model-dir /data/Qwen3-0.6B \
    --device "cuda:0" \
    --use-fgds \
    --drop-cache

# 单模式测试 - 使用 GDS（对比基准）
python test_fastsafetensors_copier.py \
    --model-dir /data/Qwen3-0.6B \
    --device "cuda:0" \
    --drop-cache

# 对比模式 - 同时运行 GDS 和 FGDS 并比较性能
python test_fastsafetensors_copier.py \
    --model-dir /data/Qwen3-0.6B \
    --device "cuda:0" \
    --compare \
    --drop-cache
```

**AMD GPU 测试说明**：在 AMD ROCm 环境下，`--device` 参数同样使用 `"cuda:0"` 格式（PyTorch ROCm 版本兼容该命名），测试脚本无需修改即可运行。

### 4.3 运行单元测试

```shell
cd /path/to/fastsafetensors/tests/unit

# 运行 FGDS 相关单元测试
pytest -v test_fastsafetensors.py -k "fgds or FgdsFileCopier" -s

# 运行所有 copier 测试
pytest -v test_fastsafetensors.py -k "Copier or copier" -s

# 运行 AutoLoader 配置测试
pytest -v test_auto_loader.py -k "fgds" -s
```

### 4.4 Copier 选择机制

fastsafetensors 根据参数自动选择合适的 Copier，优先级如下：

```python
# loader.py 中的选择逻辑（简化版）
if nogds:
    copier_type = "unified"  # 统一内存系统（如 DGX Spark）
    # 或
    copier_type = "nogds"    # 传统 CPU 中转路径
elif use_fgds:
    copier_type = "fgds"     # FGDS GPU 直访存储
elif platform.system() == "Windows":
    copier_type = "dstorage" # Windows DirectStorage
else:
    copier_type = "gds"      # NVIDIA cuFile GDS
```

选择规则：

| 参数组合                    | Copier 类型    | 说明                                          |
| ------------------------- | ------------ | ------------------------------------------- |
| `use_fgds=True`, `nogds=False` | `fgds`       | 使用 FGDS 加速路径（支持 NVIDIA/AMD）|
| `nogds=True`              | `nogds`/`unified` | 强制跳过 GDS/FGDS，走传统 CPU 中转路径或统一内存路径   |
| 默认（不指定）                 | `gds`        | 使用 NVIDIA cuFile GDS（仅 NVIDIA 平台）|
| FGDS/GDS 初始化失败            | 自动回退 `nogds` | 内部检测可用性，不可用时自动回退 |

可用的 `copier_type` 值：

| 值          | 说明                                                                 |
| ---------- | ------------------------------------------------------------------ |
| `"gds"`    | NVIDIA GPUDirect Storage via cuFile（默认）|
| `"fgds"`   | FGDS 替代实现，支持 NVIDIA 和 AMD GPU via `libfgds.so`                              |
| `"nogds"`  | Bounce-buffer pread 路径，无 GPU Direct；FGDS/GDS 不可用时自动回退                     |
| `"unified"`| 统一内存 copier，用于 CPU/GPU 共享内存系统（如 DGX Spark）；`nogds` 请求时在统一内存主机上自动选择  |
| `"dstorage"`| DirectStorage 后端（仅 Windows）                                        |

### 4.5 关键参数说明

| 参数            | 默认值        | 说明                                   |
| ------------- | ---------- | ------------------------------------ |
| `filenames`   | -          | safetensors 文件路径列表（必填）                   |
| `device`      | `"cuda:0"` | 目标 GPU 设备（NVIDIA/AMD 均使用 cuda:N 格式）|
| `use_fgds`    | `False`    | 设为 `True` 启用 FGDS GPU 直访存储加速        |
| `nogds`       | `False`    | 设为 `True` 则强制跳过 GDS/FGDS，使用标准 CPU 路径（优先级高于 use_fgds） |
| `max_threads` | `16`       | 并发读取线程数（由 `FgdsFileCopier` 使用）      |

### 4.6 设备 ID 映射

`FgdsFileCopier` 通过以下逻辑自动推导 FGDS `device_id`，同时支持 `CUDA_VISIBLE_DEVICES` 和 `HIP_VISIBLE_DEVICES` 环境变量：

```python
def _get_device_id(device: Device) -> int:
    # 优先使用 CUDA_VISIBLE_DEVICES（PyTorch ROCm 也兼容该变量）
    device_list = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    idx = device.index if device.index is not None else 0
    return int(device_list.split(",")[idx])
```

如果未设置 `CUDA_VISIBLE_DEVICES`，`device_id` 默认为 `0`（即第一块 GPU）。

***

## 5. 运行原理详解

### 5.1 加载流程

```
fastsafe_open(filenames, device="cuda:0", use_fgds=True)
    │
    ▼
SafeTensorsFileLoader.__init__()
    │
    ├── use_fgds=True → copier_type="fgds"
    ├── create_copier_constructor("fgds", ...)
    │
    ▼
new_fgds_file_copier()（注册为 "fgds"）
    │
    ├── 平台检查：仅 Linux 支持，其他平台回退 nogds
    ├── init_fgds()                     # 加载 GPU 运行时（libcudart/libamdhip64）
    ├── 检测 GPU 是否可用（is_gpu_found()）
    ├── 可用性探测：通过临时文件测试 fgds_file_handle 能否正常打开
    │   └── 失败则自动回退到 nogds
    ├── 创建 fgds_file_reader(max_threads, device_id)
    ├── 返回 construct_copier 工厂函数
    │
    ▼
FgdsFileCopier.__init__()（每个文件对应一个拷贝器实例）
    │
    ├── fgds_open(device_id)            # 打开 FGDS 设备
    │
    ▼
FgdsFileCopier.submit_io()（提交异步读取请求）
    │
    ├── 64KB 地址对齐计算（O_DIRECT 要求）
    ├── alloc_tensor_memory()               # 分配 GPU 显存（CUDA/HIP）
    ├── fgds_file_handle(path, O_DIRECT, device_id)  # 以 O_DIRECT 打开文件
    ├── fgds_regmem()                       # 注册显存（建立 GPU↔SSD 直访映射）
    ├── 按 max_copy_block_size 切分读请求
    ├── fgds_file_reader.submit_read()      # 提交多线程异步读请求
    │
    ▼
[其他文件并行加载 / 模型计算]               # I/O 与计算可重叠执行
    │
    ▼
FgdsFileCopier.wait_io()（等待并回收资源）
    │
    ├── fgds_file_reader.wait_read()        # 等待所有读请求完成
    ├── fgds_deregmem()                     # 注销显存映射
    └── 返回张量数据
```

### 5.2 fgds_ext C++ 扩展架构

`fgds_ext` 通过 `dlopen` 动态加载 `libfgds.so`，支持跨平台兼容：

```cpp
// 动态加载 libfgds.so
void* lib = dlopen("libfgds.so", RTLD_LAZY | RTLD_GLOBAL | RTLD_NODELETE);

// 加载 FGDS 核心函数
dlsym(lib, "fgds_open");
dlsym(lib, "fgds_close");
dlsym(lib, "fgds_regmem");
dlsym(lib, "fgds_deregmem");
dlsym(lib, "fgds_read");
```

主要 C++ 类：

| 类名                    | 说明                          |
| --------------------- | --------------------------- |
| `fgds_file_handle`    | 文件句柄封装，负责 O_DIRECT 打开/关闭文件 |
| `fgds_device_buffer`  | GPU 显存缓冲区描述符                |
| `fgds_file_reader`    | 多线程异步读取器，支持并发 I/O 提交        |
| `FgdsWrapper`         | libfgds.so 动态加载封装，引用计数管理    |

Windows 平台提供 stub 实现，Python 层检测到缺少 FGDS 符号时自动回退到 nogds。

### 5.3 显存管理

FGDS 通过 `fgds_regmem` 将 GPU 显存映射为可被 SSD 直接访问的虚拟地址，抽象层同时兼容 CUDA 和 HIP：

```python
# 分配 GPU 显存（由框架层抽象，PyTorch 适配 CUDA/HIP）
gbuf = self.framework.alloc_tensor_memory(aligned_length, device)
gbuf_ptr = gbuf.get_base_address()

# 注册显存（建立 SSD 直访映射）
result = fgds_regmem(device_id, gbuf_ptr, aligned_length, None)
if result != 0:
    warnings.warn(f"fgds_regmem failed with error code: {result}")

# 读取完成后注销映射
result = fgds_deregmem(device_id, gbuf_ptr, aligned_length)
```

**注意**：O_DIRECT 要求内存地址和偏移按 4KB/64KB 对齐，`FgdsFileCopier` 内部自动处理对齐逻辑。

***

## 6. 故障排查

### 6.1 `fgds_open` 失败

```
RuntimeError: open fgds error -1
UserWarning: fgds_open failed with error code: -1
```

- **原因**：FGDS 内核模块未加载，或设备节点不存在
- **排查步骤**：

```shell
lsmod | grep fgdsfs
ls /dev/fgds_dev*
```

- **解决方案**：执行 `bash scripts/load_fgds.sh` 加载内核模块

### 6.2 FGDS 自动回退到 nogds

如果看到警告信息：
```
UserWarning: FGDS is not available: ... Falling back to NoGDS.
```

可能原因及排查：

1. **libfgds.so 未找到**：
   ```shell
   # 检查 LD_LIBRARY_PATH
   echo $LD_LIBRARY_PATH
   ls /path/to/fgds/build/lib/libfgds.so
   ```

2. **GPU 运行时未找到**：
   - NVIDIA: 检查 `libcudart.so` 是否在库路径中
   - AMD: 检查 `libamdhip64.so` 是否在库路径中（通常在 `/opt/rocm/lib`）

3. **非 Linux 平台**：FGDS 仅支持 Linux，Windows 下自动回退

### 6.3 `fgds_regmem` 失败

```
fgds_regmem failed with error code: ...
```

- **原因**：GPU 显存不足，或注册地址未按 64KB 对齐
- **排查步骤**：

```shell
# NVIDIA
nvidia-smi                            # 查看 GPU 显存使用情况

# AMD
rocm-smi                              # 查看 AMD GPU 显存使用情况
```

- **解决方案**：
  - 确保分配的内存大小是 64KB (65536) 的整数倍（FgdsFileCopier 已自动处理）
  - 释放不必要的 GPU 显存占用
  - 降低 `max_copy_block_size` 参数值

### 6.4 性能异常低下

若加载性能未达预期，请按以下步骤排查：

- **检查存储介质**：确保使用 NVMe SSD 且通过 PCIe 插槽接入
- **检查文件系统**：需支持 `O_DIRECT`（推荐使用 ext4 或 xfs 文件系统）
- **清理系统缓存干扰**：

```shell
sync && echo 3 > /proc/sys/vm/drop_caches
```

- **检查 GPU 驱动**：
  - NVIDIA: 确保 CUDA 驱动版本与 PyTorch 匹配
  - AMD: 确保 ROCm 版本与 PyTorch 匹配，运行 `rocminfo` 验证

### 6.5 设备 ID 不匹配

- **问题**：`CUDA_VISIBLE_DEVICES` 设置不当导致 `device_id` 映射错误
- **解决方案**：确保 `CUDA_VISIBLE_DEVICES` 与实际使用的 GPU 对应

```shell
# 如果只使用 GPU 0
export CUDA_VISIBLE_DEVICES=0

# 如果使用 GPU 2（物理ID）
export CUDA_VISIBLE_DEVICES=2
```

**注意**：ROCm 环境下 PyTorch 同样使用 `CUDA_VISIBLE_DEVICES` 环境变量控制设备可见性。


## 7. 性能验证

### 7.1 测试环境（NVIDIA H100）

| 项目   | 配置             |
| ---- | -------------- |
| GPU  | NVIDIA H100 80GB |
| CPU  | Hygon C86 7490 |
| 存储   | NVMe Gen4 SSD   |
| 模型   | Qwen-32B       |
| 模型大小 | ~64GB (BF16)  |

### 7.2 测试结果

使用 FGDS 通过 fastsafetensors 加载 Qwen-32B 模型：

| 路径                 | 加载延迟    | 性能提升 | 说明                         |
| ------------------ | ------- | ---- | -------------------------- |
| CPU 中转（无 GDS/FGDS） | 31s     | 1.0x | SSD → CPU 内存 → GPU 显存，两次数据拷贝 |
| FGDS 加速（GPU 直访存储）  | **12s** | 2.6x | SSD 直通 GPU 显存，零拷贝数据通路          |

FGDS 微基准测试结果：相比 GDS 性能提升 11%~109%，相比 POSIX 提升 40%~143%。

### 7.3 AMD 平台测试

在 AMD GPU 平台上使用相同的测试脚本即可进行验证：

```shell
# 验证 ROCm 环境
rocminfo
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

# 运行 FGDS 测试
python test_fastsafetensors_copier.py \
    --model-dir /path/to/model \
    --device "cuda:0" \
    --use-fgds \
    --compare
```

***

## 8. 性能调优建议

### 8.1 线程数调整

`max_threads` 参数控制并发读取线程数，默认值为 16。建议根据存储设备的 IOPS 能力进行调整：

```python
# NVMe Gen4（高性能 SSD）：max_threads=16~32
# NVMe Gen3（标准 SSD）：  max_threads=8~16
```

### 8.2 文件预对齐

`safetensors` 格式按张量粒度存储，天然适合并行读取。为获得最佳性能，建议：

- 单个 safetensors 文件大小 ≥ 128MB 时效果最佳
- 避免大量小文件（< 1MB），建议合并为大文件

### 8.3 系统调优

```shell
# 设置 CPU 调度策略（避免中断干扰）
bash scripts/set_cpu_freq.sh

# 使用 irqbalance 或手动绑定中断到特定 CPU 核心
# 确保 NVMe SSD 中断和 GPU 中断不发生冲突

# NVIDIA 特定：启用 Persistence Mode
nvidia-smi -pm 1

# AMD 特定：设置大页内存（可选）
# echo always > /sys/kernel/mm/transparent_hugepage/enabled
```

### 8.4 平台特定建议

| 平台 | 建议 |
| ---- | ---- |
| NVIDIA CUDA | 确保 CUDA 驱动 ≥ 535，使用 `nvidia-smi -pm 1` 启用持久化模式 |
| AMD ROCm | 确保 ROCm ≥ 6.0，使用 `rocm-smi` 监控 GPU 状态，必要时设置 `HSA_ENABLE_SDMA=0` |
