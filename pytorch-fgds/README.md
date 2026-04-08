# pytorch-fgds

这个目录包含了 `fgds.py` 以及用于性能对比的 benchmark 脚本。

## 在 PyTorch 中接入 FGDS API

要在 PyTorch 项目中启用 `torch.cuda.fgds`，需要做下面两步：

1. 将本目录下的 `fgds.py` 添加到 PyTorch 源码目录：
   - `torch/cuda/fgds.py`
2. 修改 PyTorch 源码中的 `torch/cuda/__init__.py`，增加导入语句：
   - `from . import fgds`

完成后，Python 侧即可通过 `from torch.cuda import fgds` 使用 FGDS API。

## `fgds.py` 设计思路

`fgds.py` 的 Python 接口设计整体参照了 `torch/cuda/gds.py`：

- `gds.py` 对外导出 `gds_register_buffer`、`gds_deregister_buffer`、`GdsFile`
- `fgds.py` 对外导出对应的 `fgds_register_buffer`、`fgds_deregister_buffer`、`FgdsFile`
- `GdsFile` 对外提供 `load_storage` / `save_storage`
- `FgdsFile` 也对外提供 `load_storage` / `save_storage`

这样做的目的是让 `torch.cuda.fgds` 的使用习惯与 `torch.cuda.gds` 保持一致，便于迁移和对比测试。

`fgds.py` 中主要函数/方法作用如下：

- `_maybe_prepend_repo_python_to_path`
  - 在源码树场景下，自动把 `python/` 加入 `sys.path`，便于加载 `python/fgds`。

- `_load_fgds_bind`
  - 统一加载 `fgds.fgds_bind`，做一次性缓存，并在加载失败时给出明确错误信息。

- `_fgds_file_constructed` / `_fgds_file_destroyed`
  - 以 `device_id` 维度维护 `FgdsFile` 实例计数。
  - 首个实例创建时调用 `fgds_open(device_id)`，最后一个实例销毁时调用 `fgds_close(device_id)`。

- `_fgds_file_session_active`
  - 查询指定设备是否已有活跃 FGDS 会话（用于注册 buffer 前置校验）。

- `_cuda_device_index`
  - 从 `Storage` 解析 CUDA device index，并校验 storage 必须在 CUDA 上。

- `fgds_register_buffer`
  - 调用 `fgds_regmem` 注册 GPU storage（要求同设备上已先创建 `FgdsFile`）。

- `fgds_deregister_buffer`
  - 调用 `fgds_deregmem` 反注册 GPU storage。

- `FgdsFile.__init__`
  - 打开文件（`os.O_DIRECT`）、持有 FGDS 会话、创建 `fgds_fileid_t`。

- `FgdsFile.__del__`
  - 释放文件 fd，并在需要时减少会话计数，触发 `fgds_close`。

- `FgdsFile.load_storage`
  - 调用 `fgds_read`，将文件内容读入 CUDA storage。

- `FgdsFile.save_storage`
  - 调用 `fgds_write`，将 CUDA storage 内容写入文件。

## benchmark 目录说明

`benchmark/` 目录用于比较不同读写路径的性能（默认测试 10GB 文件）：

- `bench_fgds_pytorch.py`
  - 测试 `torch.cuda.fgds` 的文件写入/读取性能。
  - 每轮执行一次 write + read，打印每轮和平均时延/带宽。

- `bench_gds_pytorch.py`
  - 测试 `torch.cuda.gds`（cuFile 路径）的写入/读取性能。
  - 每轮执行一次 write + read，打印每轮和平均时延/带宽。

- `bench_posix_pytorch.py`
  - 测试两种普通文件路径：
    - `torch.save` / `torch.load`（page cache 路径）
    - `direct-io`（`O_DIRECT` 原始读写路径）
  - 输出每轮和平均时延/带宽，并做读回一致性校验。

- `common.py`
  - 公共配置和工具函数（如测试文件路径、测试大小、参数解析、分块一致性校验）。

- `run_all_io_tests.sh`
  - 串行执行上述 benchmark 脚本，并在每个脚本之间 `sleep 3` 秒。

## 使用方式

先进入 benchmark 目录：

```bash
cd benchmark
```

运行单个 benchmark（参数：`GPU_ID` `NUM_ROUNDS`）：

```bash
python bench_fgds_pytorch.py 0 3
python bench_gds_pytorch.py 0 3
python bench_posix_pytorch.py 0 3
```

运行全部 benchmark：

```bash
./run_all_io_tests.sh 0 3
```

