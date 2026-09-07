# FGDS：Fast GPUDirect Storage

[English](../README.md) | 中文

FGDS 是对 GDS 的优化替代方案，具备更高性能、更易部署、更好的应用兼容性。[GDS](https://docs.nvidia.com/gpudirect-storage/)（GPUDirect Storage）是 NVIDIA 推出的技术，可在存储设备（如高速 NVMe SSD）与 GPU 显存之间提供直接数据通路。

基于 [Phoenix](https://github.com/nicexlab/phoenix) 的 commit [`798208d`](https://github.com/nicexlab/phoenix/tree/798208d720b234954fef433b306a485093350e2a) 进行修改，FGDS 经过了大量优化与问题修复，目前仍在积极开发中。欢迎使用并反馈——我们会及时响应，也欢迎任何形式的贡献。

## 亮点

- **高性能**
    - 借助 Phoenix，消除传统 GDS 因 phony buffer 带来的开销
    - 基于 io_uring，FGDS 进一步大幅提升磁盘读写性能，读性能最高提升 115%，写性能最高提升 40%
- **更易部署**
    - 无需再安装 MLNX_OFED 内核驱动套件
    - 新增支持用户显式指定使用部分 GPU
- **更好的应用兼容性**
    - 新增 Python API
    - 新增 LMCache 后端，使 vLLM 可通过 LMCache 使用 FGDS 卸载 KV cache，从而加速推理性能
    - 新增 PyTorch API，兼容 PyTorch GDS API，可提升大模型训练时读写 checkpoint 等场景的性能
    - 提供 (近乎) POSIX 兼容的 API
- **问题修复**
    - 内核模块加载失败
    - 部分环境下 GPU 读写 NVMe 盘时主机宕机重启
    - 内核模块加载时资源泄露
    - 完善测试代码和性能测试程序

## 示例

下面的代码片段展示了 FGDS 的主要 API。

除 GPU 缓冲区注册（`fgds_regmem` / `fgds_deregmem`）外，文件 I/O 通过对已注册映射（`target_addr`）使用标准 POSIX `pread` / `pwrite` 完成。

为简洁起见省略了错误检查与环境准备细节；完整可运行程序见 [example/example.cc](./example/example.cc)。

```cpp
int device_id = 0;
size_t io_size = 4 * 1024 * 1024;  // 4MB
void *gpu_buffer = nullptr, *target_addr = nullptr;

int fd = open("/data/test.bin", O_CREAT | O_RDWR | O_DIRECT, 0644);

fgds_open(device_id);
cudaMalloc(&gpu_buffer, io_size);
fgds_regmem(device_id, gpu_buffer, io_size, &target_addr);  // extra step vs. POSIX

pwrite(fd, target_addr, io_size, 0);
pread(fd, target_addr, io_size, 0);

fgds_deregmem(device_id, gpu_buffer, io_size);
cudaFree(gpu_buffer);
fgds_close(device_id);
close(fd);
```

## 性能结果

所用测试环境：

- 物理机
- Linux 6.6 内核
- NVIDIA H100 GPU
- CUDA 12.8
- NVIDIA 570.124.06 显卡驱动
- 海光 C86 x86_64 CPU
- Samsung NVMe SSD 990 EVO Plus 4TB（fio 测试最高读带宽 6.6GB/s，最高写带宽 5.8GB/s）
- 1TB 内存

以下为读写不同 block size 时的 FGDS、GDS、POSIX 3 种方式的带宽、时延的数据，可见 FGDS 在各 block size 区间的读写性能都领先 GDS 和 POSIX。

在磁盘带宽未打满之前，读写性能比较如下：

1. 读性能方面：FGDS 领先 GDS 11%~109%，领先 POSIX 40%~143%
2. 写性能方面：FGDS 领先 GDS 10%~71%，领先 POSIX 71%~130%

### 读带宽

![读带宽](./picture/read_bandwidth.png)

### 写带宽

![写带宽](./picture/write_bandwidth.png)

### 读时延

![读时延](./picture/read_latency.png)

### 写时延

![写时延](./picture/write_latency.png)

## 文档

| 文档 | 链接 |
| --- | --- |
| 构建 / 安装 | [docs/install.md](./docs/install.md) |
| 内核模块与字符设备接口 | [docs/fgds-fs.md](./docs/fgds-fs.md) |
| libfgds | [docs/libfgds.md](./docs/libfgds.md) |
| FGDS Python API | [python/README.md](./python/README.md) |
| FGDS vLLM LMCache 后端 | [python/lmcache.md](./python/lmcache.md) |
| PyTorch FGDS API | [pytorch-fgds/README.md](./pytorch-fgds/README.md) |
| FastSafeTensors 模型加载 | [docs/fgds-fastsafetensor.md](./docs/fgds-fastsafetensor.md) |
| 微benchmark | [docs/micro-benchmark.md](./docs/micro-benchmark.md) |

## 新闻

- **2026-07-28** — 正式开源到 openEuler 社区
- **2026-06-03** — 基于 io_uring 大幅提升读写性能
- **2026-04-21** — 修复资源泄露问题并进行了一些代码优化
- **2026-03-11** — 新增 PyTorch FGDS API，作为 PyTorch GDS API 的直接替代，可用于 checkpoint 保存等场景
- **2026-02-07** — 修复内核模块加载失败问题
- **2026-01-28** — 新增对多卡多用户环境的支持
- **2026-01-13** — 完善测试程序和相关脚本
- **2025-12-18** — 新增 FGDS 的 LMCache 后端，使 vLLM 可利用 FGDS 进行 KV cache 卸载
- **2025-11-25** — 新增 FGDS Python API
- **2025-11-06** — 基于 [Phoenix](https://github.com/nicexlab/phoenix) 的 commit [`798208d`](https://github.com/nicexlab/phoenix/tree/798208d720b234954fef433b306a485093350e2a) 进行修改

## Copyright & LICENSE

`SPDX-License-Identifier: Apache-2.0`

完整许可证文本见根目录 [LICENSE](./LICENSE)。

FGDS 基于 Apache License, Version 2.0（"License"）授权。
您不得在不符合 License 的情况下使用本软件。
您可以在以下位置获取 License 副本：

    http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，本软件基于"按原样"提供，不附带任何明示或默示的担保。
详见 License 中关于特定语言的管辖权限和限制的条款。
