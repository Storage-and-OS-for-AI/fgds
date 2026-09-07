# Getting Started

FGDS is an optimized alternative to NVIDIA [GDS](https://docs.nvidia.com/gpudirect-storage/) that provides a direct data path between NVMe SSDs and GPU memory. This document describes how to build FGDS from source and load the kernel module. See the [main README](../README.md) for an overview, API example, and performance results.

## 1. Prerequisites

- CMake >= 3.18
- GCC / G++ >= 11
- Linux kernel headers
- liburing-devel (liburing-dev on Debian/Ubuntu)
- pkg-config
- CUDA Toolkit >= 12.4 (tested with 12.8)
- NVIDIA driver with kernel module sources (`nv-reg.h` / `nv-p2p.h` under `/usr/src/nvidia-*`)

**Ubuntu / Debian:**

```shell
sudo apt install cmake build-essential linux-headers-$(uname -r) liburing-dev pkg-config
```

**RHEL / CentOS / Fedora:**

```shell
sudo dnf install cmake make gcc gcc-c++ kernel-devel liburing-devel pkgconfig
```

Install CUDA Toolkit from the [NVIDIA CUDA download page](https://developer.nvidia.com/cuda-downloads). Verify:

```shell
nvcc --version
```

## 2. Build

### 2.1 Configure

```shell
mkdir -p build && cd build
cmake ..
```

### 2.2 Compile

```shell
make -j$(nproc)
```

A successful build produces (relative to the current `build/` directory):

- `module/fgdsfs.ko` — the kernel module (loaded in step 3.2)
- `libfgds.so` — the userspace library
- `bin/example` — the C++ example (run in step 3.3)
- `bin/microbenchmark` — the micro benchmark binary

## 3. Run

Return to the project root first.

### 3.1 Edit `config.json`

Edit `config.json` in the project root to specify which GPUs to bind.

- `use_all_gpus: true` binds all GPUs; `false` binds only the GPUs listed in the `gpuids` array.
- `gpuids` is an integer array of GPU IDs, such as `[0, 1]`.

### 3.2 Load the Kernel Module

```shell
bash scripts/load_fgds.sh
```

This script binds the GPUs specified in `config.json`. The NVIDIA driver must be loaded first — run `nvidia-smi` to verify.

If some GPUs cannot be bound, FGDS skips them and keeps the rest; the module fails to load only when all specified GPUs fail to bind. See [Troubleshooting](troubleshooting.md) for details.

### 3.3 Verify

After loading, confirm the module is loaded and the GPU devices are exposed:

```shell
lsmod | grep fgdsfs
ls /dev/fgds_dev*
```

Run the C++ example to do a GPU↔SSD read/write round-trip:

```shell
./build/bin/example 0 /path/to/test.data
```

Use a path on a filesystem that supports `O_DIRECT` (a local NVMe/SSD, not `/tmp`). For performance testing, see [Micro Benchmark](micro-benchmark.md).

### 3.4 Unload the Kernel Module

```shell
bash scripts/unload_fgds.sh
```
