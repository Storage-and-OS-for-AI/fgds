# FGDS Documents

This directory contains FGDS documentation. Below is an overview of the project's source code structure.

## Documents

| File | Description |
|------|-------------|
| [getting-started.md](getting-started.md) | Build and run guide |
| [fgds-fs.md](fgds-fs.md) | Kernel module (`fgdsfs.ko`) internals — char device, mmap, ioctl |
| [libfgds.md](libfgds.md) | Userspace library API reference |
| [fgds-fastsafetensor.md](fgds-fastsafetensor.md) | FastSafeTensors integration for high-performance model loading |
| [micro-benchmark.md](micro-benchmark.md) | Micro benchmark program |
| [troubleshooting.md](troubleshooting.md) | Troubleshooting guide for common issues |

## Source Tree

```
.
├── libfgds/                # Userspace C/C++ library (libfgds.so)
├── module/                 # Linux kernel module (fgdsfs.ko)
├── python/                 # Python bindings + LMCache backend
├── pytorch-fgds/           # PyTorch GDS-compatible API
├── example/                # C++ usage example
├── benchmarks/             # Performance benchmarks
├── scripts/                # Helper and benchmark-launch scripts
├── docs/                   # Documentation (this directory)
└── picture/                # Performance charts
```

## Component Overview

### `libfgds/` — Userspace library

The core FGDS C/C++ API. Applications use `fgds_regmem()` to map GPU buffers into host-visible virtual addresses, then perform direct `pread()`/`pwrite()` for GPU↔SSD DMA. For large transfers, an io_uring pipeline is used to maximize throughput.

```
libfgds/
├── include/
│   ├── fgds.h              # Public API (fgds_open, fgds_close, fgds_regmem, read/write, etc.)
│   └── common.h            # Shared kernel-userspace structs & ioctl definitions
├── fgds.cc                 # Core logic: GPU memory registration, io_uring pipeline, mmap management
└── integration.cc          # Async I/O via CUDA stream callbacks
```

### `module/` — Kernel module

The `fgdsfs.ko` kernel module discovers NVIDIA GPUs via PCIe, remaps GPU BAR memory into kernel space (ZONE_DEVICE), and exposes `/dev/fgds_devX` character devices. These provide `mmap()` and `ioctl()` interfaces so userspace can establish direct GPU-memory-to-host mappings.

```
module/
├── fgds.c / fgds.h         # Module init/exit, char device ops, ioctl handlers
├── fgds-mem.c / fgds-mem.h # mmap handler, page table setup (vm_insert_pages)
├── nvfs-pci.c / nvfs-pci.h # NVIDIA GPU PCIe BAR discovery
├── nvfs-p2p.c / nvfs-p2p.h # NVIDIA P2P symbol integration
├── nvfs-core.h / nvfs-mmap.h  # Shared definitions
├── configure / Makefile.in # Kbuild-based build
└── README.md
```

### `python/` — Python bindings

Provides a Python package wrapping `libfgds.so` via ctypes, plus an LMCache storage backend implementation for vLLM KV cache offloading to NVMe.

```
python/
├── fgds/
│   ├── fgds_bind.py        # ctypes wrapper around libfgds.so
│   └── fgds.py             # High-level FGDS file class
├── fgds_backend/
│   └── fgds_backend.py     # LMCache StorageBackendInterface implementation
├── test/
├── setup.py
└── README.md
```

### `pytorch-fgds/` — PyTorch integration

Drop-in PyTorch API compatible with `torch.cuda.gds`, providing GPU↔NVMe checkpoint read/write for LLM training.

```
pytorch-fgds/
├── fgds_torch.py           # fgds_register_buffer, fgds_deregister_buffer, FgdsFile
├── benchmark/               # PyTorch I/O benchmarks vs GDS and POSIX
└── README.md
```

### `scripts/` — Helper scripts

Utility and benchmark-launch scripts. Includes the kernel module loader (`load_fgds.sh`) / unloader (`unload_fgds.sh`), micro-benchmark runner.

```
scripts/
├── load_fgds.sh            # Load fgdsfs.ko with GPU config from config.json
├── unload_fgds.sh          # Unload the fgdsfs.ko module
├── micro.py                # Micro-benchmark runner (FGDS / GDS / POSIX comparison)
├── breakdown.sh / kvcache.sh / nfs.sh / nvme_of.sh  # Benchmark launch scripts
├── load_safetensors.py     # Safetensors loading helper
├── logger.py               # Logging utilities
├── set_cpu_freq.sh         # CPU frequency governor setup
```

### `benchmarks/` — Benchmark suites

Six benchmark suites for comparing FGDS against GDS and POSIX across different workloads. Built with CMake + CUDA.

```
benchmarks/
├── micro-benchmarks/       # Unit-level I/O comparisons
├── end-to-end/             # Full pipeline benchmark
├── kvcache/                # KV cache workload simulation
├── safetensor/             # Safetensor format I/O
├── breakdown/              # Latency breakdown analysis
├── fio/                    # FIO external I/O engine plugin
└── utils/                  # Shared helper headers
```

## Data Flow Summary

1. **Kernel module** remaps GPU BAR memory → exposes `/dev/fgds_devX`
2. **libfgds** registers GPU buffers via ioctl, then uses io_uring to perform direct read/write on remapped addresses — bypassing CPU bounce buffers
3. **Language bindings** (Python, PyTorch) wrap libfgds for application-level use
