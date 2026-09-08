# FGDS:Fast GPUDirect Storage

FGDS is an optimized alternative to GDS, featuring higher performance, easier deployment, and better application compatibility. [GDS](https://docs.nvidia.com/gpudirect-storage/) (GPUDirect Storage) is a technology developed by NVIDIA that provides a direct data path between storage devices (like high-speed NVMe SSDs) and GPU memory.

Cloned from [Phoenix](https://github.com/nicexlab/phoenix) at commit [`798208d`](https://github.com/nicexlab/phoenix/tree/798208d720b234954fef433b306a485093350e2a), FGDS has undergone extensive optimizations and bug fixes, and is under active development. Welcome to use it and share your feedback—we will respond promptly, and contributions of any kind are welcome.

## Highlights

- **High Performance**
    - Eliminating the overhead of phony buffers incurred by traditional GDS, thanks to Phoenix
    - Based on io_uring, FGDS further significantly boosts disk read and write performance, with read performance improved by up to 115% and write performance by up to 40%
- **Easier Deployment**
    - Eliminating the need to install the MLNX_OFED kernel driver suite
    - Added support for users to explicitly specify a subset of GPUs to use
- **Better Application Compatibility**
    - Introduced Python API
    - Added the LMCache backend, enabling vLLM to offload KV cache via LMCache using FGDS, which accelerates inference performance
    - Added PyTorch APIs, compatible with the PyTorch GDS API, to improve the performance of reading and writing checkpoints during LLM training
    - Featuring (nearly) POSIX-compliant APIs
- **Fixes**
    - Kernel module loading failures
    - Host crashes and reboots when GPUs read from or write to NVMe drives in certain environments
    - Resource leak during kernel module loading
    - Enhanced test code and performance benchmarking programs

## Example

Below is a minimal runnable example. Compile it with:

```bash
nvcc -O2 -Wno-deprecated-gpu-targets -x cu -Imodule -lcuda -o example example.cc
```

Error checks and setup details are omitted for brevity. Check out the `example/` directory for more examples.

Usage:

```bash
./example <gpu_id> <file_path>
```

```cpp
#include <cuda.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include "fgds.h" /* UAPI: struct fgds_ioctl_reg_buffer + ioctl commands */
#if !defined(FGDS_IOCTL_MAGIC)
#error "must include module/fgds.h: check the -I order"
#endif

/* /dev/fgds_<bdf>: 0000:1e:00.0 -> /dev/fgds_0000_1e_00_0 */
static int fgds_dev_path(char *path, size_t len, const char *bdf)
{
    unsigned int dom, bus, dev, func;

    if (sscanf(bdf, "%x:%x:%x.%x", &dom, &bus, &dev, &func) != 4)
        return -1;
    snprintf(path, len, "/dev/fgds_%04x_%02x_%02x_%01x", dom, bus, dev,
             func);
    return 0;
}

#define BUF_SIZE (4UL * 1024 * 1024) /* 4MB, a multiple of the 64KB GPU page */
#define PATTERN_WORDS (BUF_SIZE / sizeof(uint64_t)) /* word i holds i */

#define FGDS_EXPORTER_ALIGN (64UL * 1024) /* NVIDIA dma-buf export alignment */
static_assert(BUF_SIZE % FGDS_EXPORTER_ALIGN == 0,
              "BUF_SIZE must be a multiple of FGDS_EXPORTER_ALIGN");

/* 1 iff every 64-bit word i of buf equals i; else prints a diagnostic. */
static int check_pattern(const char *dir, const uint64_t *buf)
{
    for (unsigned long i = 0; i < PATTERN_WORDS; i++) {
        if (buf[i] != (uint64_t)i) {
            fprintf(stderr, "%s check FAILED: word %lu is %lu, expect %lu\n",
                    dir, i, (unsigned long)buf[i], i);
            return 0;
        }
    }
    return 1;
}

int main(int argc, char *argv[])
{
    if (argc != 3) {
        fprintf(stderr, "usage: %s <gpu_id> <file_path>\n", argv[0]);
        return 1;
    }
    int gpu_id = atoi(argv[1]);
    const char *file_path = argv[2];
    char dev_path[64], pci_bus_id[32] = "";
    void *gpu_buf = NULL, *map_addr = MAP_FAILED;
    uint64_t *file_buf = NULL;
    int dev_fd = -1, file_fd = -1, dmabuf_fd = -1;
    struct fgds_ioctl_reg_buffer reg;
    struct fgds_ioctl_unreg_buffer unreg = {0};
    int registered = 0, mapped = 0;
    int ret = 0;

    /* 1. derive the fgds node of the selected device from its PCI BDF */
    cudaSetDevice(gpu_id);
    cudaDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), gpu_id);
    if (fgds_dev_path(dev_path, sizeof(dev_path), pci_bus_id) != 0) {
        fprintf(stderr, "cannot derive the fgds node for PCI %s\n",
                pci_bus_id);
        return 1;
    }

    /* 2. fill the host buffer with the pattern and upload it */
    file_buf = (uint64_t *)aligned_alloc(4096, BUF_SIZE);
    if (!file_buf) {
        perror("aligned_alloc");
        return 1;
    }
    for (unsigned long i = 0; i < PATTERN_WORDS; i++)
        file_buf[i] = (uint64_t)i;
    cudaMalloc(&gpu_buf, BUF_SIZE);
    cudaMemcpy(gpu_buf, file_buf, BUF_SIZE, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    /* 3. export the GPU buffer as a dma-buf fd */
    cuInit(0);
    cuMemGetHandleForAddressRange(&dmabuf_fd, (CUdeviceptr)gpu_buf, BUF_SIZE,
                                  CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0);

    /* 4. REG the exported range, mmap at offset reg.idx. MAP_POPULATE
     *    installs the pages up front (else lazily on first fault) */
    dev_fd = open(dev_path, O_RDWR);
    if (dev_fd < 0) {
        perror("open fgds char device");
        ret = 1;
        goto cleanup;
    }
    memset(&reg, 0, sizeof(reg));
    reg.dmabuf_fd = dmabuf_fd;
    reg.dmabuf_offset = 0;
    reg.size = BUF_SIZE;
    if (ioctl(dev_fd, FGDS_IOCTL_REG_BUFFER, &reg) < 0) {
        perror("ioctl FGDS_IOCTL_REG_BUFFER");
        ret = 1;
        goto cleanup;
    }
    registered = 1;
    map_addr = mmap(NULL, BUF_SIZE, PROT_READ | PROT_WRITE,
                    MAP_SHARED | MAP_POPULATE, dev_fd, (off_t)reg.idx);
    if (map_addr == MAP_FAILED) {
        perror("mmap fgds char device");
        ret = 1;
        goto cleanup;
    }
    mapped = 1;
    printf("FGDS_IOCTL_REG_BUFFER: gpu %p -> host %p, token=0x%llx (%lu bytes)\n",
           gpu_buf, map_addr, (unsigned long long)reg.idx, BUF_SIZE);

    /* 5. data file. map_addr is non-cached BAR memory: only I/O on it */
    file_fd = open(file_path, O_CREAT | O_RDWR | O_TRUNC | O_DIRECT, 0644);
    if (file_fd < 0) {
        perror("open data file");
        ret = 1;
        goto cleanup;
    }

    /* 6. pwrite: GPU -> file */
    if (pwrite(file_fd, map_addr, BUF_SIZE, 0) != (ssize_t)BUF_SIZE) {
        perror("pwrite GPU -> file");
        ret = 1;
        goto cleanup;
    }

    /* 7. zero the GPU buffer (so stale data can't pass), then pread */
    cudaMemset(gpu_buf, 0, BUF_SIZE);
    cudaDeviceSynchronize();
    if (pread(file_fd, map_addr, BUF_SIZE, 0) != (ssize_t)BUF_SIZE) {
        perror("pread file -> GPU");
        ret = 1;
        goto cleanup;
    }

    /* 8. D2H copy + pattern check: the disk is the only relay between pwrite
     *    and pread, so one check covers both directions */
    cudaMemcpy(file_buf, gpu_buf, BUF_SIZE, cudaMemcpyDeviceToHost);
    if (!check_pattern("round-trip", file_buf)) {
        ret = 1;
        goto cleanup;
    }
    printf("data consistency check PASSED: %lu bytes round-tripped GPU -> disk -> GPU\n", (unsigned long)BUF_SIZE);

cleanup:
    /* 9. teardown in API order UNREG -> munmap -> close, guarded by flags */
    if (registered) {
        unreg.idx = reg.idx;
        if (ioctl(dev_fd, FGDS_IOCTL_UNREG_BUFFER, &unreg) < 0)
            perror("ioctl FGDS_IOCTL_UNREG_BUFFER");
        else
            printf("FGDS_IOCTL_UNREG_BUFFER: token=0x%llx done\n",
                   (unsigned long long)reg.idx);
    }
    if (mapped)
        munmap(map_addr, BUF_SIZE);
    close(dev_fd);
    close(dmabuf_fd);
    close(file_fd);
    cudaFree(gpu_buf);
    free(file_buf);
    return ret;
}
```

## Performance Results

Test environment:

- Physical machine
- Linux 6.6 kernel
- NVIDIA H100 GPU
- CUDA 12.8
- NVIDIA 570.124.06 GPU driver
- Hygon C86 x86_64 CPU
- Samsung NVMe SSD 990 EVO Plus 4TB (fio peak read bandwidth 6.6GB/s, peak write bandwidth 5.8GB/s)
- 1TB memory

The following shows bandwidth and latency of FGDS, GDS, and POSIX across different block sizes. FGDS outperforms GDS and POSIX in both read and write performance across block sizes.

Before the disk bandwidth is saturated, the read/write performance comparison is as follows:

1. For reads: FGDS outperforms GDS by 11%~109%, and POSIX by 40%~143%
2. For writes: FGDS outperforms GDS by 10%~71%, and POSIX by 71%~130%

### Read Bandwidth

![Read Bandwidth](./picture/read_bandwidth.png)

### Write Bandwidth

![Write Bandwidth](./picture/write_bandwidth.png)

### Read Latency

![Read Latency](./picture/read_latency.png)

### Write Latency

![Write Latency](./picture/write_latency.png)

## Documentation

| Doc | Link |
| --- | --- |
| Getting Started | [docs/getting-started.md](./docs/getting-started.md) |
| libfgds | [docs/libfgds.md](./docs/libfgds.md) |
| FGDS Python API | [python/README.md](./python/README.md) |
| FGDS vLLM LMCache backend | [python/lmcache.md](./python/lmcache.md) |
| FGDS PyTorch API | [pytorch-fgds/README.md](./pytorch-fgds/README.md) |
| FastSafeTensors model loading | [docs/fgds-fastsafetensor.md](./docs/fgds-fastsafetensor.md) |
| Micro benchmark | [docs/micro-benchmark.md](./docs/micro-benchmark.md) |

## News

- **2026-06-03** — Significantly improved read/write performance with io_uring
- **2026-04-21** — Fixed a resource leak bug and performed some code optimizations
- **2026-03-11** — Added PyTorch FGDS API as a drop-in replacement for PyTorch GDS API in scenarios such as checkpoint saving
- **2026-02-07** — Fixed kernel module load failures
- **2026-01-28** — Added support for multi-GPU, multi-user environments
- **2026-01-13** — Improved test programs and scripts
- **2025-12-18** — Added the FGDS backend for LMCache, enabling vLLM to utilize FGDS for KV cache offloading
- **2025-11-25** — Added FGDS Python API
- **2025-11-06** — Cloned from [Phoenix](https://github.com/nicexlab/phoenix) at commit [`798208d`](https://github.com/nicexlab/phoenix/tree/798208d720b234954fef433b306a485093350e2a)

## Copyright & LICENSE

`SPDX-License-Identifier: Apache-2.0`

See the full license text in the root [LICENSE](./LICENSE).

FGDS is licensed under the Apache License, Version 2.0 (the "License").
You may not use this file except in compliance with the License.
You may obtain a copy of the License at:

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
