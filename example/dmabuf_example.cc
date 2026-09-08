// SPDX-License-Identifier: Apache-2.0 
/*
 * dmabuf_example - end-to-end example of the fgds kernel module
 * (module/) UAPI, emphasizing the user-kernel API flow:
 *
 *   user space                              kernel module
 *   ------------------------------------    --------------------------
 *   cudaMalloc + H2D cudaMemcpy(pattern)
 *   cuMemGetHandleForAddressRange      -->  export dma-buf fd
 *   open("/dev/fgds_<bdf>")
 *   ioctl(FGDS_IOCTL_REG_BUFFER, &reg) -->  bind dma-buf, returns reg.idx
 *   mmap(dev_fd, size, offset=reg.idx) -->  map the registered range
 *   pwrite(map_addr)                   -->  DMA GPU -> file
 *   pread(map_addr)                    -->  DMA file -> GPU
 *   ioctl(FGDS_IOCTL_UNREG_BUFFER)     -->  drop the registry entry
 *   munmap / close / cudaFree
 *
 * No kernels: one pattern, one round trip checking both directions.
 *
 * Build: nvcc -O2 -Wno-deprecated-gpu-targets -x cu -Imodule -o dmabuf_example example/dmabuf_example.cc -lcuda
 * Run:   ./dmabuf_example <gpu_id> <file_path>
 * Note:  file_path needs an O_DIRECT-capable filesystem (local NVMe/SSD);
 *        the fgds node is derived from the GPU PCI BDF
 *        (cudaDeviceGetPCIBusId), not from <gpu_id>.
 */
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