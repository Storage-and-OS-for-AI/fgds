// SPDX-License-Identifier: Apache-2.0
/*
 * minimal_example - user-kernel API flow of the fgds kernel
 * module (module/) UAPI. It walks every ioctl/mmap step in order and
 * exercises the pwrite/pread data flow:
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
 * Build: nvcc -O2 -Wno-deprecated-gpu-targets -x cu -Imodule -lcuda -o minimal_example example/minimal_example.cc 
 * Run:   ./minimal_example <gpu_id> <file_path>
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

#define FGDS_EXPORTER_ALIGN (64UL * 1024) /* NVIDIA dma-buf export alignment */
static_assert(BUF_SIZE % FGDS_EXPORTER_ALIGN == 0,
              "BUF_SIZE must be a multiple of FGDS_EXPORTER_ALIGN");

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

    /* 2. allocate a GPU buffer the size of the range to register */
    cudaMalloc(&gpu_buf, BUF_SIZE);

    /* 3. export the GPU buffer as a dma-buf fd */
    cuInit(0);
    cuMemGetHandleForAddressRange(&dmabuf_fd, (CUdeviceptr)gpu_buf, BUF_SIZE,
                                  CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0);

    /* 4. REG the exported range, mmap at offset reg.idx */
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

cleanup:
    /* 8. teardown in API order UNREG -> munmap -> close, guarded by flags */
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
    return ret;
}