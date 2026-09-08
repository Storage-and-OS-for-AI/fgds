// SPDX-License-Identifier: Apache-2.0
/*
 * Example using fgds on AMD/ROCm
 *
 *   user space                                   kernel module
 *   -----------------------------------------    -----------------------
 *   hipMalloc
 *   hsa_amd_portable_export_dmabuf          -->  export dma-buf fd
 *   open("/dev/fgds_<bdf>")
 *   ioctl(FGDS_IOCTL_REG_BUFFER, &reg)      -->  bind dma-buf, returns reg.idx
 *   mmap(dev_fd, size, offset=reg.idx)      -->  map the registered range
 *   pwrite(map_addr)                        -->  DMA GPU -> file
 *   pread(map_addr)                         -->  DMA file -> GPU
 *   ioctl(FGDS_IOCTL_UNREG_BUFFER)          -->  drop the registry entry
 *   munmap / close / hipFree
 *
 * Build: hipcc -O2 -I../module -o hip_example hip_example.cc -lhsa-runtime64
 * Run:   ./hip_example <gpu_id> <file_path>
 * Note:  file_path needs an O_DIRECT-capable filesystem (local NVMe/SSD);
 *
 */
#include <hip/hip_runtime.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include "fgds.h"

/* /dev/fgds_<bdf>: 0000:04:00.0 -> /dev/fgds_0000_04_00_0 */
static int fgds_dev_path(char *path, size_t len, const char *bdf)
{
    unsigned int dom, bus, dev, func;

    if (sscanf(bdf, "%x:%x:%x.%x", &dom, &bus, &dev, &func) != 4)
        return -1;
    snprintf(path, len, "/dev/fgds_%04x_%02x_%02x_%01x", dom, bus, dev,
             func);
    return 0;
}

#define BUF_SIZE (4UL * 1024 * 1024)
#define PATTERN_WORDS (BUF_SIZE / sizeof(uint64_t))

/* check buf content */
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
    uint64_t dmabuf_offset = 0;
    struct fgds_ioctl_reg_buffer reg;
    struct fgds_ioctl_unreg_buffer unreg = {0};
    int registered = 0, mapped = 0;
    int ret = 0;

    /* Get GPU PCI BDF info */
    hipSetDevice(gpu_id);
    hipDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), gpu_id);

    /* Format to fgds dev_path */
    if (fgds_dev_path(dev_path, sizeof(dev_path), pci_bus_id) != 0) {
        fprintf(stderr, "cannot derive the fgds node for PCI %s\n",
                pci_bus_id);
        return 1;
    }

    /* Fill the host buffer with the pattern and upload it */
    file_buf = (uint64_t *)aligned_alloc(4096, BUF_SIZE);
    if (!file_buf) {
        perror("aligned_alloc");
        return 1;
    }
    for (unsigned long i = 0; i < PATTERN_WORDS; i++)
        file_buf[i] = (uint64_t)i;
    
    hipMalloc(&gpu_buf, BUF_SIZE);
    (void)hipMemcpy(gpu_buf, file_buf, BUF_SIZE, hipMemcpyHostToDevice);
    (void)hipDeviceSynchronize();

    /* Export the GPU buffer as a dma-buf fd via the HSA API */
    hsa_init();
    hsa_amd_portable_export_dmabuf((const void *)gpu_buf, BUF_SIZE,
				   &dmabuf_fd, &dmabuf_offset);

    /* Open fgds device to obtain dev_fd */
    dev_fd = open(dev_path, O_RDWR);
    if (dev_fd < 0) {
        perror("open fgds char device");
        ret = 1;
        goto cleanup;
    }

    memset(&reg, 0, sizeof(reg));
    reg.dmabuf_fd = dmabuf_fd;
    reg.dmabuf_offset = dmabuf_offset;
    reg.size = BUF_SIZE;
    /* Register dmabuf_fd in fgds. MAP_POPULATE
     * installs the pages up front (else lazily on first fault) */
    if (ioctl(dev_fd, FGDS_IOCTL_REG_BUFFER, &reg) < 0) {
        perror("ioctl FGDS_IOCTL_REG_BUFFER");
        ret = 1;
        goto cleanup;
    }
    registered = 1;

    /* Setup the mapping, MAP_POPULATE is necessary */
    map_addr = mmap(NULL, BUF_SIZE, PROT_READ | PROT_WRITE,
                    MAP_SHARED | MAP_POPULATE, dev_fd, (off_t)reg.idx);
    if (map_addr == MAP_FAILED) {
        perror("mmap fgds device");
        ret = 1;
        goto cleanup;
    }
    mapped = 1;

    /* Open a data file */
    file_fd = open(file_path, O_CREAT | O_RDWR | O_TRUNC | O_DIRECT, 0644);
    if (file_fd < 0) {
        perror("open data file");
        ret = 1;
        goto cleanup;
    }

    /* Call pwrite: GPU -> file */
    if (pwrite(file_fd, map_addr, BUF_SIZE, 0) != (ssize_t)BUF_SIZE) {
        perror("pwrite GPU -> file");
        ret = 1;
        goto cleanup;
    }

    /* Zero the GPU buffer, then call pread */
    (void)hipMemset(gpu_buf, 0, BUF_SIZE);
    (void)hipDeviceSynchronize();
    if (pread(file_fd, map_addr, BUF_SIZE, 0) != (ssize_t)BUF_SIZE) {
        perror("pread file -> GPU");
        ret = 1;
        goto cleanup;
    }

    /* Do D2H copy and pattern check */
    (void)hipMemcpy(file_buf, gpu_buf, BUF_SIZE,
                    hipMemcpyDeviceToHost);
    if (!check_pattern("round-trip", file_buf)) {
        ret = 1;
        goto cleanup;
    }
    printf("data consistency check PASSED: %lu bytes round-tripped GPU -> disk -> GPU\n", (unsigned long)BUF_SIZE);

cleanup:
    /* Teardown in API order UNREG -> munmap -> close, guarded by flags */
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
    if (dev_fd >= 0)
        close(dev_fd);
    if (dmabuf_fd >= 0)
        close(dmabuf_fd);
    if (file_fd >= 0)
        close(file_fd);
    if (gpu_buf)
        (void)hipFree(gpu_buf);
    free(file_buf);
    return ret;
}
