#ifndef __FGDS_H__
#define __FGDS_H__
#include <cstddef>
#include <cstdint>
#include <stdint.h>
#include <sys/types.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdio.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct fgds_fileid {
    int fd; // 磁盘文件的fd
    int deviceID;
} fgds_fileid_t;

struct xfer_addr {
    void *target_addr;
    size_t nbyte;
};

#define MAX_NR_ADDR 4
struct fgds_xfer_addr{
    uint32_t nr_xfer_addrs;
    struct xfer_addr x_addrs[1];
};

// para for a single io operation

int fgds_open(int device_id);
int fgds_close(int device_id);
ssize_t fgds_read(fgds_fileid_t fid, void *buf, off_t buf_offset, ssize_t nbyte, off_t f_offset);
ssize_t fgds_write(fgds_fileid_t fid, void *buf, off_t buf_offset, ssize_t nbyte, off_t f_offset);

struct fgds_xfer_addr * fgds_do_xfer_addr(int device_id, const void *buf, off_t buf_offset, size_t nbyte);
int fgds_regmem(int device_id, const void *addr, size_t len, void **target_addr);
int fgds_deregmem(int device, const void *addr, size_t len);

cudaError_t fgds_read_async(fgds_fileid_t fid,
                            void* buf,
                            size_t nbytes, off_t offset,
                            ssize_t *bytes_done,
                            CUstream stream);

cudaError_t fgds_write_async(fgds_fileid_t fid,
                            void* buf,
                            size_t nbytes, off_t offset,
                            ssize_t *bytes_done,
                            CUstream stream);

#ifdef __cplusplus
}
#endif
#endif
