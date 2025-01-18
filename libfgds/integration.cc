#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <liburing.h>
#include <sys/types.h>
#include <unistd.h>

#include "fgds.h"


enum fgds_op {
    FGDS_OP_READ = 0,
    FGDS_OP_WRITE = 1,
};

struct fgds_data{
    int fd, op;
    struct fgds_xfer_addr *xfer_addr;
    off_t file_offset;
    ssize_t *bytes_done;
};

void CUDART_CB fgds_callback(void *user_data){
    auto* data = static_cast<fgds_data*>(user_data);
    *data->bytes_done = 0;
    ssize_t file_offset = data->file_offset;
    for (uint32_t i = 0; i < data->xfer_addr->nr_xfer_addrs; i++) {
        auto xfer_addr = data->xfer_addr->x_addrs[i];
        if (data->op == FGDS_OP_READ) {
            *data->bytes_done += pread(data->fd, xfer_addr.target_addr, 
                                xfer_addr.nbyte, file_offset);
        } else {
            *data->bytes_done += pwrite(data->fd, xfer_addr.target_addr, 
                                     xfer_addr.nbyte, file_offset);
        }
        file_offset += xfer_addr.nbyte;
    }
}
  

cudaError_t fgds_async(fgds_fileid_t fid, enum fgds_op op,
                            void* buf,
                            size_t nbytes, off_t offset,
                            ssize_t *bytes_done,
                            CUstream stream){

    struct fgds_xfer_addr* addrs = fgds_do_xfer_addr(fid.deviceID, buf, 0, nbytes);
    if (!addrs) return cudaErrorHostMemoryNotRegistered;
    auto* data = new fgds_data{
        .fd = fid.fd, .op = op,
        .xfer_addr = addrs, .file_offset = offset,
        .bytes_done = bytes_done
    };

    return cudaLaunchHostFunc(stream, fgds_callback, data);
}

cudaError_t fgds_read_async(fgds_fileid_t fid,
                            void* buf,
                            size_t nbytes, off_t offset,
                            ssize_t *bytes_done,
                            CUstream stream) {
    return fgds_async(fid, FGDS_OP_READ, buf, nbytes, offset, bytes_done, stream);
}

cudaError_t fgds_write_async(fgds_fileid_t fid,
                             void* buf, 
                             size_t nbytes, off_t offset,
                             ssize_t* bytes_done,
                             CUstream stream) {
    return fgds_async(fid, FGDS_OP_WRITE, buf, nbytes, offset, bytes_done, stream);
}