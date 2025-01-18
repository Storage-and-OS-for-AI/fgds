#include <sys/types.h>
#include <unistd.h>
#include <cstddef>
#include <cuda_runtime.h>
#include <fcntl.h>
#include "fgds.h"

const char *file_path = "/mnt/fgds/test.data";
static int device_id = 0;
static size_t io_size = 64 * (1 << 10); // 64KB

int main() {
    void *gpu_buffer, *target_addr;
    int ret;
    int file_fd;
    ssize_t result; 

    file_fd = open(file_path, O_CREAT | O_RDWR | O_DIRECT, 0644);


    ret = fgds_open(device_id);

    if (ret != 0) {
        printf("fgds init failed: %d\n", ret);
        return 1;
    }

    cudaMalloc(&gpu_buffer, io_size);
    cudaMemset(gpu_buffer, 0x00, io_size);
    cudaStreamSynchronize(0);

    // target_addr for register buffer less than 1GB
    ret = fgds_regmem(device_id, gpu_buffer, io_size, &target_addr);

    if (ret) {
        printf("fgds regmem failed: %d\n", ret);
        return 1;
    }

    result = pread(file_fd, target_addr, io_size, 0);

    if (result < 0) {
        perror("Read file error");
        return 1;
    }

    ret = fgds_deregmem(device_id, gpu_buffer, io_size);

    if (ret) {
        printf("fgds unregmem failed: %d\n", ret);
        return 1;
    }

    cudaFree(gpu_buffer);

    fgds_close(device_id);

    close(file_fd);
    return 0;
}