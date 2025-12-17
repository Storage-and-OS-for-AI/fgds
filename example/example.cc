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

    printf("fgds init start\n");
    ret = fgds_open(device_id);
    printf("fgds init ret: %d\n", ret);

    if (ret != 0) {
        printf("fgds init failed: %d\n", ret);
        return 1;
    }
    cudaSetDevice(device_id);
    cudaMalloc(&gpu_buffer, io_size);
    cudaMemset(gpu_buffer, 0x00, io_size);
    cudaStreamSynchronize(0);
    printf("fgds regmem start\n");
    // target_addr for register buffer less than 1GB
    ret = fgds_regmem(device_id, gpu_buffer, io_size, &target_addr);
    printf("fgds regmem ret: %d\n", ret);
    if (ret) {
        printf("fgds regmem failed: %d\n", ret);
        return 1;
    }
    printf("fgds regmem target_addr: %p\n", target_addr);
    result = pread(file_fd, target_addr, io_size, 0);
    printf("fgds pread ret: %ld\n", result);
    if (result < 0) {
        perror("Read file error");
        return 1;
    }
    printf("fgds pread done\n");
    ret = fgds_deregmem(device_id, gpu_buffer, io_size);
    printf("fgds unregmem ret: %d\n", ret);
    if (ret) {
        printf("fgds unregmem failed: %d\n", ret);
        return 1;
    }

    cudaFree(gpu_buffer);

    fgds_close(device_id);

    close(file_fd);
    return 0;
}