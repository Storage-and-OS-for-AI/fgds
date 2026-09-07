/*
 * fgds_smoke: fgds GPU<->NVMe 读写数据一致性校验（冒烟测试）。
 */
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstddef>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <iostream>
#include <cerrno>
#include <cstring>
#include <cstdlib>
#include "fgds.h"
#include <ctime>

static int device_id = 0;
static const char *file_path = "/data3/10GB_ddrand";

// Linux 单次 read/write 系统调用最多只能传输 2GB（0x7FFFF000 字节），
// 读写更大文件时必须分块。以下两个函数循环调用 pread/pwrite 直到完成全部数据。
#define FGDS_MAX_IO_BYTES (1UL << 30)  // 单次最多 1GB，避免触及 2GB 上限

static ssize_t pread_full(int fd, void *buf, size_t count, off_t offset) {
    size_t done = 0;
    while (done < count) {
        size_t chunk = count - done;
        if (chunk > FGDS_MAX_IO_BYTES) {
            chunk = FGDS_MAX_IO_BYTES;
        }
        ssize_t n = pread(fd, (char *)buf + done, chunk, offset + done);
        if (n < 0) {
            return n;
        }
        if (n == 0) {  // 读到 EOF
            break;
        }
        done += (size_t)n;
    }
    return (ssize_t)done;
}

// 文件拷贝校验：GPU<->磁盘 IO 走 fgds_read / fgds_write（支持超过 1GB 的文件）。
// 主机侧校验仍使用 pread。
int fgds_check_1() {
    void *gpu_buffer = NULL, *target_addr = NULL;
    int ret;
    int src_file_fd = -1, dst_file_fd = -1;
    ssize_t result;
    struct stat file_stat;
    size_t file_size;
    void *src_buffer = NULL, *dst_buffer = NULL;
    const char *dst_file_path = NULL;
    char dst_path[512];
    int cmp_result = 0;
    fgds_fileid_t src_fid = {};
    fgds_fileid_t dst_fid = {};

    if (stat(file_path, &file_stat) < 0) {
        perror("stat source file error");
        return 1;
    }
    file_size = file_stat.st_size;
    if (file_size == 0) {
        printf("source file is empty: %s\n", file_path);
        return 1;
    }
    printf("fgds_check_1 start, file_path: %s, file_size: %zu bytes (%.2f MB), device_id: %d\n",
           file_path, file_size, file_size / (1024.0 * 1024.0), device_id);

    src_file_fd = open(file_path, O_RDONLY | O_DIRECT, 0644);
    if (src_file_fd < 0) {
        perror("open source file error");
        return 1;
    }

    snprintf(dst_path, sizeof(dst_path), "%s_fgds_copy_rw", file_path);
    dst_file_path = dst_path;

    dst_file_fd = open(dst_file_path, O_CREAT | O_WRONLY | O_TRUNC | O_DIRECT, 0644);
    if (dst_file_fd < 0) {
        perror("open destination file error");
        close(src_file_fd);
        return 1;
    }

    if (ftruncate(dst_file_fd, (off_t)file_size) != 0) {
        perror("ftruncate destination file error");
        close(src_file_fd);
        close(dst_file_fd);
        unlink(dst_file_path);
        return 1;
    }

    ret = fgds_open(device_id);
    if (ret != 0) {
        printf("fgds_open failed: %d\n", ret);
        close(src_file_fd);
        close(dst_file_fd);
        unlink(dst_file_path);
        return 1;
    }

    cudaError_t cuda_ret = cudaMalloc(&gpu_buffer, file_size);
    if (cuda_ret != cudaSuccess) {
        printf("cudaMalloc failed: %s\n", cudaGetErrorString(cuda_ret));
        fgds_close(device_id);
        close(src_file_fd);
        close(dst_file_fd);
        unlink(dst_file_path);
        return 1;
    }
    cudaMemset(gpu_buffer, 0x00, file_size);
    cudaStreamSynchronize(0);

    ret = fgds_regmem(device_id, gpu_buffer, file_size, &target_addr);
    if (ret) {
        printf("fgds regmem failed: %d\n", ret);
        cudaFree(gpu_buffer);
        fgds_close(device_id);
        close(src_file_fd);
        close(dst_file_fd);
        unlink(dst_file_path);
        return 1;
    }

    src_fid.fd = src_file_fd;
    src_fid.deviceID = device_id;
    dst_fid.fd = dst_file_fd;
    dst_fid.deviceID = device_id;

    // 步骤 1：源文件 -> GPU（经 fgds_read）
    printf("Step 1: Reading from source file to GPU memory via fgds_read...\n");
    result = fgds_read(src_fid, gpu_buffer, /*buf_offset*/ 0, (ssize_t)file_size, /*f_offset*/ 0);
    if (result < 0 || (size_t)result != file_size) {
        printf("fgds_read failed: ret %ld, expected %zu\n", result, file_size);
        ret = 1;
        goto cleanup;
    }

    // 校验 GPU buffer 与源文件一致（主机侧仅用 pread 做对比）
    {
        void *host_from_file = NULL;
        void *host_from_gpu  = NULL;
        ssize_t vread;
        int verr;

        if (posix_memalign(&host_from_file, 4096, file_size) != 0 ||
            posix_memalign(&host_from_gpu, 4096, file_size) != 0) {
            printf("verify gpu_buffer: posix_memalign failed\n");
            if (host_from_file) free(host_from_file);
            if (host_from_gpu)  free(host_from_gpu);
            ret = 1;
            goto cleanup;
        }

        vread = pread_full(src_file_fd, host_from_file, file_size, 0);
        if (vread < 0 || (size_t)vread != file_size) {
            printf("verify gpu_buffer: pread source file failed: %ld\n", vread);
            free(host_from_file);
            free(host_from_gpu);
            ret = 1;
            goto cleanup;
        }

        cudaError_t vcuda = cudaMemcpy(host_from_gpu, gpu_buffer, file_size, cudaMemcpyDeviceToHost);
        if (vcuda != cudaSuccess) {
            printf("verify gpu_buffer: cudaMemcpy failed: %s\n", cudaGetErrorString(vcuda));
            free(host_from_file);
            free(host_from_gpu);
            ret = 1;
            goto cleanup;
        }

        verr = memcmp(host_from_file, host_from_gpu, file_size);
        if (verr == 0) {
            printf("verify gpu_buffer: PASSED, gpu_buffer content == source file content\n");
        } else {
            printf("verify gpu_buffer: FAILED, gpu_buffer content != source file content\n");
        }

        free(host_from_file);
        free(host_from_gpu);
    }

    // 步骤 2：GPU -> 目标文件（经 fgds_write）
    printf("Step 2: Writing from GPU memory to destination file via fgds_write...\n");
    result = fgds_write(dst_fid, gpu_buffer, /*buf_offset*/ 0, (ssize_t)file_size, /*f_offset*/ 0);
    if (result < 0 || (size_t)result != file_size) {
        printf("fgds_write failed: ret %ld, expected %zu\n", result, file_size);
        ret = 1;
        goto cleanup;
    }

    fsync(dst_file_fd);

    // 步骤 3：比较源文件与目标文件内容
    printf("Step 3: Verifying file content consistency...\n");

    ret = posix_memalign(&src_buffer, 4096, file_size);
    if (ret != 0) {
        printf("posix_memalign for src_buffer failed\n");
        ret = 1;
        goto cleanup;
    }
    ret = posix_memalign(&dst_buffer, 4096, file_size);
    if (ret != 0) {
        printf("posix_memalign for dst_buffer failed\n");
        ret = 1;
        goto cleanup;
    }

    close(src_file_fd);
    close(dst_file_fd);
    src_file_fd = open(file_path, O_RDONLY | O_DIRECT, 0644);
    if (src_file_fd < 0) {
        perror("reopen source file error");
        ret = 1;
        goto cleanup;
    }
    dst_file_fd = open(dst_file_path, O_RDONLY | O_DIRECT, 0644);
    if (dst_file_fd < 0) {
        perror("open destination file for verification error");
        ret = 1;
        goto cleanup;
    }

    result = pread_full(src_file_fd, src_buffer, file_size, 0);
    if (result < 0 || (size_t)result != file_size) {
        printf("read source file for verification failed: %ld\n", result);
        ret = 1;
        goto cleanup;
    }
    result = pread_full(dst_file_fd, dst_buffer, file_size, 0);
    if (result < 0 || (size_t)result != file_size) {
        printf("read destination file for verification failed: %ld\n", result);
        ret = 1;
        goto cleanup;
    }

    cmp_result = memcmp(src_buffer, dst_buffer, file_size);
    if (cmp_result == 0) {
        printf("Verification PASSED: Source and destination files are identical!\n");
        ret = 0;
    } else {
        printf("Verification FAILED: Files differ\n");
        for (size_t i = 0; i < file_size; i++) {
            if (((char*)src_buffer)[i] != ((char*)dst_buffer)[i]) {
                printf("First difference at offset %zu: src=0x%02x, dst=0x%02x\n",
                       i, ((unsigned char*)src_buffer)[i], ((unsigned char*)dst_buffer)[i]);
                break;
            }
        }
        ret = 1;
    }

cleanup:
    if (gpu_buffer != NULL && target_addr != NULL) {
        int dereg_ret = fgds_deregmem(device_id, gpu_buffer, file_size);
        printf("fgds deregmem ret: %d\n", dereg_ret);
    }
    if (gpu_buffer != NULL) {
        cudaFree(gpu_buffer);
    }
    fgds_close(device_id);
    if (src_file_fd >= 0) {
        close(src_file_fd);
    }
    if (dst_file_fd >= 0) {
        close(dst_file_fd);
    }
    if (src_buffer != NULL) {
        free(src_buffer);
    }
    if (dst_buffer != NULL) {
        free(dst_buffer);
    }
    if (ret != 0 && dst_file_path != NULL) {
        unlink(dst_file_path);
        printf("Cleaned up destination file due to error\n");
    } else if (ret == 0) {
        printf("Destination file saved as: %s\n", dst_file_path);
    }
    printf("fgds_check_1 test success\n\n");

    return ret;
}

// 1GB 随机数据 NVMe 往返校验：GPU<->NVMe IO 走 fgds_write / fgds_read。
int fgds_check_2() {
    const size_t gpu_buf_size = 1UL * 1024 * 1024 * 1024; // 1GB
    const char *test_file_path = "/data/fgds_test_rw";

    void *gpu_buf_src = NULL;
    void *gpu_buf_dst = NULL;
    void *host_random  = NULL;
    void *host_from_src = NULL;
    void *host_from_dst = NULL;
    void *target_addr_src = NULL;
    void *target_addr_dst = NULL;
    int fd = -1;
    int ret = 0;
    int diff = 0;
    unsigned int seed = 123456789;
    unsigned char *p = NULL;
    ssize_t io_ret;
    fgds_fileid_t fid = {};

    printf("fgds_check_2: start, gpu_buf_size = %zu bytes (%.2f GB), file: %s\n",
           gpu_buf_size, gpu_buf_size / (1024.0 * 1024.0 * 1024.0), test_file_path);

    ret = fgds_open(device_id);
    if (ret != 0) {
        printf("fgds_open failed: %d\n", ret);
        return 1;
    }

    cudaError_t cuda_ret;
    cuda_ret = cudaMalloc(&gpu_buf_src, gpu_buf_size);
    if (cuda_ret != cudaSuccess) {
        printf("cudaMalloc gpu_buf_src failed: %s\n", cudaGetErrorString(cuda_ret));
        ret = 1;
        goto cleanup;
    }
    cuda_ret = cudaMalloc(&gpu_buf_dst, gpu_buf_size);
    if (cuda_ret != cudaSuccess) {
        printf("cudaMalloc gpu_buf_dst failed: %s\n", cudaGetErrorString(cuda_ret));
        ret = 1;
        goto cleanup;
    }

    if (posix_memalign(&host_random, 4096, gpu_buf_size) != 0 ||
        posix_memalign(&host_from_src, 4096, gpu_buf_size) != 0 ||
        posix_memalign(&host_from_dst, 4096, gpu_buf_size) != 0) {
        printf("fgds_check_2: posix_memalign failed\n");
        ret = 1;
        goto cleanup;
    }

    p = (unsigned char *)host_random;
    for (size_t i = 0; i < gpu_buf_size; ++i) {
        seed = seed * 1103515245 + 12345;
        p[i] = (unsigned char)((seed >> 16) & 0xFF);
    }

    cuda_ret = cudaMemcpy(gpu_buf_src, host_random, gpu_buf_size, cudaMemcpyHostToDevice);
    if (cuda_ret != cudaSuccess) {
        printf("fgds_check_2: cudaMemcpy host_random -> gpu_buf_src failed: %s\n",
               cudaGetErrorString(cuda_ret));
        ret = 1;
        goto cleanup;
    }
    cudaStreamSynchronize(0);

    ret = fgds_regmem(device_id, gpu_buf_src, gpu_buf_size, &target_addr_src);
    if (ret) {
        printf("fgds_check_2: fgds_regmem src failed\n");
        ret = 1;
        goto cleanup;
    }

    ret = fgds_regmem(device_id, gpu_buf_dst, gpu_buf_size, &target_addr_dst);
    if (ret) {
        printf("fgds_check_2: fgds_regmem dst failed\n");
        ret = 1;
        goto cleanup;
    }

    unlink(test_file_path);
    fd = open(test_file_path, O_CREAT | O_RDWR | O_TRUNC | O_DIRECT, 0644);
    if (fd < 0) {
        perror("fgds_check_2: open test file error");
        ret = 1;
        goto cleanup;
    }

    if (ftruncate(fd, (off_t)gpu_buf_size) != 0) {
        perror("fgds_check_2: ftruncate test file error");
        ret = 1;
        goto cleanup;
    }

    fid.fd = fd;
    fid.deviceID = device_id;

    // 步骤 1：GPU src -> 文件（经 fgds_write）
    printf("fgds_check_2 Step 1: fgds_write from GPU (src) to file...\n");
    io_ret = fgds_write(fid, gpu_buf_src, /*buf_offset*/ 0, (ssize_t)gpu_buf_size, /*f_offset*/ 0);
    if (io_ret < 0 || (size_t)io_ret != gpu_buf_size) {
        printf("fgds_check_2: fgds_write failed, ret = %ld, expected = %zu\n",
               io_ret, gpu_buf_size);
        ret = 1;
        goto cleanup;
    }
    fsync(fd);

    // 步骤 2：文件 -> GPU dst（经 fgds_read）
    printf("fgds_check_2 Step 2: fgds_read from file to GPU (dst)...\n");
    io_ret = fgds_read(fid, gpu_buf_dst, /*buf_offset*/ 0, (ssize_t)gpu_buf_size, /*f_offset*/ 0);
    if (io_ret < 0 || (size_t)io_ret != gpu_buf_size) {
        printf("fgds_check_2: fgds_read failed, ret = %ld, expected = %zu\n",
               io_ret, gpu_buf_size);
        ret = 1;
        goto cleanup;
    }

    // 步骤 3：在主机侧比较两块 GPU buffer
    printf("fgds_check_2 Step 3: compare GPU src buffer and dst buffer...\n");
    cuda_ret = cudaMemcpy(host_from_src, gpu_buf_src, gpu_buf_size, cudaMemcpyDeviceToHost);
    if (cuda_ret != cudaSuccess) {
        printf("fgds_check_2: cudaMemcpy gpu_buf_src -> host_from_src failed: %s\n",
               cudaGetErrorString(cuda_ret));
        ret = 1;
        goto cleanup;
    }
    cuda_ret = cudaMemcpy(host_from_dst, gpu_buf_dst, gpu_buf_size, cudaMemcpyDeviceToHost);
    if (cuda_ret != cudaSuccess) {
        printf("fgds_check_2: cudaMemcpy gpu_buf_dst -> host_from_dst failed: %s\n",
               cudaGetErrorString(cuda_ret));
        ret = 1;
        goto cleanup;
    }

    diff = memcmp(host_from_src, host_from_dst, gpu_buf_size);
    if (diff == 0) {
        printf("fgds_check_2: PASSED, GPU src buffer == GPU dst buffer (after NVMe round-trip)\n");
        ret = 0;
    } else {
        printf("fgds_check_2: FAILED, GPU src buffer != GPU dst buffer\n");
        ret = 1;
    }

cleanup:
    if (target_addr_src != NULL && gpu_buf_src != NULL) {
        fgds_deregmem(device_id, gpu_buf_src, gpu_buf_size);
    }
    if (target_addr_dst != NULL && gpu_buf_dst != NULL) {
        fgds_deregmem(device_id, gpu_buf_dst, gpu_buf_size);
    }
    if (gpu_buf_src != NULL) {
        cudaFree(gpu_buf_src);
    }
    if (gpu_buf_dst != NULL) {
        cudaFree(gpu_buf_dst);
    }
    if (fd >= 0) {
        close(fd);
    }
    if (host_random != NULL) {
        free(host_random);
    }
    if (host_from_src != NULL) {
        free(host_from_src);
    }
    if (host_from_dst != NULL) {
        free(host_from_dst);
    }

    fgds_close(device_id);
    printf("fgds_check_2 test success\n\n");
    return ret;
}

// 用法：./fgds_smoke <gpu_id> <file_path>
// 注意：传入的测试文件大小必须是 64KB 的正整数倍
//   可用如下 dd 命令生成一个 4MB（=64×64KB）的测试文件：
//     dd if=/dev/urandom of=/data/test bs=1M count=4
int main(int argc, char* argv[]) {
    if (argc < 3) {
        printf("Usage: %s <gpu_id> <file_path>\n", argv[0]);
        return 1;
    }
    device_id = std::atoi(argv[1]);
    file_path = argv[2];
    cudaSetDevice(device_id);

    // 依次执行 2 个一致性校验，对照来看：
    //   fgds_check_1（文件 -> GPU -> 文件，比较两份文件）：
    //     源文件 --fgds_read--> GPU 显存 --fgds_write--> 目标文件，再读回两份文件到 host 做 memcmp。
    //     校验：源文件内容 == 目标文件内容（覆盖 >1GB 的大文件场景）。
    //   fgds_check_2（GPU -> 文件 -> GPU，比较两块显存）：
    //     GPU src 显存 --fgds_write--> NVMe 文件 --fgds_read--> GPU dst 显存，再拷回 host 做 memcmp。
    //     校验：GPU src buffer == GPU dst buffer（1GB 随机数据往返前后一致）。
    // todo: 2个check文件可抽取一些可复用的基础函数
    int rc = 0;
    printf("fgds_check_1 start\n");
    rc += (fgds_check_1() != 0);
    printf("fgds_check_2 start\n");
    rc += (fgds_check_2() != 0);

    if (rc != 0) {
        printf("fgds_smoke: %d check(s) FAILED\n", rc);
        return 1;
    }
    printf("fgds_smoke: all checks PASSED\n");
    return 0;
}
