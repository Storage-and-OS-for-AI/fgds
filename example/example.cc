/*
 * fgds 最小使用示例。
 *
 * fgds 让文件与 GPU 显存之间的 IO 不必经过主机内存（GPUDirect Storage 封装）。
 * 通用流程：fgds_open -> cudaMalloc -> fgds_regmem -> 读写 -> fgds_deregmem -> fgds_close
 *
 * fgds_demo            注册后拿到 target_addr，直接当作 pread/pwrite 的缓冲区。
 * fgds_read_write_demo 使用 fgds_read/fgds_write 接口完成 文件<->显存 的搬移。
 *
 * 用法：./example <gpu_id> <file_path>
 * 注意：file_path 需在支持 O_DIRECT 的文件系统上（本地 NVMe/SSD，非 /tmp），且长度 >= 4MB。
 * 注意：demo 只读写 file_path 开头的前 4MB（IO_SIZE），并不会搬移整个文件、也不校验内容一致性。校验读写内容一致性请使用 tests/fgds_smoke.cc
 * 注意：demo 会在 file_path 同目录下生成一个带 .out 后缀的新文件，不会回写原文件。
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <fcntl.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include "fgds.h"

// 本示例以读写一个 4MB 文件为例：两个 demo 都只对文件开头的前 4MB 做一次读、一次写。
// 4MB 是 64KB 的整数倍，满足 fgds_regmem / O_DIRECT 的对齐要求。
#define IO_SIZE (4 * 1024 * 1024)

// 在输入 file_path 的同目录下生成一个新文件（file_path + suffix），避免回写原文件。
static void make_out_path(const char *in, const char *suffix, char *out, size_t n) {
    size_t suffix_len = strlen(suffix);
    strncpy(out, in, n - suffix_len - 1);
    out[n - suffix_len - 1] = '\0';
    strcat(out, suffix);
}

// 示例 1：注册显存后，把返回的 target_addr 直接作为 pread/pwrite 的缓冲区。
// 从文件开头的前 4MB 读入 GPU 显存，再写回同目录下的新文件。
static int fgds_demo(int device_id, const char *in_path) {
    const size_t io_size = IO_SIZE;
    char out_path[PATH_MAX];
    make_out_path(in_path, ".fgds_demo.out", out_path, sizeof(out_path));

    int ret = 1;
    int fd_in = -1, fd_out = -1;
    void *gpu_buf = NULL, *target_addr = NULL;
    cudaError_t cuda_err;

    fd_in = open(in_path, O_RDONLY | O_DIRECT);
    if (fd_in < 0) {
        perror("open input file");
        goto cleanup;
    }
    fd_out = open(out_path, O_RDWR | O_CREAT | O_TRUNC | O_DIRECT, 0644);
    if (fd_out < 0) {
        perror("open output file");
        goto cleanup;
    }

    if (fgds_open(device_id) != 0) {
        fprintf(stderr, "fgds_open failed\n");
        goto cleanup;
    }
    if ((cuda_err = cudaMalloc(&gpu_buf, io_size)) != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cuda_err));
        goto cleanup;
    }
    if (fgds_regmem(device_id, gpu_buf, io_size, &target_addr) != 0) {
        fprintf(stderr, "fgds_regmem failed\n");
        goto cleanup;
    }

    // target_addr 是注册后得到的映射地址，可直接作为 pread/pwrite 的缓冲区
    if (pread(fd_in, target_addr, io_size, 0) != (ssize_t)io_size) {
        perror("pread to GPU memory");
        goto cleanup;
    }
    if (pwrite(fd_out, target_addr, io_size, 0) != (ssize_t)io_size) {
        perror("pwrite from GPU memory");
        goto cleanup;
    }

    printf("fgds_demo success: pread/pwrite -> %s\n", out_path);
    ret = 0;

cleanup:
    if (target_addr) fgds_deregmem(device_id, gpu_buf, io_size);
    if (gpu_buf) cudaFree(gpu_buf);
    fgds_close(device_id);
    if (fd_in >= 0) close(fd_in);
    if (fd_out >= 0) close(fd_out);
    return ret;
}

// 示例 2：使用 fgds_read/fgds_write 把文件开头的前 4MB 搬进显存，再写入同目录下的新文件。
static int fgds_read_write_demo(int device_id, const char *in_path) {
    const size_t io_size = IO_SIZE;
    char out_path[PATH_MAX];
    make_out_path(in_path, ".fgds_read_write_demo.out", out_path, sizeof(out_path));

    int ret = 1;
    int fd_in = -1, fd_out = -1;
    void *gpu_buf = NULL, *target_addr = NULL;
    cudaError_t cuda_err;
    fgds_fileid_t fid_in, fid_out;

    fd_in = open(in_path, O_RDONLY | O_DIRECT);
    if (fd_in < 0) {
        perror("open input file");
        goto cleanup;
    }
    fd_out = open(out_path, O_RDWR | O_CREAT | O_TRUNC | O_DIRECT, 0644);
    if (fd_out < 0) {
        perror("open output file");
        goto cleanup;
    }

    if (fgds_open(device_id) != 0) {
        fprintf(stderr, "fgds_open failed\n");
        goto cleanup;
    }
    if ((cuda_err = cudaMalloc(&gpu_buf, io_size)) != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cuda_err));
        goto cleanup;
    }
    // fgds_read/fgds_write 直接使用 gpu_buf；target_addr 只在走 pread/pwrite 原语时才需要
    if (fgds_regmem(device_id, gpu_buf, io_size, &target_addr) != 0) {
        fprintf(stderr, "fgds_regmem failed\n");
        goto cleanup;
    }

    fid_in = {fd_in, device_id};
    fid_out = {fd_out, device_id};
    if (fgds_read(fid_in, gpu_buf, 0, io_size, 0) != (ssize_t)io_size) {
        fprintf(stderr, "fgds_read failed\n");
        goto cleanup;
    }
    if (fgds_write(fid_out, gpu_buf, 0, io_size, 0) != (ssize_t)io_size) {
        fprintf(stderr, "fgds_write failed\n");
        goto cleanup;
    }

    printf("fgds_read_write_demo success: fgds_read/fgds_write -> %s\n", out_path);
    ret = 0;

cleanup:
    if (target_addr) fgds_deregmem(device_id, gpu_buf, io_size);
    if (gpu_buf) cudaFree(gpu_buf);
    fgds_close(device_id);
    if (fd_in >= 0) close(fd_in);
    if (fd_out >= 0) close(fd_out);
    return ret;
}

// 两个 demo 均只读写文件开头的前 4MB（IO_SIZE），生成测试文件的例子（文件需 >= 4MB）：
// 生成一个4MB文件的例子：
// dd if=/dev/urandom of=/data/test bs=1M count=4
//
// fgds_demo是基于pread/pwrite读写的，fgds_read_write_demo是基于fgds_read/write读写的。对于fgds，这2组api都可以用于读写。
// fgds_read/write 与 pread/pwrite 的区别与联系：
//   - fgds_read/write 内部最终会调用 pread/pwrite（在注册后的映射地址上做真正的 IO）。
//   - 裸 pread/pwrite 一次只能读写 1GB 以内的数据（映射按 1GB 分段、段间不连续）。
//   - fgds_read/write 内部实现了 io_uring 流水线，在较大 IO 上能显著提升吞吐。
// 日常使用建议直接用 fgds_read/write，pread/pwrite 留给底层/自定义场景。
int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: %s <gpu_id> <file_path>\n", argv[0]);
        return 1;
    }
    int device_id = std::atoi(argv[1]);
    const char *file_path = argv[2];

    cudaError_t cuda_err = cudaSetDevice(device_id);
    if (cuda_err != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice(%d) failed: %s\n", device_id, cudaGetErrorString(cuda_err));
        return 1;
    }

    if (fgds_demo(device_id, file_path) != 0)
        return 1;
    if (fgds_read_write_demo(device_id, file_path) != 0)
        return 1;

    printf("all demos passed\n");
    return 0;
}
