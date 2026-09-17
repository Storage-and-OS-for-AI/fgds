/*
 * fgds_multi_device: 验证 libfgds 的多设备支持（阶段一回归测试）。单线程多设备，
 * 不涉及多线程并发（库的 open/close 尚未做线程安全）。
 *
 * 覆盖：
 *   1. 多设备同时打开/注册：先把所有可用 /dev/fgds_devN 一起 fgds_open，
 *      再同时 fgds_regmem，最后一起 deregmem/close。用于回归“只有第一个
 *      open 的设备可用”的老问题（老代码里第二个设备 open 返回 0 但未真正打开）。
 *      升序、降序各跑一遍验证顺序无关。
 *   2. close 隔离：两块设备都 open+regmem 后关闭其中一块，另一块仍能做真实 I/O，
 *      且被关闭的设备可以重新 open 并再次 regmem。
 *   3. 交错 I/O：所有设备同时打开并注册，先各自写文件，再统一读回校验；
 *      按 256KB / 1MB / 4MB 三种尺寸覆盖 direct 与 io_uring 读写路径。
 *   4. 负例与边界：
 *        - 越界 id（fgds_open/regmem/close/read/write）返回错误且不崩溃；
 *        - 合法 id 但未 open 时 regmem/read/write 返回 ENODEV（拦截发生在碰 fd 之前）；
 *        - fgds_open(-1) 作为越界 id 被 EINVAL 拒绝（-1 不再是“打开全部”哨兵）；
 *        - 非 root 下用 chmod 000 制造 open 失败，恢复后 open 必须成功（可重试）。
 *
 * 用法：
 *   ./fgds_multi_device
 *   ./fgds_multi_device /path/on/nvme/base
 *       (需在支持 O_DIRECT 的文件系统上，会生成 <base>.dev<id>)
 *
 * 退出码：0 全部通过；非 0 表示有失败项。
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <cuda_runtime.h>
#include "fgds.h"

#define FGDS_TEST_MAX_DEV 16
/* 库当前支持的最大设备数，必须与 libfgds/fgds.cc 的 FGDS_MAX_DEVICES 保持一致。
 * 内核模块上限是 16（MAX_DEV_NUM），阶段三才会把两者统一；在此之前枚举不能
 * 超过本值，否则在 >8 卡且模块暴露了 dev8~15 的机器上，fgds_open(id>=8) 会返回
 * EINVAL，被误判为测试失败。阶段三落地后把这里改成 16 即可。 */
#define FGDS_LIB_MAX_DEV 8
/* 三种尺寸分别覆盖 direct / 混合 / io_uring 读写路径：
 *   256KB: read<512KB 走 direct, write<2MB 走 direct
 *   1MB  : read>=512KB 走 uring, write<2MB 走 direct
 *   4MB  : read/write 都走 io_uring
 * 均为 64KB 的正整数倍，满足 fgds_regmem 对齐要求。 */
#define FGDS_TEST_IO_DIRECT (256 * 1024)
#define FGDS_TEST_IO_MIXED (1024 * 1024)
#define FGDS_TEST_IO_URING (4 * 1024 * 1024)

static int g_fail = 0;

struct io_dev {
    int id;
    bool opened;
    bool alloc_src;
    bool alloc_dst;
    bool reg_src;
    bool reg_dst;
    void *gpu_src;
    void *gpu_dst;
    void *target_src;
    void *target_dst;
    void *host_src;
    void *host_dst;
    int fd;
};

static bool fgds_node_exists(int id) {
    char path[64];
    snprintf(path, sizeof(path), "/dev/fgds_dev%d", id);
    return access(path, F_OK) == 0;
}

static int enumerate_devices(int *ids, int max_ids) {
    int n = 0;
    /* 只枚举库支持的范围，避免在 >8 卡机器上把 EINVAL 误判为失败 */
    for (int id = 0; id < FGDS_LIB_MAX_DEV && n < max_ids; ++id) {
        if (fgds_node_exists(id))
            ids[n++] = id;
    }
    return n;
}

static void fill_pattern(unsigned char *p, size_t n, int dev) {
    unsigned int seed = 0x12345678u + (unsigned int)dev;
    for (size_t i = 0; i < n; ++i) {
        seed = seed * 1103515245u + 12345u;
        p[i] = (unsigned char)((seed >> 16) & 0xFF);
    }
}

/* 轻量探针：在指定设备上 regmem 一小块并立刻 deregmem，证明会话仍然可用。 */
static int regmem_probe(int dev, size_t size) {
    void *gpu = NULL, *target = NULL;
    int ret = -1;
    if (cudaSetDevice(dev) != cudaSuccess)
        return -1;
    if (cudaMalloc(&gpu, size) != cudaSuccess)
        return -1;
    if (fgds_regmem(dev, gpu, size, &target) != 0)
        goto out;
    fgds_deregmem(dev, gpu, size);
    ret = 0;
out:
    cudaFree(gpu);
    return ret;
}

/* 一次性打开所有设备并同时注册显存，再统一释放。
 * reverse=true 时按降序处理，用于验证顺序无关。 */
static int check_multi_open_register(const int *ids, int n, bool reverse) {
    struct io_dev devs[FGDS_TEST_MAX_DEV];
    char order[128] = {0};
    const size_t size = FGDS_TEST_IO_MIXED;
    int fails = 0;

    memset(devs, 0, sizeof(devs));
    for (int k = 0; k < n; ++k) {
        int idx = reverse ? (n - 1 - k) : k;
        devs[k].id = ids[idx];
        devs[k].fd = -1;
    }

    for (int k = 0; k < n; ++k)
        snprintf(order + strlen(order), sizeof(order) - strlen(order), "%s%d",
                 k ? "," : "", devs[k].id);
    printf("---- multi open/register pass (%s order: %s) ----\n",
           reverse ? "descending" : "ascending", order);

    /* 步骤 1：同时打开全部设备 */
    for (int k = 0; k < n; ++k) {
        errno = 0;
        int ret = fgds_open(devs[k].id);
        if (ret != 0) {
            printf("  dev %2d: fgds_open FAILED (ret=%d, %s)\n",
                   devs[k].id, ret, strerror(errno));
            fails++;
            continue;
        }
        devs[k].opened = true;
    }

    /* 步骤 2：为每个已打开设备分配显存并注册（保持全部同时打开） */
    for (int k = 0; k < n; ++k) {
        if (!devs[k].opened)
            continue;
        if (cudaSetDevice(devs[k].id) != cudaSuccess) {
            printf("  dev %2d: cudaSetDevice FAILED\n", devs[k].id);
            fails++;
            continue;
        }
        if (cudaMalloc(&devs[k].gpu_src, size) != cudaSuccess) {
            printf("  dev %2d: cudaMalloc FAILED\n", devs[k].id);
            fails++;
            continue;
        }
        devs[k].alloc_src = true;

        errno = 0;
        int rr = fgds_regmem(devs[k].id, devs[k].gpu_src, size, &devs[k].target_src);
        if (rr != 0) {
            printf("  dev %2d: fgds_regmem FAILED (ret=%d, %s)  <-- multi-device bug\n",
                   devs[k].id, rr, strerror(errno));
            fails++;
            continue;
        }
        devs[k].reg_src = true;
        printf("  dev %2d: open + regmem OK\n", devs[k].id);
    }

    /* 步骤 3：统一释放 */
    for (int k = 0; k < n; ++k) {
        if (devs[k].reg_src)
            fgds_deregmem(devs[k].id, devs[k].gpu_src, size);
        if (devs[k].alloc_src)
            cudaFree(devs[k].gpu_src);
        if (devs[k].opened)
            fgds_close(devs[k].id);
    }
    return fails;
}

/* close 隔离：全部 open+regmem 后关闭设备 0，设备 1 仍应可做真实 I/O，
 * 随后设备 0 应能重新 open 并 regmem。 */
static int check_close_isolation(const int *ids, int n, const char *base) {
    struct io_dev d[FGDS_TEST_MAX_DEV];
    const size_t size = FGDS_TEST_IO_DIRECT;
    int fails = 0;

    printf("---- close isolation ----\n");
    if (n < 2) {
        printf("  SKIP: need >= 2 devices\n");
        return 0;
    }

    memset(d, 0, sizeof(d));
    for (int i = 0; i < n; ++i) {
        d[i].id = ids[i];
        d[i].fd = -1;
    }

    for (int i = 0; i < n; ++i) {
        if (fgds_open(d[i].id) != 0) {
            printf("  dev %d: open FAILED\n", d[i].id);
            fails++;
            continue;
        }
        d[i].opened = true;
    }

    for (int i = 0; i < n; ++i) {
        if (!d[i].opened)
            continue;
        if (cudaSetDevice(d[i].id) != cudaSuccess ||
            cudaMalloc(&d[i].gpu_src, size) != cudaSuccess) {
            printf("  dev %d: cudaMalloc src FAILED\n", d[i].id);
            fails++;
            continue;
        }
        d[i].alloc_src = true;
        if (cudaMalloc(&d[i].gpu_dst, size) != cudaSuccess ||
            posix_memalign(&d[i].host_src, 4096, size) != 0 ||
            posix_memalign(&d[i].host_dst, 4096, size) != 0) {
            printf("  dev %d: alloc FAILED\n", d[i].id);
            fails++;
            continue;
        }
        d[i].alloc_dst = true;
        if (fgds_regmem(d[i].id, d[i].gpu_src, size, &d[i].target_src) != 0 ||
            fgds_regmem(d[i].id, d[i].gpu_dst, size, &d[i].target_dst) != 0) {
            printf("  dev %d: regmem FAILED\n", d[i].id);
            fails++;
            continue;
        }
        d[i].reg_src = true;
        d[i].reg_dst = true;
        fill_pattern((unsigned char *)d[i].host_src, size, d[i].id);
        cudaMemcpy(d[i].gpu_src, d[i].host_src, size, cudaMemcpyHostToDevice);
    }

    printf("  closing dev %d (peer must remain usable)\n", d[0].id);
    fgds_close(d[0].id);
    d[0].opened = false;
    d[0].reg_src = false;
    d[0].reg_dst = false;

    if (base != NULL) {
        char path[512];
        snprintf(path, sizeof(path), "%s.dev%d", base, d[1].id);
        unlink(path);
        d[1].fd = open(path, O_CREAT | O_RDWR | O_TRUNC | O_DIRECT, 0644);
        if (d[1].fd < 0) {
            printf("  dev %d: open %s FAILED\n", d[1].id, path);
            fails++;
        } else {
            fgds_fileid_t fid = { d[1].fd, d[1].id };
            ssize_t w, r;
            ftruncate(d[1].fd, (off_t)size);
            w = fgds_write(fid, d[1].gpu_src, 0, (ssize_t)size, 0);
            fsync(d[1].fd);
            r = fgds_read(fid, d[1].gpu_dst, 0, (ssize_t)size, 0);
            if (w != (ssize_t)size || r != (ssize_t)size) {
                printf("  dev %d: real I/O after peer close FAILED\n", d[1].id);
                fails++;
            } else if (cudaMemcpy(d[1].host_dst, d[1].gpu_dst, size,
                                  cudaMemcpyDeviceToHost) != cudaSuccess ||
                       memcmp(d[1].host_src, d[1].host_dst, size) != 0) {
                printf("  dev %d: real I/O content mismatch\n", d[1].id);
                fails++;
            } else {
                printf("  dev %d: real I/O while dev %d closed OK\n", d[1].id, d[0].id);
            }
        }
    } else {
        if (regmem_probe(d[1].id, size) != 0) {
            printf("  dev %d: regmem probe after peer close FAILED\n", d[1].id);
            fails++;
        } else {
            printf("  dev %d: regmem probe while dev %d closed OK\n", d[1].id, d[0].id);
        }
    }

    if (fgds_open(d[0].id) != 0) {
        printf("  dev %d: reopen FAILED\n", d[0].id);
        fails++;
    } else {
        d[0].opened = true;
        if (regmem_probe(d[0].id, size) != 0) {
            printf("  dev %d: not usable after reopen\n", d[0].id);
            fails++;
        } else {
            printf("  dev %d: reopen + regmem probe OK\n", d[0].id);
        }
    }

    for (int i = 0; i < n; ++i) {
        if (d[i].fd >= 0)
            close(d[i].fd);
        if (d[i].opened) {
            if (d[i].reg_src)
                fgds_deregmem(d[i].id, d[i].gpu_src, size);
            if (d[i].reg_dst)
                fgds_deregmem(d[i].id, d[i].gpu_dst, size);
        }
        if (d[i].alloc_src)
            cudaFree(d[i].gpu_src);
        if (d[i].alloc_dst)
            cudaFree(d[i].gpu_dst);
        free(d[i].host_src);
        free(d[i].host_dst);
        if (d[i].opened)
            fgds_close(d[i].id);
    }
    return fails;
}

/* 交错 I/O：所有设备同时打开并注册，先各自写文件、再统一读回校验。
 * 内容 seed 按设备区分，若串了设备必然被 memcmp 抓到。 */
static int check_interleaved_roundtrip(const int *ids, int n, const char *base, size_t size) {
    struct io_dev d[FGDS_TEST_MAX_DEV];
    int fails = 0;

    printf("---- interleaved roundtrip (all open, size=%zu) ----\n", size);
    if (base == NULL) {
        printf("  SKIP: no file base given\n");
        return 0;
    }

    memset(d, 0, sizeof(d));
    for (int i = 0; i < n; ++i) {
        d[i].id = ids[i];
        d[i].fd = -1;
    }

    for (int i = 0; i < n; ++i) {
        if (fgds_open(d[i].id) != 0) {
            printf("  dev %d: open FAILED\n", d[i].id);
            fails++;
            continue;
        }
        d[i].opened = true;
        if (cudaSetDevice(d[i].id) != cudaSuccess ||
            cudaMalloc(&d[i].gpu_src, size) != cudaSuccess) {
            printf("  dev %d: cudaMalloc src FAILED\n", d[i].id);
            fails++;
            continue;
        }
        d[i].alloc_src = true;
        if (cudaMalloc(&d[i].gpu_dst, size) != cudaSuccess ||
            posix_memalign(&d[i].host_src, 4096, size) != 0 ||
            posix_memalign(&d[i].host_dst, 4096, size) != 0) {
            printf("  dev %d: alloc FAILED\n", d[i].id);
            fails++;
            continue;
        }
        d[i].alloc_dst = true;
        if (fgds_regmem(d[i].id, d[i].gpu_src, size, &d[i].target_src) != 0 ||
            fgds_regmem(d[i].id, d[i].gpu_dst, size, &d[i].target_dst) != 0) {
            printf("  dev %d: regmem FAILED\n", d[i].id);
            fails++;
            continue;
        }
        d[i].reg_src = true;
        d[i].reg_dst = true;
        fill_pattern((unsigned char *)d[i].host_src, size, d[i].id);
        cudaMemcpy(d[i].gpu_src, d[i].host_src, size, cudaMemcpyHostToDevice);
    }

    /* 阶段 A：所有设备（同时保持打开）各自写入文件 */
    for (int i = 0; i < n; ++i) {
        char path[512];
        if (!d[i].reg_src)
            continue;
        snprintf(path, sizeof(path), "%s.dev%d", base, d[i].id);
        unlink(path);
        d[i].fd = open(path, O_CREAT | O_RDWR | O_TRUNC | O_DIRECT, 0644);
        if (d[i].fd < 0) {
            printf("  dev %d: open %s FAILED\n", d[i].id, path);
            fails++;
            continue;
        }
        ftruncate(d[i].fd, (off_t)size);
        fgds_fileid_t fid = { d[i].fd, d[i].id };
        if (fgds_write(fid, d[i].gpu_src, 0, (ssize_t)size, 0) != (ssize_t)size) {
            printf("  dev %d: fgds_write FAILED\n", d[i].id);
            fails++;
        }
        fsync(d[i].fd);
    }

    /* 阶段 B：统一读回并校验 */
    for (int i = 0; i < n; ++i) {
        fgds_fileid_t fid;
        if (!d[i].reg_dst || d[i].fd < 0)
            continue;
        fid.fd = d[i].fd;
        fid.deviceID = d[i].id;
        if (fgds_read(fid, d[i].gpu_dst, 0, (ssize_t)size, 0) != (ssize_t)size) {
            printf("  dev %d: fgds_read FAILED\n", d[i].id);
            fails++;
            continue;
        }
        if (cudaMemcpy(d[i].host_dst, d[i].gpu_dst, size,
                       cudaMemcpyDeviceToHost) != cudaSuccess ||
            memcmp(d[i].host_src, d[i].host_dst, size) != 0) {
            printf("  dev %d: content mismatch FAILED\n", d[i].id);
            fails++;
            continue;
        }
        printf("  dev %d: roundtrip OK\n", d[i].id);
    }

    for (int i = 0; i < n; ++i) {
        if (d[i].fd >= 0)
            close(d[i].fd);
        if (d[i].opened) {
            if (d[i].reg_src)
                fgds_deregmem(d[i].id, d[i].gpu_src, size);
            if (d[i].reg_dst)
                fgds_deregmem(d[i].id, d[i].gpu_dst, size);
        }
        if (d[i].alloc_src)
            cudaFree(d[i].gpu_src);
        if (d[i].alloc_dst)
            cudaFree(d[i].gpu_dst);
        free(d[i].host_src);
        free(d[i].host_dst);
        if (d[i].opened)
            fgds_close(d[i].id);
    }
    return fails;
}

/* open 失败可重试：非 root 下 chmod 000 设备节点使 open 失败，恢复后必须成功。
 * root 会绕过权限检查，直接 skip。 */
static int check_open_retry(const int *ids, int n) {
    char path[64];
    struct stat st;

    printf("---- open failure retry ----\n");
    if (n < 1)
        return 0;
    if (geteuid() == 0) {
        printf("  SKIP: running as root, chmod won't block open\n");
        return 0;
    }

    snprintf(path, sizeof(path), "/dev/fgds_dev%d", ids[0]);
    if (stat(path, &st) != 0 || chmod(path, 0) != 0) {
        printf("  SKIP: cannot chmod %s\n", path);
        return 0;
    }

    errno = 0;
    int first = fgds_open(ids[0]);
    chmod(path, st.st_mode); /* 立即恢复权限 */

    if (first == 0) {
        printf("  FAILED: fgds_open succeeded while node mode 000\n");
        fgds_close(ids[0]);
        return 1;
    }
    printf("  open with mode 000 rejected (ret=%d, %s)\n", first, strerror(errno));

    errno = 0;
    if (fgds_open(ids[0]) != 0) {
        printf("  FAILED: fgds_open did not recover after restore (ret=%d, %s)\n",
               -1, strerror(errno));
        return 1;
    }
    fgds_close(ids[0]);
    printf("  open recovered after restore OK\n");
    return 0;
}

/* 幂等性：连续两次 fgds_open 都应返回 0；连续两次 fgds_close，第二次应为 0(no-op)。 */
static int check_idempotency(const int *ids, int n) {
    int fails = 0;

    printf("---- idempotent open/close ----\n");
    if (n < 1)
        return 0;

    if (fgds_open(ids[0]) != 0) {
        printf("  first fgds_open(%d) FAILED\n", ids[0]);
        fails++;
    } else if (fgds_open(ids[0]) != 0) {
        printf("  second fgds_open(%d) FAILED (should be idempotent)\n", ids[0]);
        fails++;
    } else {
        printf("  fgds_open(%d) twice OK\n", ids[0]);
    }

    if (fgds_close(ids[0]) != 0) {
        printf("  first fgds_close(%d) FAILED\n", ids[0]);
        fails++;
    } else if (fgds_close(ids[0]) != 0) {
        printf("  second fgds_close(%d) FAILED (should be no-op)\n", ids[0]);
        fails++;
    } else {
        printf("  fgds_close(%d) twice OK\n", ids[0]);
    }
    return fails;
}

/* 非法 id、未 open 的合法 id、open(-1) 的边界检查。 */
static int check_negative(const int *ids, int n) {
    int fails = 0;
    void *dummy = NULL;
    fgds_fileid_t bad = { -1, 99 };

    printf("---- invalid id / unopened id / open(-1) ----\n");

    errno = 0;
    int r99 = fgds_open(99);
    if (r99 == 0) {
        printf("  fgds_open(99) unexpectedly succeeded\n");
        fails++;
    } else {
        printf("  fgds_open(99) rejected (ret=%d, %s)\n", r99, strerror(errno));
    }

    errno = 0;
    if (fgds_regmem(99, (const void *)0x1, FGDS_TEST_IO_DIRECT, &dummy) == 0) {
        printf("  fgds_regmem(99) unexpectedly succeeded\n");
        fails++;
    } else {
        printf("  fgds_regmem(99) rejected (errno=%s)\n", strerror(errno));
    }

    if (fgds_close(99) == 0) {
        printf("  fgds_close(99) unexpectedly succeeded\n");
        fails++;
    } else {
        printf("  fgds_close(99) rejected\n");
    }

    /* read/write 的 deviceID=99 负例（direct 路径入口即被拦下） */
    errno = 0;
    if (fgds_read(bad, (void *)0x1, 0, 4096, 0) != -1) {
        printf("  fgds_read(deviceID=99) unexpectedly succeeded\n");
        fails++;
    } else {
        printf("  fgds_read(deviceID=99) rejected (errno=%s)\n", strerror(errno));
    }
    errno = 0;
    if (fgds_write(bad, (void *)0x1, 0, 4096, 0) != -1) {
        printf("  fgds_write(deviceID=99) unexpectedly succeeded\n");
        fails++;
    } else {
        printf("  fgds_write(deviceID=99) rejected (errno=%s)\n", strerror(errno));
    }

    /* 合法 id 但未 open：必须是 ENODEV，且在触碰 fd 之前就被拦下
     * （read/write 传 fd=-1，若先碰 fd 会得到 EBADF 而非 ENODEV）。 */
    for (int k = 0; k < n; ++k)
        fgds_close(ids[k]);

    dummy = NULL;
    errno = 0;
    if (fgds_regmem(ids[0], (const void *)0x1, FGDS_TEST_IO_DIRECT, &dummy) == 0 ||
        errno != ENODEV) {
        printf("  regmem on unopened dev %d not rejected with ENODEV\n", ids[0]);
        fails++;
    } else {
        printf("  regmem on unopened dev %d rejected with ENODEV\n", ids[0]);
    }

    fgds_fileid_t unopened = { -1, ids[0] };
    errno = 0;
    if (fgds_read(unopened, (void *)0x1, 0, 4096, 0) != -1 || errno != ENODEV) {
        printf("  read on unopened dev %d not rejected with ENODEV\n", ids[0]);
        fails++;
    } else {
        printf("  read on unopened dev %d rejected with ENODEV (before fd)\n", ids[0]);
    }
    errno = 0;
    if (fgds_write(unopened, (void *)0x1, 0, 4096, 0) != -1 || errno != ENODEV) {
        printf("  write on unopened dev %d not rejected with ENODEV\n", ids[0]);
        fails++;
    } else {
        printf("  write on unopened dev %d rejected with ENODEV (before fd)\n", ids[0]);
    }

    /* -1 不再是“打开所有设备”的哨兵：与其它越界 id 一样返回 EINVAL。
     * 需要打开全部设备的调用方自行枚举 /dev/fgds_dev*（见 enumerate_devices）。 */
    errno = 0;
    int rneg = fgds_open(-1);
    if (rneg == 0 || errno != EINVAL) {
        printf("  fgds_open(-1) not rejected with EINVAL (ret=%d, %s)\n",
               rneg, strerror(errno));
        fails++;
    } else {
        printf("  fgds_open(-1) rejected with EINVAL\n");
    }

    return fails;
}

int main(int argc, char **argv) {
    const char *base = (argc > 1) ? argv[1] : NULL;
    int ids[FGDS_TEST_MAX_DEV];
    int n = enumerate_devices(ids, FGDS_TEST_MAX_DEV);

    printf("fgds_multi_device: found %d device(s)", n);
    for (int k = 0; k < n; ++k)
        printf(" %d", ids[k]);
    printf("\n");

    if (n == 0) {
        printf("no /dev/fgds_dev* found; is the fgdsfs module loaded?\n");
        return 1;
    }
    if (n < 2)
        printf("WARNING: only 1 device available; multi-device cannot be fully verified\n");

    g_fail += check_multi_open_register(ids, n, false);
    g_fail += check_multi_open_register(ids, n, true);
    g_fail += check_close_isolation(ids, n, base);
    g_fail += check_interleaved_roundtrip(ids, n, base, FGDS_TEST_IO_DIRECT);
    g_fail += check_interleaved_roundtrip(ids, n, base, FGDS_TEST_IO_MIXED);
    g_fail += check_interleaved_roundtrip(ids, n, base, FGDS_TEST_IO_URING);
    g_fail += check_open_retry(ids, n);
    g_fail += check_idempotency(ids, n);
    g_fail += check_negative(ids, n);

    if (g_fail != 0) {
        printf("fgds_multi_device: %d check(s) FAILED\n", g_fail);
        return 1;
    }
    printf("fgds_multi_device: all checks PASSED\n");
    return 0;
}
