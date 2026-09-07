/*
 * This file was modified by KylinSoft. Co., Ltd. on 2026
 */
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <sys/time.h>
#include <linux/types.h>
#include <pthread.h>
#include <sys/types.h>
#include <unistd.h>
#include <vector>
#include <string>

#include "fgds.h"
#include <liburing.h>

#define FGDS_GPU_PAGE_SIZE (64 * 1024)
#define SMALL_PAGE_SIZE (4 * 1024)
#define FGDS_MMAP_SEGMENT_SIZE (1024 * 1024 * 1024)
#define FGDS_MAX_DEVICES 8
#define QD 128

/* iouring配置策略：
 * 对于fgds读写操作，如果fgds_read/fgds_write的入参nbytes的值小于FGDS_URING_READ_THRESH/FGDS_URING_WRITE_THRESH，则使用单次 pread/pwrite 操作，否则使用 io_uring 流水线。
 * 因为iouring流水线也是有使用成本的，例如iouring环的内存开销、每个子io的fgds_do_xfer_addr计算开销、同步等待cqe完成的开销、子io数量过多时的尾延迟影响等，所以当nbytes较小时，不使用iouring
 */
// 大 IO 的 io_uring 流水线：每个子 IO 大小为 FGDS_SUB_IO_SIZE
// 低于各操作阈值时回退为单次 pread/pwrite，避免小传输上的 io_uring setup/submit 开销。
#define FGDS_SUB_IO_SIZE         (256 * 1024)        // io_uring 批次内的子 IO 块大小
#define FGDS_URING_QD            256                 // 每批最多在途的子 IO 数

#define FGDS_URING_READ_THRESH   (512 * 1024)        // < 512KB, 走 fgds_read_direct，直接pread
#define FGDS_URING_WRITE_THRESH  (2 * 1024 * 1024)   // < 2MB  ,走fgds_write_direct，直接pwrite

typedef struct fgds_mmap_node_s { // 管理一次fgds_regmem对应的一整段显存映射关系，可能拆分成多次mmap
    void **vaddrs; // 保存每次 mmap 返回的地址数组 （因为单次最多映射 FGDS_MMAP_SEGMENT_SIZE=1GB，大区间会拆多次）
    size_t mmap_count; // 拆成了几段 mmap
    uint64_t gpu_addr; // 注册的显存buffer的起始地址
    size_t length; // 注册总长度
    size_t length_left; // 最后一次mmap的长度
    bool is_mapped; // 是否已经完成mmap和ioctl，即是否已经建立用户态内存和显存的映射关系 
    struct fgds_mmap_node_s *next; // 指向下一个节点的指针
} fgds_mmap_node_t;

typedef struct fgds_mmap_buffer_s { // 管理一个GPU在libfgds中的显存映射关系
    int device_id;
    int dev_fd; // 对应 /dev/fgds_devX 的文件描述符, 在fgds_open中获取
    bool initialized; // 该设备是否已成功 open 初始化
    struct fgds_mmap_node_s *head;
    pthread_mutex_t lock;
    struct io_uring uring;        // 每设备一个 ring，大 IO 时惰性初始化
    bool uring_init;              // uring 已完成初始化则为 true
    pthread_mutex_t uring_lock;   // 保护 io_uring ring 的并发访问, TODO: 解决锁粒度太粗的问题
} fgds_mmap_buffer_t;

static int g_device_count = FGDS_MAX_DEVICES;
static fgds_mmap_buffer_t g_dev_ctx[FGDS_MAX_DEVICES];
static std::vector<std::string> fgds_dev_path = {
    "/dev/fgds_dev0", "/dev/fgds_dev1",
    "/dev/fgds_dev2", "/dev/fgds_dev3",
    "/dev/fgds_dev4", "/dev/fgds_dev5",
    "/dev/fgds_dev6", "/dev/fgds_dev7"
};

static std::vector<bool> fgds_initialized(FGDS_MAX_DEVICES, false);

void fgds_free_mmap_nodes(fgds_mmap_buffer_t *buffer) {
    struct fgds_mmap_node_s *current = buffer->head;
    struct fgds_mmap_node_s *next;
    int count = 0;
    pthread_mutex_lock(&buffer->lock);

    while (current) {
        next = current->next;
        // For nodes that are not pinned, manual release is required
        free(current);
        current = next;
        count++;
    }

    buffer->head = NULL;
    pthread_mutex_unlock(&buffer->lock);

}

static int __fgds_close(fgds_mmap_buffer_t *mbuffer) {
    if (!mbuffer->initialized)
        return -1;

    if (mbuffer->uring_init) {
        io_uring_queue_exit(&mbuffer->uring);
        mbuffer->uring_init = false;
    }

    if (mbuffer->dev_fd > 0)
        close(mbuffer->dev_fd);

    fgds_free_mmap_nodes(mbuffer);
    pthread_mutex_destroy(&mbuffer->lock);
    pthread_mutex_destroy(&mbuffer->uring_lock);
    return 0;
}

static int fgds_close_all() {
    for (int i = 0; i < g_device_count; i++) {
        if (fgds_initialized[i]) {
            __fgds_close(&g_dev_ctx[i]);
            fgds_initialized[i] = false;
        }
    }
    return 0;
}

int fgds_close(int device_id) {
    if (device_id >= 0 && device_id < g_device_count) {
        if (fgds_initialized[device_id]) {
            __fgds_close(&g_dev_ctx[device_id]);
            fgds_initialized[device_id] = false;
        }
        return 0;
    }
    return -1;
}

static int __fgds_open(const char *dev_path, fgds_mmap_buffer_t *mbuffer) {
    mbuffer->dev_fd = open(dev_path, O_RDWR);
    
    if (mbuffer->dev_fd == -1) {
        printf("failed to open file %s\n", dev_path);
        return -1;
    }
    mbuffer->head = NULL;
    pthread_mutex_init(&mbuffer->lock, NULL);
    pthread_mutex_init(&mbuffer->uring_lock, NULL);
    mbuffer->initialized = true;
    mbuffer->uring_init = false;
    return 0;
}



bool is_fgds_initialized() {
    bool initialized = false;
    for (int i = 0; i < g_device_count; i++) {
        initialized = initialized | fgds_initialized[i];
    }
    return initialized;
}

int fgds_open(int deviceID) {
    int ret;
    if (!is_fgds_initialized()) { //只会初始化一个GPU，若已经有GPU被初始化，则不进行其他GPU的初始化
        if (deviceID == -1) {
            for (int id = 0; id < g_device_count; ++id) {
                if (!fgds_initialized[id]) {
                    ret = __fgds_open(fgds_dev_path[id].c_str(), &g_dev_ctx[id]);
                    if (ret < 0) {
                        fgds_close_all();
                        return ret;
                    }
                    fgds_initialized[id] = true;
                }
            }
        } else if (deviceID >= 0 && deviceID < g_device_count) {
            if (!fgds_initialized[deviceID]) {
                ret = __fgds_open(fgds_dev_path[deviceID].c_str(), &g_dev_ctx[deviceID]);
                fgds_initialized[deviceID] = true;
                return ret;
            }
        } else { // only start device id = 0
            if (!fgds_initialized[0]) {
                ret = __fgds_open(fgds_dev_path[0].c_str(), &g_dev_ctx[0]);
                fgds_initialized[0] = true;
                return ret;
            }
        }
    }
    return 0;
}


int insert_fgds_mmap_node(fgds_mmap_buffer_t *mbuffer, fgds_mmap_node_t *new_node) {
    if (!new_node) {
        fprintf(stderr, "%s: new_node is NULL\n", __func__);
        return -1;
    }

    pthread_mutex_lock(&mbuffer->lock);
    new_node->next = mbuffer->head;
    mbuffer->head = new_node;
    pthread_mutex_unlock(&mbuffer->lock);
    return 0;
}

fgds_mmap_node_t *find_fgds_mmap_node(fgds_mmap_buffer_t *mbuffer, u64 gpu_addr, u64 len) {
    fgds_mmap_node_t *current = mbuffer->head;
    while (current) {
        if (current->gpu_addr <= gpu_addr && ((current->gpu_addr + current->length) >= (gpu_addr + len))) {
            return current;
        }
        current = current->next;
    }
    return NULL; // 未找到节点
}

int delete_fgds_mmap_node(fgds_mmap_buffer_t *mbuffer, fgds_mmap_node_t *node) {
    fgds_mmap_node_t *current = mbuffer->head;
    fgds_mmap_node_t *previous = NULL;
    pthread_mutex_lock(&mbuffer->lock);
    while (current) {
        if (current->gpu_addr == node->gpu_addr) {
            if (previous) {
                previous->next = current->next;
            } else {
                mbuffer->head = current->next;
            }
            pthread_mutex_unlock(&mbuffer->lock);
            return 0;
        }
        previous = current;
        current = current->next;
    }
    pthread_mutex_unlock(&mbuffer->lock);
    return -1;
}

static inline int __fgds_regmem(fgds_mmap_buffer_t *mbuffer, u64 gpu_addr, u64 host_vaddr, size_t len) {
    fgds_ioctl_para_t para;
    int ret;

    if (!mbuffer->initialized) {
        return -1;
    }

    para.map_param.gpu_addr = (u64)gpu_addr;
    para.map_param.host_vaddr = (u64)host_vaddr;
    para.map_param.host_vaddr_size = len;
    para.map_param.gpu_addr_size = len;
    para.map_param.dev.dev_id = mbuffer->device_id;
    ret = ioctl(mbuffer->dev_fd, FGDS_IOCTL_MAP, &para);

    return ret;
}

// 注册到 fgds_regmem 的缓冲长度都必须 64KB 正整数倍
// 但是文件/传输层没有 FGDS 64KB 下限,只要 O_DIRECT 块对齐(4KB,个别盘 512B).例如注册 64KB buffer 后传输 4KB、用 4KB 文件是可以的
int fgds_regmem(int device_id, const void *gpu_addr, size_t len, void **target_addr) {
    int i, ret = 0;
    unsigned long mmaped_len;
    size_t mmap_len;
    fgds_mmap_buffer_t *dev_mbuffer = &g_dev_ctx[device_id];
    fgds_mmap_node_t *mmap_node = (fgds_mmap_node_t *)malloc(sizeof(fgds_mmap_node_t));
    if (!mmap_node) {
        fprintf(stderr, "%s: new_node is NULL\n", __func__);
        return -1;
    }

    mmap_node->vaddrs = NULL;
    mmap_node->length = len;
    dev_mbuffer->device_id = device_id;

    if (mmap_node->length % FGDS_GPU_PAGE_SIZE != 0) {
        fprintf(stderr, "%s: node->length is not aligned\n", __func__);
        free(mmap_node);
        return -EFAULT;
    }

    mmap_node->mmap_count = mmap_node->length / FGDS_MMAP_SEGMENT_SIZE;
    if (mmap_node->length % FGDS_MMAP_SEGMENT_SIZE) {
        mmap_node->mmap_count++;
        mmap_node->length_left = mmap_node->length % FGDS_MMAP_SEGMENT_SIZE;
    } else {
        mmap_node->length_left = 0;
    }

    mmap_node->vaddrs = (void **)malloc(mmap_node->mmap_count * sizeof(void *));
    if (mmap_node->vaddrs == NULL) {
        fprintf(stderr, "%s: node->vaddrs malloc fail\n", __func__);
        free(mmap_node);
        return -1;
    }

    mmap_node->gpu_addr = (u64)gpu_addr;
    mmaped_len = 0;

    for(i = 0; i < (int)mmap_node->mmap_count; i++) {
        mmap_node->vaddrs[i] = NULL;
        mmap_len = (i == ((int)mmap_node->mmap_count - 1) && mmap_node->length_left) ?
                   mmap_node->length_left : FGDS_MMAP_SEGMENT_SIZE;

        mmap_node->vaddrs[i] = mmap(NULL, mmap_len, PROT_READ|PROT_WRITE, MAP_SHARED, dev_mbuffer->dev_fd, 0);
        if (mmap_node->vaddrs[i] == MAP_FAILED) {
            fprintf(stderr, "%s: node->vaddrs mmap fail, errno=%d\n", __func__, errno);
            // Cleanup: munmap already mapped regions
            for (int j = 0; j < i; j++) {
                size_t cleanup_len = (j == ((int)mmap_node->mmap_count - 1) && mmap_node->length_left) ?
                                   mmap_node->length_left : FGDS_MMAP_SEGMENT_SIZE;
                munmap(mmap_node->vaddrs[j], cleanup_len);
            }
            free(mmap_node->vaddrs);
            free(mmap_node);
            return -EFAULT;
        }

        ret = __fgds_regmem(dev_mbuffer, (u64)mmap_node->gpu_addr + mmaped_len, (u64)mmap_node->vaddrs[i], mmap_len);
        if (ret) {
            fprintf(stderr, "%s: node->vaddrs _fgds_regmem fail\n", __func__);
            // Cleanup: munmap all mapped regions including current
            for (int j = 0; j <= i; j++) {
                size_t cleanup_len = (j == ((int)mmap_node->mmap_count - 1) && mmap_node->length_left) ?
                                   mmap_node->length_left : FGDS_MMAP_SEGMENT_SIZE;
                munmap(mmap_node->vaddrs[j], cleanup_len);
            }
            free(mmap_node->vaddrs);
            free(mmap_node);
            return -EFAULT;
        }
        mmaped_len += mmap_len;
    }
    mmap_node->is_mapped = 1;
    insert_fgds_mmap_node(dev_mbuffer, mmap_node);
    *target_addr = mmap_node->vaddrs[0];

    return 0;
}


int __fgds_deregmem(fgds_mmap_buffer_t *pb, u64 n_addr, u64 c_addr, size_t len) {
    fgds_ioctl_para_t para;
    int ret;

    if(!pb->initialized) {
        return -1;
    }

    para.map_param.gpu_addr = (u64)n_addr;
    para.map_param.host_vaddr = (u64)c_addr;
    para.map_param.host_vaddr_size = len;
    para.map_param.gpu_addr_size = len;
    para.map_param.dev.dev_id = pb->device_id;

    // print_mem(&para, sizeof(fgds_ioctl_para_t));
    ret = ioctl(pb->dev_fd, FGDS_IOCTL_UNMAP, &para);

    return ret;
}

int fgds_deregmem(int device_id, const void *gpu_addr, size_t len) {
    size_t i, ret = 0;
    unsigned long unmapped_len = 0;
    fgds_mmap_node_t *mmap_node;
    fgds_mmap_buffer_t *mmap_buffer = &g_dev_ctx[device_id];

    mmap_node = find_fgds_mmap_node(mmap_buffer, (u64)gpu_addr, len);
    if(mmap_node == NULL) {
        fprintf(stderr, "%s: node is not found fail\n", __func__);
        return -1;
    }

    if(mmap_node->length != len || !mmap_node->is_mapped) {
        fprintf(stderr, "%s: node is not match or has not reg\n", __func__);
        return -1;
    }

    ret = delete_fgds_mmap_node(mmap_buffer, mmap_node);
    if (ret) {
        fprintf(stderr, "%s: delete_fgds_mmap_node fail! not found the map\n", __func__);
        return -1;
    }

    unmapped_len = 0;
    ret = 0; // Initialize ret to 0
    for(i = 0; i < mmap_node->mmap_count; i++) {
        size_t cleanup_len = FGDS_MMAP_SEGMENT_SIZE;
        if(i == (mmap_node->mmap_count - 1) && mmap_node->length_left) {
            cleanup_len = mmap_node->length_left;
        }

        // Try to deregmem first
        int dereg_ret = __fgds_deregmem(mmap_buffer, (u64)mmap_node->gpu_addr + unmapped_len, (u64)mmap_node->vaddrs[i], cleanup_len);
        if(dereg_ret) {
            fprintf(stderr, "%s: node->vaddrs _fgds_deregmem fail at index %zu, continuing cleanup\n", __func__, i);
            ret = -1; // Set error flag but continue cleanup
        }

        unmapped_len += cleanup_len;

        // Always try to munmap, even if deregmem failed, to avoid memory leaks
        int munmap_ret = munmap(mmap_node->vaddrs[i], cleanup_len);
        if(munmap_ret) {
            fprintf(stderr, "%s: node->vaddrs munmap fail at index %zu, error: %s\n", __func__, i, strerror(errno));
            ret = -1; // Set error flag but continue cleanup
        }
    }

    free(mmap_node);
    return ret;
}

int fgds_close(fgds_fileid_t fid) {
    return close(fid.fd);
}

// 原始单次 pread 路径：用于小 IO（nbyte < FGDS_URING_READ_THRESH），
// 同时也是 io_uring 流水线的基础构建块。
static ssize_t fgds_read_direct(fgds_fileid_t fid, void *gpu_buf,
                                off_t buf_offset, ssize_t nbyte, off_t f_offset) {
    struct fgds_xfer_addr *xfer_addr;
    ssize_t nbyte_per_iter;
    ssize_t nbyte_total = 0;
    ssize_t ret, i;
    void *target_addr;
    xfer_addr = fgds_do_xfer_addr(fid.deviceID, gpu_buf, buf_offset, nbyte);

    if (xfer_addr == NULL) {
        fprintf(stderr, "%s: fgds_do_xfer_addr error\n", __func__);
        return -1;
    }

    for (i = 0;i < xfer_addr->nr_xfer_addrs; i++){
        target_addr = xfer_addr->x_addrs[i].target_addr;
        nbyte_per_iter = xfer_addr->x_addrs[i].nbyte;

        ret = pread(fid.fd, target_addr, nbyte_per_iter, f_offset + nbyte_total);
        if(ret != nbyte_per_iter && ret < 0){
            fprintf(stderr, "%s: pread error: ret is %ld, %s\n", __func__, ret, strerror(errno));
            std::free(xfer_addr);
            return -1;
        }
        nbyte_total += ret;
    }
    std::free(xfer_addr);
    return nbyte_total;
}

// 新版 fgds_read：当 IO ≥ FGDS_URING_READ_THRESH（512KB）时，内部按 FGDS_SUB_IO_SIZE
// 切分，并通过 io_uring 分批提交。这样可以尽量填满 NVMe 命令流水线，减少设备空闲，提高吞吐。
//
// IO < 512KB 时回退到原始单次 pread 路径，该尺寸下已足够优。
// 入参语义：从 fid.fd 的 f_offset 读 nbyte 字节，写到 gpu_buf + buf_offset 对应的 GPU 显存。
ssize_t fgds_read(fgds_fileid_t fid, void *gpu_buf, off_t buf_offset,
                  ssize_t nbyte, off_t f_offset) {
    // --- 小 IO：直接 pread，避免 io_uring 开销 ---
    if (nbyte < (ssize_t)FGDS_URING_READ_THRESH)
        return fgds_read_direct(fid, gpu_buf, buf_offset, nbyte, f_offset);

    fgds_mmap_buffer_t *dev_mb = &g_dev_ctx[fid.deviceID];

    // --- 每设备 io_uring ring 惰性初始化（double-checked locking）---
    if (!dev_mb->uring_init) {
        pthread_mutex_lock(&dev_mb->uring_lock);
        if (!dev_mb->uring_init) {
            if (io_uring_queue_init(FGDS_URING_QD, &dev_mb->uring, 0) < 0) {
                fprintf(stderr, "%s: io_uring_queue_init failed\n", __func__);
                pthread_mutex_unlock(&dev_mb->uring_lock);
                return -1;
            }
            dev_mb->uring_init = true;
        }
        pthread_mutex_unlock(&dev_mb->uring_lock);
    }

    size_t  done       = 0; // 已经完成并提交过的批次累计的逻辑字节 （外层 while 跨批推进）
    ssize_t total_read = 0;

    while (done < (size_t)nbyte) {
        pthread_mutex_lock(&dev_mb->uring_lock);

        int    sqe_count   = 0; // 本批已放入的 SQE 个数（受 FGDS_URING_QD 限制）
        size_t batch_bytes = 0; // 当前这一批（一个 io_uring submit 周期）已经规划进 SQ、准备一起提交给内核的逻辑字节数 （内层 while 内累加，每批从 0 开始）
        int    batch_error = 0;

        // ---------- 阶段 1：向 SQ 填充子io请求 ----------
        while (sqe_count < FGDS_URING_QD && done + batch_bytes < (size_t)nbyte) {
            // 计算子io的大小与偏移
            size_t sub_size = FGDS_SUB_IO_SIZE;
            if (done + batch_bytes + sub_size > (size_t)nbyte)
                sub_size = (size_t)nbyte - done - batch_bytes;

            off_t sub_buf_off = buf_offset + done + batch_bytes;
            off_t sub_f_off   = f_offset   + done + batch_bytes;

            struct fgds_xfer_addr *xfer;
            // todo:这里每个子io都会调用一次fgds_do_xfer_addr，且是串行同步等待，后续考虑是否可以优化
            xfer = fgds_do_xfer_addr(fid.deviceID, gpu_buf,
                                     sub_buf_off, sub_size);
            if (!xfer) {
                fprintf(stderr, "%s: fgds_do_xfer_addr error at sub_buf_off=%ld\n",
                        __func__, sub_buf_off);
                batch_error = -1;
                break;
            }

            // 每个 xfer 段对应一个 SQE,1个SQE就是一个子io；若会超过 ring 深度则先提交当前批
            // （例如 sqe_count==15 且 nr_xfer_addrs==2）
            if (sqe_count + xfer->nr_xfer_addrs > FGDS_URING_QD) {
                std::free(xfer);
                break;
            }

            // 每个 xfer 段 → 一个 SQE
            // 注意：在粗锁保护下，上一批 CQE 已全部 reap，且 sqe_count 受 FGDS_URING_QD 限制，
            // SQ 必然有空位，io_uring_get_sqe 不会失败，无需重试逻辑
            for (uint32_t j = 0; j < xfer->nr_xfer_addrs; j++) {
                struct io_uring_sqe *sqe = io_uring_get_sqe(&dev_mb->uring);
                if (!sqe) {
                    fprintf(stderr, "%s: unexpected SQE exhaustion (sqe_count=%d)\n",
                            __func__, sqe_count);
                    std::free(xfer);
                    batch_error = -1;
                    break;
                }
                io_uring_prep_read(sqe, fid.fd,
                                   xfer->x_addrs[j].target_addr,
                                   xfer->x_addrs[j].nbyte,
                                   sub_f_off);
                sub_f_off += (off_t)xfer->x_addrs[j].nbyte;
                sqe_count++;
            }
            if (batch_error) {
                break;
            }
            std::free(xfer);
            batch_bytes += sub_size;
        }

        // ---------- 阶段 2：提交本批 ----------
        int submitted = 0;
        if (!batch_error && sqe_count > 0) {
            submitted = io_uring_submit(&dev_mb->uring);
            if (submitted < 0) {
                fprintf(stderr, "%s: io_uring_submit failed\n", __func__);
                batch_error = -1;
            }
        }

        // ---------- 阶段 3：回收完成事件 ----------
        /* 本 for 循环原因：上面提交了 submitted 个 SQE，内核完成后会产生 submitted 个 CQE
           （一一对应，完成顺序可以乱序）。本函数是 sync 语义：这一批必须全部读完才能继续下一批或返回。
        */
        int cqe_reaped = 0;
        if (!batch_error) {
            for (int k = 0; k < submitted; k++) {
                struct io_uring_cqe *cqe;
                int wait_ret = io_uring_wait_cqe(&dev_mb->uring, &cqe);
                if (wait_ret < 0) {
                    fprintf(stderr, "%s: io_uring_wait_cqe failed\n", __func__);
                    batch_error = -1;
                    break;
                }
                cqe_reaped++;
                if (cqe->res < 0) {
                    fprintf(stderr, "%s: sub-read failed, ret=%d\n",
                            __func__, cqe->res);
                    io_uring_cqe_seen(&dev_mb->uring, cqe);
                    batch_error = -1;
                    break;
                }
                total_read += cqe->res;
                io_uring_cqe_seen(&dev_mb->uring, cqe);
            }
        }

        // ---------- 错误路径清理：reap掉本批所有剩余CQE，防止ring被污染卡死 ----------
        if (batch_error && submitted > 0) {
            // 先非阻塞peek并reap所有已经在CQ中的CQE
            struct io_uring_cqe *cqe;
            while (io_uring_peek_cqe(&dev_mb->uring, &cqe) == 0) {
                io_uring_cqe_seen(&dev_mb->uring, cqe);
                cqe_reaped++;
            }
            // 如果还有未完成的SQE（即CQE数量不足），需要等待并reap剩余的
            // 因为内核已经收到这些SQE，它们最终一定会完成产生CQE，必须全部reap
            while (cqe_reaped < submitted) {
                int wait_ret = io_uring_wait_cqe(&dev_mb->uring, &cqe);
                if (wait_ret < 0) {
                    // wait_cqe失败，无法继续reap，只能放弃（ring可能已损坏）
                    fprintf(stderr, "%s: failed to reap remaining CQEs during cleanup\n", __func__);
                    break;
                }
                io_uring_cqe_seen(&dev_mb->uring, cqe);
                cqe_reaped++;
            }
            pthread_mutex_unlock(&dev_mb->uring_lock);
            return -1;
        }

        if (!batch_error) {
            done += batch_bytes;
        }

        pthread_mutex_unlock(&dev_mb->uring_lock);

        if (batch_error) {
            return -1;
        }
    }

    return total_read; // 返回本次 fgds_read 调用累计成功读到的总字节数
}

// 原始单次 pwrite 路径：用于小 IO（nbyte < FGDS_URING_WRITE_THRESH）。
static ssize_t fgds_write_direct(fgds_fileid_t fid, void *gpu_buf,
                                 off_t buf_offset, ssize_t nbyte, off_t f_offset) {
    struct fgds_xfer_addr *xfer_addr;
    size_t nbyte_per_iter;
    size_t nbyte_total = 0;
    uint32_t i;
    void *target_addr;
    int ret;

    xfer_addr = fgds_do_xfer_addr(fid.deviceID, gpu_buf, buf_offset, nbyte);
    if (xfer_addr == NULL) {
        fprintf(stderr, "%s: fgds_do_xfer_addr error\n", __func__);
        return -1;
    }

    for (i = 0;i < xfer_addr->nr_xfer_addrs; i++){
        target_addr = xfer_addr->x_addrs[i].target_addr;
        nbyte_per_iter = xfer_addr->x_addrs[i].nbyte;
        ret = pwrite(fid.fd, target_addr, nbyte_per_iter, f_offset + nbyte_total);

        if((size_t)ret != nbyte_per_iter){
            fprintf(stderr, "%s: pwrite error: ret is %d, %s\n", __func__, ret, strerror(errno));
            std::free(xfer_addr);
            return -1;
        }
        nbyte_total += nbyte_per_iter;
    }
    std::free(xfer_addr);
    return (ssize_t)nbyte_total;
}

// 新版 fgds_write：当 IO ≥ FGDS_URING_WRITE_THRESH（2MB）时，按 FGDS_SUB_IO_SIZE
// 切分并通过 io_uring 并发提交。这样可以让 NVMe 流水线util更大，性能更好。ring 与 fgds_read 共用——若那边已初始化则直接复用。
// 入参语义：从 gpu_buf + buf_offset 对应的 GPU 显存，写 nbyte 字节到 fid.fd 的 f_offset。
ssize_t fgds_write(fgds_fileid_t fid, void *gpu_buf, off_t buf_offset,
                   ssize_t nbyte, off_t f_offset) {
    // --- 小 IO：直接 pwrite，避免 io_uring 开销 ---
    if (nbyte < (ssize_t)FGDS_URING_WRITE_THRESH)
        return fgds_write_direct(fid, gpu_buf, buf_offset, nbyte, f_offset);

    fgds_mmap_buffer_t *dev_mb = &g_dev_ctx[fid.deviceID];

    // --- 每设备 io_uring ring 惰性初始化（double-checked locking，与 fgds_read 共用）---
    if (!dev_mb->uring_init) {
        pthread_mutex_lock(&dev_mb->uring_lock);
        if (!dev_mb->uring_init) {
            if (io_uring_queue_init(FGDS_URING_QD, &dev_mb->uring, 0) < 0) {
                fprintf(stderr, "%s: io_uring_queue_init failed\n", __func__);
                pthread_mutex_unlock(&dev_mb->uring_lock);
                return -1;
            }
            dev_mb->uring_init = true;
        }
        pthread_mutex_unlock(&dev_mb->uring_lock);
    }

    size_t  done        = 0; // 已经完成并提交过的批次累计的逻辑字节（外层 while 跨批推进）
    ssize_t total_written = 0;

    while (done < (size_t)nbyte) {
        pthread_mutex_lock(&dev_mb->uring_lock);

        int    sqe_count   = 0;
        size_t batch_bytes = 0; // 本批正在组装中已装进 SQ 的逻辑字节（内层 while 内累加，每批从 0 开始）
        int    batch_error = 0;

        // ---------- 阶段 1：向 SQ 填充子io请求 ----------
        while (sqe_count < FGDS_URING_QD && done + batch_bytes < (size_t)nbyte) {
            size_t sub_size = FGDS_SUB_IO_SIZE;
            if (done + batch_bytes + sub_size > (size_t)nbyte)
                sub_size = (size_t)nbyte - done - batch_bytes;

            off_t sub_buf_off = buf_offset + done + batch_bytes;
            off_t sub_f_off   = f_offset   + done + batch_bytes;

            struct fgds_xfer_addr *xfer;
            // 由于 sub_size 很小，fgds_do_xfer_addr 查询的显存地址范围很小，所以 xfer->nr_xfer_addrs 最多为 2，绝大多数情况为 1
            xfer = fgds_do_xfer_addr(fid.deviceID, gpu_buf,
                                     sub_buf_off, sub_size);
            if (!xfer) {
                fprintf(stderr, "%s: fgds_do_xfer_addr error at sub_buf_off=%ld\n",
                        __func__, sub_buf_off);
                batch_error = -1;
                break;
            }

            // 每个 xfer 段对应一个 SQE；若会超过 ring 深度则先提交当前批
            if (sqe_count + xfer->nr_xfer_addrs > FGDS_URING_QD) {
                std::free(xfer);
                break;
            }

            for (uint32_t j = 0; j < xfer->nr_xfer_addrs; j++) {
                // 同 fgds_read：粗锁保护下 SQE 必然可用，无需重试逻辑
                struct io_uring_sqe *sqe = io_uring_get_sqe(&dev_mb->uring);
                if (!sqe) {
                    fprintf(stderr, "%s: unexpected SQE exhaustion (sqe_count=%d)\n",
                            __func__, sqe_count);
                    std::free(xfer);
                    batch_error = -1;
                    break;
                }
                io_uring_prep_write(sqe, fid.fd,
                                    xfer->x_addrs[j].target_addr,
                                    xfer->x_addrs[j].nbyte,
                                    sub_f_off);
                sub_f_off += (off_t)xfer->x_addrs[j].nbyte;
                sqe_count++;
            }
            if (batch_error) {
                break;
            }
            std::free(xfer);
            batch_bytes += sub_size;
        }

        // ---------- 阶段 2：提交本批 ----------
        int submitted = 0;
        if (!batch_error && sqe_count > 0) {
            submitted = io_uring_submit(&dev_mb->uring);
            if (submitted < 0) {
                fprintf(stderr, "%s: io_uring_submit failed\n", __func__);
                batch_error = -1;
            }
        }

        // ---------- 阶段 3：回收完成事件 ----------
        int cqe_reaped = 0;
        if (!batch_error) {
            for (int k = 0; k < submitted; k++) {
                struct io_uring_cqe *cqe;
                int wait_ret = io_uring_wait_cqe(&dev_mb->uring, &cqe);
                if (wait_ret < 0) {
                    fprintf(stderr, "%s: io_uring_wait_cqe failed\n", __func__);
                    batch_error = -1;
                    break;
                }
                cqe_reaped++;
                if (cqe->res < 0) {
                    fprintf(stderr, "%s: sub-write failed, ret=%d\n",
                            __func__, cqe->res);
                    io_uring_cqe_seen(&dev_mb->uring, cqe);
                    batch_error = -1;
                    break;
                }
                total_written += cqe->res;
                io_uring_cqe_seen(&dev_mb->uring, cqe);
            }
        }

        // ---------- 错误路径清理：reap掉本批所有剩余CQE，防止ring被污染卡死 ----------
        if (batch_error && submitted > 0) {
            struct io_uring_cqe *cqe;
            while (io_uring_peek_cqe(&dev_mb->uring, &cqe) == 0) {
                io_uring_cqe_seen(&dev_mb->uring, cqe);
                cqe_reaped++;
            }
            while (cqe_reaped < submitted) {
                int wait_ret = io_uring_wait_cqe(&dev_mb->uring, &cqe);
                if (wait_ret < 0) {
                    fprintf(stderr, "%s: failed to reap remaining CQEs during cleanup\n", __func__);
                    break;
                }
                io_uring_cqe_seen(&dev_mb->uring, cqe);
                cqe_reaped++;
            }
            pthread_mutex_unlock(&dev_mb->uring_lock);
            return -1;
        }

        if (!batch_error) {
            done += batch_bytes;
        }

        pthread_mutex_unlock(&dev_mb->uring_lock);

        if (batch_error) {
            return -1;
        }
    }

    return total_written; // 返回本次 fgds_write调用累计成功写的总字节数
}

// fgds_do_xfer_addr：把一次 GPU<->文件 IO 的显存窗口翻译成一串 host 虚拟地址段表。
//
// 背景（来自 fgds_regmem）：
//   注册长度为 len 的显存时（len 必须 64KB 对齐），因单次 mmap/ioctl 最多映射
//   FGDS_MMAP_SEGMENT_SIZE=1GB，会把整块显存切成 mmap_count = ceil(len / 1GB) 段，
//   每段 ≤1GB，各自一次 mmap 得到独立的 host VA（vaddrs[i]），再 ioctl 绑定
//   [gpu_addr + i*1GB, +本段长度] ↔ vaddrs[i]。各段 host VA 之间不连续。
//   regmem 返回的 target_addr 只是 vaddrs[0]（仅覆盖第 0 段）。
//
// 问题：
//   访问 GPU 偏移 off 时不能直接 target_addr+off——一旦跨过 1GB 段边界就会落到
//   另一段不相关的 host VA（mmap 间不连续），触发 EFAULT。必须用
//   vaddrs[off/1GB] + (off % 1GB)。本函数就是做这个翻译：给定 IO 窗口
//   [buf_offset, buf_offset+nbyte)（相对注册区起点的偏移），返回它跨越的每段的
//   (host target_addr, nbyte)。
//
// 段数（nr_xfer_addrs）的计算：
//   start = buf_offset / 1GB                  // 窗口起点落在第几个段
//   end   = (buf_offset + nbyte - 1) / 1GB    // 窗口终点落在第几个段
//   nr_xfer_addrs = end - start + 1           // 窗口跨越的段数
//   即“窗口跨越了几个 regmem 建的 1GB 段”。注意：1GB 边界是相对注册区起点的偏移
//   整数倍，与 GPU 绝对地址无关；段是否存在只取决于注册长度 len（len≤1GB 则恒 1 段，
//   绝对 GPU 地址是 800MB 还是 0 不影响分段）。
//
//
//   推论：当 nbyte < 1GB 时，窗口比 1GB 间距短，至多跨 1 个边界，故 nr_xfer_addrs ∈ {1,2}：
//     1：窗口完全落在同一段内（len≤1GB 时必为 1）。
//     2：窗口骑跨 1GB 段边界（仅 len>1GB 且窗口跨过相对偏移 1GB 整数倍位置才出现）。
//
//   例子：len=2GB（段0=偏移[0,1GB)、段1=[1GB,2GB)），IO buf_offset=768MB、nbyte=512MB
//   → 窗口 [768MB, 1.28GB) 骑跨偏移 1GB → start=0, end=1, nr_xfer_addrs=2 → 2 次 pread：
//     段0 target=vaddrs[0]+768MB、nbyte=1GB-768MB=256MB；
//     段1 target=vaddrs[1]、nbyte=(768MB+512MB-1)%1GB+1=256MB，合计 512MB。
//
// 每段条目构造（for i 从 start 到 end）：
//   首段 i==start：target = vaddrs[start] + (buf_offset % 1GB)；
//                  nbyte  = (count>1 ? 1GB - (buf_offset % 1GB) : nbyte)   // 到段尾的剩余，单段时即整窗
//   末段 i==end  ：target = vaddrs[end]；
//                  nbyte  = (buf_offset + nbyte - 1) % 1GB + 1            // 末段占的字节数
//   中间段        ：target = vaddrs[i]；nbyte = 1GB                       // 整段
//   末尾校验 nbyte_left==0 且 nr_xfer_addrs==count，保证切片自洽。
//
// 调用方：fgds_read_direct/fgds_write_direct 对每个 x_addrs[] 做一次 pread/pwrite；
//   io_uring 大 IO 路径按 256KB 子 IO 调本函数，每个子 IO 的 nr_xfer_addrs 至多 2。
struct fgds_xfer_addr *fgds_do_xfer_addr(int device_id, const void *gpu_buf, off_t buf_offset, size_t nbyte) {
    struct fgds_xfer_addr *xfer_addr;
    fgds_mmap_buffer_t *_local_mbuffer = &g_dev_ctx[device_id];
    fgds_mmap_node_t *mmap_node;
    uint32_t start, end, count;
    uint64_t offset_in_page;
    uint64_t nbyte_per_iter;
    uint64_t nbyte_left = nbyte;
    uint32_t i;

    mmap_node = find_fgds_mmap_node(_local_mbuffer, (u64)gpu_buf, nbyte);

    if(!mmap_node) {
        fprintf(stderr, "%s: node is not found fail\n", __func__);
        return NULL;
    }

    if(!mmap_node->is_mapped) {
        fprintf(stderr, "%s: node is not match or has not reg\n", __func__);
        return NULL;
    }
    
    if((nbyte + buf_offset) > mmap_node->length) {
        fprintf(stderr, "%s: Read/Write out of range 0, nbyte is %lu, buf_offset is %lu, length is %lu\n",
                __func__, nbyte, buf_offset, mmap_node->length);
        return NULL;
    }

    start = buf_offset / FGDS_MMAP_SEGMENT_SIZE;
    end = (buf_offset + nbyte - 1) / FGDS_MMAP_SEGMENT_SIZE;
    count = end - start + 1;
 
    xfer_addr = (struct fgds_xfer_addr *)std::malloc(sizeof(struct fgds_xfer_addr) + (count - 1) * sizeof(struct xfer_addr));
    if (!xfer_addr) {
        fprintf(stderr, "%s: addr malloc fail\n", __func__);
        return NULL;
    }

    xfer_addr->nr_xfer_addrs = 0;

    if ((start + count) > mmap_node->mmap_count || 
        end > mmap_node->mmap_count || count > mmap_node->mmap_count) {
        fprintf(stderr, "%s: Write out of range 1\n", __func__);
        return NULL;
    }

    for (i = start; i <= end; i++) {
        if(i == start){
            offset_in_page = buf_offset % FGDS_MMAP_SEGMENT_SIZE;
            if(count > 1)
                nbyte_per_iter = FGDS_MMAP_SEGMENT_SIZE - offset_in_page;
            else
                nbyte_per_iter = nbyte_left;
            xfer_addr->x_addrs[xfer_addr->nr_xfer_addrs++] = (struct xfer_addr){
                .target_addr = (void*)((u64)mmap_node->vaddrs[i] + offset_in_page),
                .nbyte = nbyte_per_iter
            };
        }
        else if(i == end){
            xfer_addr->x_addrs[xfer_addr->nr_xfer_addrs++] = (struct xfer_addr){
                .target_addr = mmap_node->vaddrs[i],
                .nbyte = (buf_offset + nbyte - 1) % FGDS_MMAP_SEGMENT_SIZE + 1
            };
            nbyte_per_iter = xfer_addr->x_addrs[xfer_addr->nr_xfer_addrs - 1].nbyte;
        }
        else{
            xfer_addr->x_addrs[xfer_addr->nr_xfer_addrs++] = (struct xfer_addr){
                .target_addr = mmap_node->vaddrs[i],
                .nbyte = FGDS_MMAP_SEGMENT_SIZE
            };
            nbyte_per_iter = xfer_addr->x_addrs[xfer_addr->nr_xfer_addrs - 1].nbyte;
        }
        nbyte_left -= nbyte_per_iter;
    }
    if (nbyte_left != 0 || xfer_addr->nr_xfer_addrs != count) {
        fprintf(stderr, "%s: fgds_write error !\n", __func__);
        return NULL;
    }

    return xfer_addr;
}

