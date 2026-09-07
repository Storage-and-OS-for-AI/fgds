#ifndef __FGDS_MEM_H__
#define __FGDS_MEM_H__

#include <linux/types.h>
#include <linux/mm.h>
#include <linux/file.h>
#include "fgds.h"

#define PAGE_SHIFT 12
#define FGDS_MIN_BASE_INDEX ((unsigned long)1L<<32)
#define FGDS_MAX_SHADOW_PAGES 4096
#define FGDS_MAX_SHADOW_ALLOCS_ORDER 12

struct fgds_mmap_buffer { // 一次 mmap 映射区的内核侧「账本」，由 fgds_mmap 创建，由 ioctl MAP 时填充
    atomic_t ref; // 引用计数，为 0 时释放
    struct hlist_node hash_link; // 挂入全局哈希表 fgds_io_mbuffer_hash
    u64 c_vaddr; // 用户态 mmap 得到的 CPU 虚拟地址
    u64 map_len; // 用户态 mmap 得到的映射长度
    u64 dev_addr; // ioctl MAP 时绑定的 GPU 显存地址
    u64 dev_len; // ioctl MAP 时绑定的 GPU 显存长度
    unsigned long base_index; // 在 hash table 中的索引
    unsigned long dev_id; // gpu id
    u64 *dev_page_addrs; // GPU 显存页的物理地址，调 nvidia 内核模块接口获取
    unsigned long dev_page_num; // GPU 显存页数
    unsigned long cpu_pages_per_gpu_page; // (dev_page_size / PAGE_SIZE) 一个 GPU 显存页对应多少个 CPU 页。目前是 64KB GPU 页 = 16 个 4KB CPU 页
    struct page **ppages; // 要插入进用户 VMA 的 CPU 内存页数组（映射到 GPU 显存页）
    unsigned long host_page_num; // 对应的主机页数，等于 dev_page_num
    struct vm_area_struct *vma; // 反向指向这块 mmap 的 VMA
    struct fgds_dev *dev;
    bool remap; // 是否已完成页插入映射（成功 MAP 后置 true；由 remap_pfn_range 设置）
    struct p2p_vmap* map;
};
typedef struct fgds_mmap_buffer* fgds_mmap_buffer_t;


typedef vm_fault_t fgds_vma_fault_t;


int fgds_map_dev_addr_inner(fgds_mmap_buffer_t pbuffer, u64 devaddr, u64 dev_len);
int fgds_map_dev_addr(fgds_ioctl_map_t *map_param, u64 devaddr, u64 dev_len, u64 cpuvaddr, u64 length);
int fgds_mmap(struct file *filp, struct vm_area_struct *vma);
void fgds_mbuffer_init(void);
void fgds_mbuffer_put(fgds_mmap_buffer_t pbuffer);
fgds_mmap_buffer_t fgds_mbuffer_get(unsigned long base_index);

#endif