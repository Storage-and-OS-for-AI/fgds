#ifndef __FGDS_H__
#define __FGDS_H__

#include <linux/types.h>
#include <linux/blk-mq.h>
#include <linux/nvme.h>
#include <linux/memremap.h>
#include <linux/genalloc.h>
#include <linux/cdev.h>
#include <linux/pci.h>

#define MAX_DEV_NUM 16

struct pci_p2pdma {
    struct gen_pool *pool;
    bool p2pmem_published;
    struct xarray map_types;
};

struct pci_p2pdma_pagemap {
    struct dev_pagemap pgmap;
    struct pci_dev provider;
    u64 bus_offset;
};

struct fgds_dev {
    struct pci_dev *dev; /*pci device */
    int domain;
    unsigned int bus;
    unsigned int devfn;
    u64 size; /* GPU PCIe BAR size (largest BAR used for mapping) */
    u64 paddr; /* GPU PCIe BAR start physical address */
    struct device device; /* char device. */
    struct cdev cdev;
    int idx;
    struct pci_p2pdma_pagemap *p2p_pgmap; /* struct dev_pagemap pgmap; */
    void *dev_remap_addr; // 目前暂未用到
    void __iomem *pci_mem_va; /* get from devm_memremap_pages, which is the start virtual address of the GPU Bar in host kernel memory*/
    bool remap;
    unsigned int dev_page_size; // 目前暂未用到
};

struct fgds_ctrl {
    struct fgds_dev gpu_dev[MAX_DEV_NUM];
    int dev_num;
};

struct fgds_dev_info_s {
    u64 dev_id;
} __attribute__((packed, aligned(8)));
typedef struct fgds_dev_info_s fgds_dev_info_t;

struct fgds_ioctl_map_s {
    struct fgds_dev_info_s dev;
    u64 host_vaddr;
    u64 host_vaddr_size;
    u64 gpu_addr;
    u64 gpu_addr_size;
    u64 end_addr;
    u32 shadow_blocks;
} __attribute__((packed, aligned(8)));
typedef struct fgds_ioctl_map_s fgds_ioctl_map_t;

struct fgds_ioctl_io_s {
    u64 cpuvaddr; /* cpu vaddr */
    loff_t offset; /* file offset */
    u64 size; /* Read/Write length */
    u64 end_fence_value; /* End fence value for DMA completion */
    s64 ioctl_return;
    int fd; /* File descriptor */
} __attribute__((packed, aligned(8)));
typedef struct fgds_ioctl_io_s fgds_ioctl_io_t;

struct fgds_ioctl_ret_s {
    s64 ret;
    u8 padding[40];
} __attribute__((packed, aligned(8)));
typedef struct fgds_ioctl_ret_s fgds_ioctl_ret_t;

union fgds_ioctl_para_s {
    struct fgds_ioctl_map_s map_param;
    struct fgds_ioctl_io_s io_para;
    struct fgds_ioctl_ret_s ret;
} __attribute__((packed, aligned(8)));
typedef union fgds_ioctl_para_s fgds_ioctl_para_t;


#define FGDS_IOCTL 0x88 /* 0x4c */
#define FGDS_IOCTL_MAP _IOW(FGDS_IOCTL, 1, struct fgds_ioctl_map_s)
#define FGDS_IOCTL_UNMAP _IOW(FGDS_IOCTL, 2, struct fgds_ioctl_map_s)

void fgds_map_dev_release(fgds_ioctl_map_t *map_param, u64 devaddr, u64 dev_len, u64 cpuvaddr, u64 length);

#endif