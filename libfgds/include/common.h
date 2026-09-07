#ifndef __FGDS_COMMON_H__
#define __FGDS_COMMON_H__
#include <linux/types.h>
#include <asm/ioctl.h>


#define u64 __u64
#define s64 __s64 
#define u8 __u8
#define u32 __u32 
#define loff_t __u64

#define DEV_MEM_SIZE 1024 * 1024 * 1024 * 2

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
    u64 cpuvaddr;           // cpu vaddr
    loff_t offset;          // file offset
    u64 size;               // Read/Write length
    u64 end_fence_value;    // End fence-value for DMA completion
    s64 ioctl_return;
    int fd;                 // File descriptor
} __attribute__((packed, aligned(8)));
typedef struct fgds_ioctl_io_s fgds_ioctl_io_t;

struct fgds_ioctl_ret_s {
    s64 ret;
    u8 padding[40];
} __attribute__((packed, aligned(8)));
typedef struct fgds_ioctl_ret_s fgds_ioctl_ret_t;

union fgds_ioctl_para_s {
    struct fgds_ioctl_map_s map_param;
    struct fgds_ioctl_io_s  io_para;
    struct fgds_ioctl_ret_s ret;
} __attribute__((packed, aligned(8)));
typedef union fgds_ioctl_para_s fgds_ioctl_para_t;

#define FGDS_IOCTL 0x88 /* 0x4c */
#define FGDS_IOCTL_MAP   _IOW(FGDS_IOCTL, 1, struct fgds_ioctl_map_s)
#define FGDS_IOCTL_UNMAP _IOW(FGDS_IOCTL, 2, struct fgds_ioctl_map_s)

#endif
