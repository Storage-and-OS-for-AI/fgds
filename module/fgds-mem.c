/*
 * This file was modified by KylinSoft. Co., Ltd. on 2026
 */
#include <linux/kernel.h>
#include <linux/blkdev.h> 
#include <linux/blk_types.h> 
#include <linux/random.h> 
#include <linux/file.h> 
#include <linux/hash.h> 

#include <linux/memory.h> 
#include <linux/hashtable.h>

 
#include "fgds-mem.h"
#include "nvfs-core.h"
#include "nvfs-p2p.h"
#include "config-host.h"

static DEFINE_HASHTABLE(fgds_io_mbuffer_hash, FGDS_MAX_SHADOW_ALLOCS_ORDER);
static spinlock_t lock ____cacheline_aligned; 
atomic_t base_index_cnt = ATOMIC_INIT(0);

/*
 * Common helper to release resources associated with a p2p_vmap.
 * It:
 *   - puts the NVIDIA P2P pages (if still present)
 *   - frees the gpu_region wrapper
 *   - frees map->pages
 *
 */
static void __fgds_release_gpu_memory_core(struct p2p_vmap *map)
{
    if (!map)
        return;

    struct gpu_region *gd = (struct gpu_region *)map->data;
    if (gd != NULL)
    {
        if (gd->pages != NULL)
        {
            nvfs_nvidia_p2p_put_pages(0, 0, map->gpuvaddr, gd->pages);
            gd->pages = NULL;
        }
        kfree(gd);
        map->data = NULL;
    }

    if (map->pages != NULL)
    {
        kfree(map->pages);
        map->pages = NULL;
    }

    if(map!=NULL) {
        kfree(map);
    }
}

void release_gpu_memory(struct p2p_vmap* map)
{
    printk("in fgds-mem.c: release_gpu_memory\n");
    if (!map)
        return;

    /* Release all subordinate resources first */
    __fgds_release_gpu_memory_core(map);
}

static void force_release_gpu_memory(struct p2p_vmap* map)
{
    if (!map)
        return;

    printk("in fgds-mem.c: force_release_gpu_memory, Nvidia driver forcefully reclaimed %lu GPU pages\n", map->dev_page_num);

    /*
     * NVIDIA 驱动通过 free_callback 回调到这里，通知我们这段 P2P
     * 映射需要被强制回收。
     */
    __fgds_release_gpu_memory_core(map);
}

fgds_mmap_buffer_t fgds_lookup_mmap_buffer(u64 cpuvaddr, u64 length) {
    fgds_mmap_buffer_t mbuffer = NULL;
    struct mm_struct *mm = current->mm;
    struct vm_area_struct *vma;

    if (!cpuvaddr) {
        printk("fgds_lookup_mmap_buffer get cpuvaddr error");
        goto out;
    }

    if (cpuvaddr % PAGE_SIZE) {
        printk("fgds_lookup_mmap_buffer cpuvaddr not aligned");
        goto out;
    }

    vma = vma_lookup(mm, cpuvaddr);
    if (vma == NULL)
        goto out;


    mbuffer = (fgds_mmap_buffer_t)vma->vm_private_data;
    if (mbuffer!= NULL) {
        if (mbuffer->c_vaddr!= cpuvaddr || mbuffer->map_len!= length) {
            printk("reg region is not same as mmap region");
            goto out;
        } else {
            return mbuffer;
        }
    } else {
        printk("vma found, ·but mbuffer is none!\n");
        goto out;
    }

out:
    return NULL;
} 

/**
 * @brief Map a device address to a user-space virtual address. It will get the physical pages of the GPU memory region and create a mapping between the GPU memory region and the host memory region.
 * @param mbuffer: Pointer to the mmap buffer structure.
 * @param devaddr: The device address to be mapped.
 * @param dev_len: The length of the device memory region.
 * @return On success, 0 is returned.
 *         On failure, a negative error code is returned.
 */
int fgds_map_dev_addr_inner(fgds_mmap_buffer_t mbuffer, u64 devaddr, u64 dev_len) {
    struct fgds_dev *f_dev = NULL;
    struct gpu_region* gd = NULL;
    struct vm_area_struct *vma;
    u64 *dev_page_addrs = NULL;
    u64 gpu_page_size;
    u64 nr_dev_pages;
    u64 pci_bar_off;
    u64 cpu_vaddr;
    unsigned long host_page_num;
    int ret, i, j;

    vma = mbuffer->vma;
    gpu_page_size = GPU_PAGE_SIZE;
    mbuffer->cpu_pages_per_gpu_page = gpu_page_size / PAGE_SIZE;
    f_dev = mbuffer->dev;
    
    if (f_dev == NULL || f_dev->pci_mem_va == NULL) {
        printk("fgds_map_dev_addr_inner get gpu info error\n");
        ret = -ENOMEM;
        goto out;
    }
    
    // calculate the number of device pages and host pages needed for the mapping
    nr_dev_pages = DIV_ROUND_UP(dev_len, gpu_page_size);

    mbuffer->dev_page_num = nr_dev_pages;
    if (dev_len < GPU_PAGE_SIZE) {
        if (dev_len % PAGE_SIZE != 0){
            ret = -EINVAL;
            goto out;
        }
        mbuffer->host_page_num = DIV_ROUND_UP(dev_len, PAGE_SIZE);
    }else{
        mbuffer->host_page_num = nr_dev_pages * (mbuffer->cpu_pages_per_gpu_page);
    }
    
    dev_page_addrs = kzalloc(nr_dev_pages * sizeof(u64), GFP_KERNEL);
    if (dev_page_addrs == NULL) {
        ret = -ENOMEM;
        goto out;
    }

    mbuffer->ppages = (struct page **) kmalloc(mbuffer->host_page_num * sizeof(struct page *), GFP_KERNEL);
    if (mbuffer->ppages == NULL)
    {
        ret = -ENOMEM;
        goto out;
    }

    mbuffer->map = kmalloc(sizeof(struct p2p_vmap) + (nr_dev_pages - 1) * sizeof(uint64_t), GFP_KERNEL);
    if (mbuffer->map == NULL)
    {
        printk("Failed to allocate mapping descriptor\n");
        ret = -ENOMEM;
        goto out;
    }

    // save the information of the mapping descriptor
    mbuffer->map->page_size = GPU_PAGE_SIZE;
    mbuffer->map->release = release_gpu_memory;
    mbuffer->map->size = dev_len;
    mbuffer->map->gpuvaddr = devaddr;
    mbuffer->map->dev_page_num = mbuffer->dev_page_num;
    mbuffer->map->pages = NULL;
    for (i = 0; i < mbuffer->map->dev_page_num; ++i)
    {
        mbuffer->map->addrs[i] = 0;
    } 
    gd = kmalloc(sizeof(struct gpu_region), GFP_KERNEL);
    if (gd == NULL)
    {
        printk("Failed to allocate mapping descriptor\n");
        ret = -ENOMEM;
        goto out;
    }
    gd->pages = NULL;
    mbuffer->map->data = (struct gpu_region*)gd;

    // get the physical pages of the GPU memory region, you can use replace it with your own function
    ret = nvfs_nvidia_p2p_get_pages(0, 0, mbuffer->map->gpuvaddr, GPU_PAGE_SIZE * mbuffer->map->dev_page_num, &gd->pages, 
        (void (*)(void*)) force_release_gpu_memory, mbuffer->map);   
    
    // save the physical addresses of the GPU memory region
    for(i = 0; i < mbuffer->map->dev_page_num; i++)
    {
        if(gd->pages->pages[i]==NULL)
        {
            printk("mem allocation not success, i is %d!\n",i);
            goto out;
        }
        dev_page_addrs[i] = gd->pages->pages[i]->physical_address;
    }

    mbuffer->dev_page_addrs = dev_page_addrs;
    host_page_num = mbuffer->host_page_num;
    if (IS_ERR_OR_NULL(mbuffer->ppages)) {
        ret = -ENOMEM;
        goto out;
    }
    
    // create the mapping between the GPU memory pages and the host memory pages
    for (i = 0; i < nr_dev_pages; i++) {
        pci_bar_off = dev_page_addrs[i] - mbuffer->dev->paddr;
        cpu_vaddr = (uint64_t)(mbuffer->dev->pci_mem_va + pci_bar_off);

        // Validate pci_bar_off to prevent out-of-bounds access
        if (pci_bar_off < 0 || pci_bar_off > (f_dev->size - GPU_PAGE_SIZE)) {
            printk("Invalid pci_bar_off: 0x%llx, dev_size: 0x%llx\n", pci_bar_off, f_dev->size);
            ret = -EINVAL;
            goto out;
        }

        for (j = 0; j < mbuffer->cpu_pages_per_gpu_page; j++) {
            mbuffer->ppages[i * mbuffer->cpu_pages_per_gpu_page + j] = virt_to_page(cpu_vaddr + j * PAGE_SIZE);
        }
    }

    // establish the mapping between the GPU memory region and the host memory region via vm_insert_pages
    ret = vm_insert_pages(vma, mbuffer->c_vaddr, mbuffer->ppages, &host_page_num);
    if (ret) {
        printk("vm_insert_pages failed, ret=%d, total_pages=%lu\n", ret, host_page_num);
        goto out;
    }
    mbuffer->remap = 1;
    return ret;
    
out:
    if (gd != NULL) {
        kfree(gd);
        gd = NULL;
    }
    if (mbuffer->map != NULL) {
        kfree(mbuffer->map);
        mbuffer->map = NULL;
    }
    if (mbuffer->ppages != NULL) {
        kfree(mbuffer->ppages);
        mbuffer->ppages = NULL;
    }
    if (dev_page_addrs != NULL) {
        kfree(dev_page_addrs);
        dev_page_addrs = NULL;
    }
    return ret;
}

/**
 * @brief Map a device address to a user-space virtual address.
 * @param map_param: Pointer to the mapping parameters structure.
 * @param devaddr: The device address to be mapped.
 * @param dev_len: The length of the device memory region.
 * @param cpuvaddr: The user-space virtual address to map to.
 * @param length: The length of the user-space virtual address region.
 * @return On success, 0 is returned.
 *         On failure, a negative error code is returned.
 */
int fgds_map_dev_addr(fgds_ioctl_map_t *map_param, u64 devaddr, u64 dev_len, u64 cpuvaddr, u64 length) {
    int ret = -EINVAL;
    fgds_mmap_buffer_t mbuffer;
    
    // check and bind the mmap buffer
    mbuffer = fgds_lookup_mmap_buffer(cpuvaddr, length);
    if (mbuffer == NULL || mbuffer->vma == NULL || devaddr <= length) {
        return ret;
    } else {
        ret = 0;
        mbuffer->dev_addr = devaddr;
        mbuffer->dev_len = dev_len;
        // map the GPU virtual address to the virtual address space created by mmap
        ret = fgds_map_dev_addr_inner(mbuffer, devaddr, dev_len);
        printk("fgds_map_dev_addr_inner done, ret: %d\n", ret);
        return ret;
    }
}

void fgds_mbuffer_put(fgds_mmap_buffer_t mbuffer);
/**
 * @brief Release the mapping descriptor and put the mmap buffer.
 * @param map_param: Pointer to the mapping parameters structure.
 * @param devaddr: The device address to be unmapped.
 * @param dev_len: The length of the device memory region.
 * @param cpuvaddr: The user-space virtual address to unmap from.
 * @param length: The length of the user-space virtual address region.
 */
void fgds_map_dev_release(fgds_ioctl_map_t *map_param, u64 devaddr, u64 dev_len, u64 cpuvaddr, u64 length) {
    fgds_mmap_buffer_t mbuffer;
    // query the mmap buffer from the hash table using the user-space virtual address
    mbuffer = fgds_lookup_mmap_buffer(cpuvaddr, length);
    if (mbuffer == NULL) {
        printk("fgds_map_dev_release: mbuffer not found for cpuvaddr=0x%llx, length=0x%llx\n", cpuvaddr, length);
        return;
    }
    // put the mmap buffer and delete the mapping descriptor
    fgds_mbuffer_put(mbuffer);
}

static void fgds_mbuffer_free(fgds_mmap_buffer_t mbuffer) {
    spin_lock(&lock);
    hash_del_rcu(&mbuffer->hash_link);
    spin_unlock(&lock);

    if (mbuffer->remap) {
        release_gpu_memory(mbuffer->map);
        mbuffer->map = NULL;
        mbuffer->remap = 0;
    }

    if (mbuffer->dev_page_addrs != NULL) {
        kfree(mbuffer->dev_page_addrs);
        mbuffer->dev_page_addrs = NULL;
    }
    if (mbuffer->ppages != NULL) {
        kfree(mbuffer->ppages);
        mbuffer->ppages = NULL;
    }

    mbuffer->dev = NULL;
    mbuffer->vma = NULL;
    mbuffer->base_index = 0;
    mbuffer->dev_addr = 0;
    mbuffer->dev_len = 0;
}

void fgds_mbuffer_init(void) {
    spin_lock_init(&lock);
    hash_init(fgds_io_mbuffer_hash);
}


void fgds_mbuffer_get_ref(fgds_mmap_buffer_t mbuffer) {
    atomic_inc(&mbuffer->ref);
}

bool fgds_mbuffer_put_ref(fgds_mmap_buffer_t mbuffer) {
    return atomic_dec_and_test(&mbuffer->ref);
}

static void fgds_mbuffer_put_internal(fgds_mmap_buffer_t mbuffer) {
    if (mbuffer == NULL) return;
    
    if (fgds_mbuffer_put_ref(mbuffer)) {
        fgds_mbuffer_free(mbuffer);
        kfree(mbuffer);
    }
}

void fgds_mbuffer_put(fgds_mmap_buffer_t mbuffer) {
    return fgds_mbuffer_put_internal(mbuffer);
}

void fgds_mbuffer_put_dma(fgds_mmap_buffer_t mbuffer) {
    return fgds_mbuffer_put_internal(mbuffer);
}

// 代码段开始

// 获取未加锁的 mmap buffer
static inline fgds_mmap_buffer_t fgds_mbuffer_get_unlocked(unsigned long base_index) {
    fgds_mmap_buffer_t fgds_mbuffer;
    hash_for_each_possible_rcu(fgds_io_mbuffer_hash, fgds_mbuffer, hash_link, base_index) {
        if (fgds_mbuffer->base_index == base_index) {
            fgds_mbuffer_get_ref(fgds_mbuffer);
            return fgds_mbuffer;
        }
    }
    // printk("base_index %lx not found \n", base_index);
    return NULL;
}

// 获取 mmap buffer
fgds_mmap_buffer_t fgds_mbuffer_get(unsigned long base_index) {
    fgds_mmap_buffer_t fgds_mbuffer;
    rcu_read_lock();
        fgds_mbuffer = fgds_mbuffer_get_unlocked(base_index);
    rcu_read_unlock();
    return fgds_mbuffer;
}

int fgds_setup_mmap_buffer(struct file *filp, struct vm_area_struct *vma) {
    u64 buffer_len;
    int ret = -EINVAL, tries = 10;
    unsigned long base_index;
    struct fgds_dev *dev;
    fgds_mmap_buffer_t fgds_mbuffer, fgds_new_mbuffer;

    buffer_len = vma->vm_end - vma->vm_start;

    dev = (struct fgds_dev *)filp->private_data;
    if (dev == NULL)
        goto error;

    // if the length is smaller than 2M, check for 2M alignment
    if (buffer_len < GPU_PAGE_SIZE && (buffer_len % GPU_PAGE_SIZE)) {
        // printk("mmap size not a multiple of 64k: 0x%llx for size >64k \n", buffer_len);
    }

    fgds_new_mbuffer = (fgds_mmap_buffer_t)kzalloc(sizeof(struct fgds_mmap_buffer), GFP_KERNEL);
    if (!fgds_new_mbuffer) {
        ret = -ENOMEM;
        goto error;
    }

    spin_lock(&lock);
    tries = 10;
    do {
        base_index = FGDS_MIN_BASE_INDEX + atomic_inc_return(&base_index_cnt);
        fgds_new_mbuffer->base_index = base_index;
        atomic_set(&fgds_new_mbuffer->ref, 1);
        hash_add_rcu(fgds_io_mbuffer_hash, &fgds_new_mbuffer->hash_link, base_index);
        fgds_mbuffer = fgds_new_mbuffer;
        fgds_new_mbuffer = NULL;
        break;
        // }
    } while (tries);
    spin_unlock(&lock);

    if (fgds_new_mbuffer != NULL) {
        kfree(fgds_new_mbuffer);
        ret = -ENOMEM;
        goto error;
    }

    if (vma->vm_private_data == NULL) {
        vma->vm_private_data = (void *)fgds_mbuffer;
    } else {
       
        printk("vma->vm_private_data!=NULL\n");
        goto error;
    }

    fgds_mbuffer->vma = vma;
    fgds_mbuffer->dev = dev;
    fgds_mbuffer->dev_id = dev->idx;
    fgds_mbuffer->c_vaddr = vma->vm_start;
    fgds_mbuffer->map_len = buffer_len;
    fgds_mbuffer->remap = 0;

    return 0;

error:
    return ret;
}

/**
 * @file fgds-mem.c
 * @brief fgds-fs character device mmap operation. It will set the vma flags and save the vma into the hash table.
 * @param filp: Pointer to the device file structure.
 * @param vma: Pointer to the virtual memory area structure.
 * @return On success, 0 is returned.
 *         On failure, a negative error code is returned.
 */
int fgds_mmap(struct file *filp, struct vm_area_struct *vma) {
    int ret;
    struct mm_struct *mm = current->mm;

    // set the vma flags for the memory mapping
#ifdef NVFS_VM_FLAGS_NOT_CONSTANT
    vma->vm_flags &= ~VM_PFNMAP; // this region is not a direct physical page frame mapping
    vma->vm_flags &= ~VM_IO;    // not used for device IO memory
    vma->vm_flags |= VM_MIXEDMAP;    // allow both anonymous and page-frame mappings
    vma->vm_flags |= mm->def_flags;
#else
    unsigned long vm_flags;
    vm_flags = ACCESS_PRIVATE(vma, __vm_flags);
    vm_flags &= ~VM_PFNMAP;
    vm_flags &= ~VM_IO;
    vm_flags |= VM_MIXEDMAP;
    vm_flags |= mm->def_flags;
    vm_flags_set(vma, vm_flags);
#endif
    vma->vm_pgoff = 0;
    // set the page protection to non-cached
    vma->vm_page_prot = pgprot_noncached(vma->vm_page_prot);

    if (vma->vm_pgoff == 0) {
         // save the vma into the hash table
        ret = fgds_setup_mmap_buffer(filp, vma);
        return ret;
    }

    return -EINVAL;
}
