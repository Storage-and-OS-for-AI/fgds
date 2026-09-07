/*
 * This file was modified by KylinSoft. Co., Ltd. on 2026
 */
#include <asm/page.h>
#include <linux/cdev.h>
#include <linux/ctype.h> //for isdigit()
#include <linux/device.h>
#include <linux/fcntl.h>
#include <linux/fdtable.h>
#include <linux/fs.h>
#include <linux/init.h>
#include <linux/ioctl.h>
#include <linux/ioport.h>
#include <linux/kernel.h>
#include <linux/kthread.h>
#include <linux/memory.h>
#include <linux/module.h>
#include <linux/nvme_ioctl.h>
#include <linux/pci-p2pdma.h>
#include <linux/pci.h>
#include <linux/pci_ids.h>
#include <linux/printk.h>
#include <linux/sched.h>
#include <linux/seq_buf.h>
#include <linux/thread_info.h>
#include <linux/types.h>
#include <linux/uaccess.h>
#include <linux/xarray.h>

#include "config-host.h"
#include "fgds-mem.h"
#include "fgds.h"

#include "nvfs-p2p.h"
#include "nvfs-pci.h"

static DEFINE_IDA(fgds_chr_minor_ida);
static dev_t fgds_chr_devt;
static struct class *fgds_chr_class;
struct device fgds_chr_dev_device;
struct cdev fgds_chr_dev;

#define FGDS_MINORS 1

struct fgds_ctrl ctrl;

#define NUM_THREADS 128
#define MAX_GPUIDS 16

u32 gpu_count;
extern uint64_t gpu_info_table[MAX_GPU_DEVS];

/* use_all_gpus=1: 使用所有 GPU; =0: 仅使用 gpuids 中指定索引的 GPU，如 gpuids=0,2 */
static int use_all_gpus = 0;
static int gpuids[MAX_GPUIDS] = { 0 };
static int gpuids_count = 1;

module_param(use_all_gpus, int, 0644);
module_param_array(gpuids, int, &gpuids_count, 0644);
MODULE_PARM_DESC(use_all_gpus, "1=use all GPUs, 0=use only GPUs whose index is in gpuids");
MODULE_PARM_DESC(gpuids, "GPU device indices to use when use_all_gpus=0 (e.g. 0,2 for GPU 0 and 2)");

static bool fgds_use_device(int idx)
{
	int i;
	if (use_all_gpus)
		return true;
	for (i = 0; i < gpuids_count; i++)
		if (gpuids[i] == idx)
			return true;
	return false;
}

int extract_trailing_number(const char str[]) {
	int number = 0;
	int multiplier = 1;
	size_t len;
	int found_digit = 0;
	int i;
	len = strlen(str);

	for (i = len - 1; i >= 0; --i) {
		if (isdigit(str[i])) {
			number += (str[i] - '0') * multiplier;
			found_digit = 1;
			if (multiplier == 1) {
				multiplier = 10;
			} else if (found_digit) {
				break;
			}
		} else if (found_digit) {
			break;
		}
	}

	if (found_digit) {
		return number;
	} else {
		return -1;
	}
}

static int fgds_devm_memremap(struct fgds_dev *gpu_dev) {
	int ret = 1;
	struct dev_pagemap *pgmap;

	gpu_dev->p2p_pgmap = devm_kzalloc(&gpu_dev->dev->dev,  //分配的是内核内存，这里指针的意思是关联了这个gpu，设备驱动卸载时内存会自动释放
									sizeof(struct pci_p2pdma_pagemap), GFP_KERNEL);  //分配的是内核虚拟内存
	if (gpu_dev->p2p_pgmap == NULL)
		return -ENOMEM;

	pgmap = &gpu_dev->p2p_pgmap->pgmap;
	pgmap->range.start = gpu_dev->paddr;
	pgmap->range.end = gpu_dev->paddr + gpu_dev->size - 1;
	printk("gpu->pgmap->res.start is %#llx, end is %#llx\n", pgmap->range.start,
			pgmap->range.end);
	pgmap->nr_range = 1;
	// pgmap->type = MEMORY_DEVICE_PCI_P2PDMA;
	// 用MEMORY_DEVICE_PCI_P2PDMA时，跑example和micro.py会异常重启。改成MEMORY_DEVICE_GENERIC能正常跑，经测试验证，对性能的影响理论上不大。
	// todo:理论上应用MEMORY_DEVICE_PCI_P2PDMA,待排查处理。
	pgmap->type = MEMORY_DEVICE_GENERIC;

	// 把GPU的PCIE Bar地址映射到内核的ZONE_DEVICE类型的虚拟内存，让内核可以访问gpu bar地址，并分配page来管理bar地址,达到统一地址管理的效果，例如支持dma和用户空间mmap这块内核虚拟内存。
	gpu_dev->pci_mem_va = devm_memremap_pages(&gpu_dev->dev->dev, pgmap);
	// printk("gpu numa is %d\n", gpu_dev->dev->dev.numa_node);

	if (IS_ERR_OR_NULL(gpu_dev->pci_mem_va)) {
		// devm_memremap_pages容易因为映射的bar地址范围中由于pat冲突导致失败。pat是x86才有的，这种问题在非x86架构下暂不考虑。且arm架构下跳过bar首部地址2MB容易出异常,导致调用fgds时机器异常重启。
		// 查看机器上的pat属性冲突可通过 /sys/kernel/debug/x86/pat_memtype_list ，该文件需要先在机器上挂载debugfs才会有
#if defined(CONFIG_X86) || defined(__x86_64__) || defined(__i386__)
		// 经验规则：devm_memremap_pages的pat冲突，一般是发生在bar的首地址。在x86架构下尝试分别跳过BAR前2、4、6MB进行探测。但不能跳过8MB，否则调用fgds时机器会异常重启。
		printk("devm_memremap_pages fail! there is maybe a conflict in pat, should adjust the address mapping range of BAR.\n");

		const int skip_sizes_mb[] = {2, 4, 6};
		const int num_attempts = sizeof(skip_sizes_mb) / sizeof(skip_sizes_mb[0]);
		int attempt;

		for (attempt = 0; attempt < num_attempts; ++attempt) {
			resource_size_t skip_sz = (resource_size_t)skip_sizes_mb[attempt] * 1024 * 1024;
			resource_size_t new_start = gpu_dev->paddr + skip_sz;
			resource_size_t new_size = gpu_dev->size - skip_sz;
			void __iomem *probe_addr = NULL;

			if (new_size <= 0) {
				printk("fgds: Adjusted BAR size is not valid after skipping %d MB, there is no bar sapce\n", skip_sizes_mb[attempt]);
				break;
			}

			// 用ioremap探测此范围是否可用
			probe_addr = ioremap(new_start, new_size);
			printk("ioremap probe [%dMB skip from bar header], new_start=%#llx, new_size=%#llx\n",
				skip_sizes_mb[attempt], new_start, new_size);
			if (!probe_addr) {
				printk("fgds: ioremap probe after skipping %d MB failed\n", skip_sizes_mb[attempt]);
				iounmap(probe_addr);
				continue;
			} else {
				printk("ioremap [%dMB skip] success, addr is %#lx\n", skip_sizes_mb[attempt], (uintptr_t)probe_addr);
				iounmap(probe_addr);
			}

			//用devm_memremap_pages再次尝试
			pgmap->range.start = new_start;
			pgmap->range.end = new_start + new_size - 1;
			printk("devm_memremap_pages try [%dMB skip from bar header], start=%#llx, end=%#llx\n",
				skip_sizes_mb[attempt], pgmap->range.start, pgmap->range.end);

			gpu_dev->pci_mem_va = devm_memremap_pages(&gpu_dev->dev->dev, pgmap);

			if (IS_ERR_OR_NULL(gpu_dev->pci_mem_va)) {
				printk("fgds: devm_memremap_pages after skipping %d MB still failed!\n", skip_sizes_mb[attempt]);
				continue;
			} else {
				printk("fgds: devm_memremap_pages success on region after skipping %d MB, addr is %#lx\n",
					skip_sizes_mb[attempt], (uintptr_t)gpu_dev->pci_mem_va);
				gpu_dev->remap = 1;
				gpu_dev->paddr = new_start;
				gpu_dev->size = new_size;
				return 0;
			}
		}
		// 所有探测均失败，清理并返回错误
		printk("fgds: devm_memremap_pages failed after all ioremap probes.\n");
		devm_kfree(&gpu_dev->dev->dev, gpu_dev->p2p_pgmap);
		gpu_dev->p2p_pgmap = NULL;
		return -ENOMEM;
#else
		// 非x86架构，不进行探测，直接返回错误
		printk("fgds: devm_memremap_pages failed and not x86 architecture, skipping probe. Not supported.\n");
		devm_kfree(&gpu_dev->dev->dev, gpu_dev->p2p_pgmap);
		gpu_dev->p2p_pgmap = NULL;
		return -ENOMEM;
#endif
	}

	printk("gpu devm_memremap_pages success, addr is %#lx\n",
			(uintptr_t)gpu_dev->pci_mem_va);
	gpu_dev->remap = 1;
	ret = 0;
	return ret;
}

/**
 * @brief Initialize the fgds-fs control structure and remap the GPU device's BAR memory to the kernel space.
 * @param dev_ctrl: Pointer to the fgds-fs control structure.
 * @param dev_num: Number of GPU devices.
 * @return On success, 0 is returned.
 *         On failure, a negative error code is returned.
 */
static int fgds_ctrl_init(struct fgds_ctrl *dev_ctrl, u32 dev_num) {
	int i, j, ret;
	int flag = -1;
	u64 size;
	u16 bus, fn;

	if (!use_all_gpus && gpuids_count == 0) {
		printk("fgds: no GPU devices to be specified for use, please edit the config.json to specify the GPU devices to be used\n");
		return -EINVAL;
	}

	// get the PCIe BAR information of each GPU device
	dev_ctrl->dev_num = dev_num;
	for (i = 0; i < dev_ctrl->dev_num; i++) {
		if (!fgds_use_device(i)) {
			continue;
		}
		
        // get the PCIe BAR information of each GPU device
		bus = (gpu_info_table[i] >> 8) & 0xFF;
		fn = gpu_info_table[i] & 0xFF;
		dev_ctrl->gpu_dev[i].dev = pci_get_domain_bus_and_slot(0, bus, fn);
		// printk("gpu%u: pci_get_domain_bus_and_slot success, bus is %x, fn is %x\n", i, bus, fn);
		if (dev_ctrl->gpu_dev[i].dev == NULL) {
			printk("gpu%u: pci_get_domain_bus_and_slot failed\n", i);
			return -1;
		}
		for (j = 0; j < PCI_STD_NUM_BARS; j++) {
			size = pci_resource_len(dev_ctrl->gpu_dev[i].dev, j);
			// 考虑打日志，输出每个bar区域的size和paddr
			// printk("gpu%u: bar%d size is 0x%llx, paddr is 0x%llx\n", i, j, size, pci_resource_start(dev_ctrl->gpu_dev[i].dev, j));
			if (size > dev_ctrl->gpu_dev[i].size){
				// get the maximum BAR size for each GPU device, which is the size of the GPU memory
				dev_ctrl->gpu_dev[i].paddr = pci_resource_start(dev_ctrl->gpu_dev[i].dev, j);
				dev_ctrl->gpu_dev[i].size = size;
			}
		}
		dev_ctrl->gpu_dev[i].idx = i;
		dev_ctrl->gpu_dev[i].remap = 0;
		printk("gpu%u: bus is %x, size is %llu, paddr is %#llx, try to remap bar memory to kernel space\n", i,
			dev_ctrl->gpu_dev[i].dev->bus->number, dev_ctrl->gpu_dev[i].size,
			dev_ctrl->gpu_dev[i].paddr);
		
		// remap the GPU device's BAR memory to the kernel space
		ret = fgds_devm_memremap(&dev_ctrl->gpu_dev[i]);
		if (ret) {
			/* remap 失败：fgds_devm_memremap 内部已 devm_kfree(p2p_pgmap)；此处仅释放 PCI 引用 */
			printk("gpu%u: fgds_devm_memremap failed, release PCI reference\n", i);
			pci_dev_put(dev_ctrl->gpu_dev[i].dev);
			dev_ctrl->gpu_dev[i].dev = NULL;
		} else {
			// 只要有至少一块GPU remap成功，就返回0，就加载内核模块成功
			flag = 0;
		}
	}
	return flag;
}

/**
 * @file fgds.c
 * @brief fgds-fs character device open operation. It will save the device metadata in the file structure.
 */
static int fgds_open(struct inode *inode, struct file *filp) {
	int ret = 0;
	int dev_idx;
	char *file_name;
	file_name = filp->f_path.dentry->d_iname; 

	if (file_name != NULL) {
		dev_idx = extract_trailing_number(file_name);
		printk("fgds_open %s, gpu_idx is %d\n", file_name, dev_idx);
		if (dev_idx < 0 || dev_idx >= ctrl.dev_num) {
			ret = -1;
			goto out;
		}
		/* 仅允许打开 remap 成功的设备 */
		if (!ctrl.gpu_dev[dev_idx].remap) {
			ret = -ENODEV;
			goto out;
		}
		// save the device metadata in the file structure
		filp->private_data = &ctrl.gpu_dev[dev_idx];
	}
out:
	printk("fgds_open %d\n", ret);
	return ret;
}

static int fgds_release(struct inode *inode, struct file *filp) { return 0; }

/**
 * @file fgds.c
 * @brief fgds-fs character device ioctl operation. It will handle the IOCTL commands for mapping and unmapping device addresses.
 * @param filp: Pointer to the device file structure.
 * @param cmd: The IOCTL command.
 * @param arg: The argument for the IOCTL command.
 * @return On success, 0 is returned.
 *         On failure, a negative error code is returned.
 */
static long fgds_ioctl(struct file *filp, unsigned int cmd,
                        unsigned long arg) {
	void __user *argp = (void *)arg;
	switch (cmd) {
		//  map a device address to a user-space virtual address
		case FGDS_IOCTL_MAP: {
			struct fgds_ioctl_map_s map_param;
			if (copy_from_user(&map_param, argp, sizeof(struct fgds_ioctl_map_s)))
				return -EFAULT;
			return fgds_map_dev_addr(&map_param, map_param.gpu_addr, map_param.gpu_addr_size,
									map_param.host_vaddr, map_param.host_vaddr_size);
		}
		// unmap and clean up the device address mapping
		case FGDS_IOCTL_UNMAP: {
			struct fgds_ioctl_map_s map_param;
			if (copy_from_user(&map_param, argp, sizeof(struct fgds_ioctl_map_s)))
				return -EFAULT;
			fgds_map_dev_release(&map_param, map_param.gpu_addr, map_param.gpu_addr_size,
								map_param.host_vaddr, map_param.host_vaddr_size);
			return 0;
		}
		default:
			return -ENOTTY;
	}
}

static const struct file_operations fgds_chr_fops = {
    .owner = THIS_MODULE,
    .open = fgds_open,
    .release = fgds_release,
    .unlocked_ioctl = fgds_ioctl,
    .mmap = fgds_mmap,
};

/**
 * @brief Delete the fgds-fs character device and unmap the remapped GPU device's BAR memory from the kernel space.
 * @param cdev: Pointer to the character device structure.
 * @param cdev_device: Pointer to the device structure associated with the character device.
 * @param dev: Pointer to the fgds-fs device structure.
 */
void fgds_cdev_del(struct cdev *cdev, struct device *cdev_device,
                    struct fgds_dev *dev) {
	cdev_device_del(cdev, cdev_device);
	// unmap the remapped GPU device's BAR memory from the kernel space
	if (dev->remap) {
		if (dev->p2p_pgmap != NULL) {
			devm_memunmap_pages(&dev->dev->dev, &dev->p2p_pgmap->pgmap);
		}
		dev->pci_mem_va = NULL;
		dev->remap = 0;
	}
	if (dev->p2p_pgmap != NULL) {
		devm_kfree(&dev->dev->dev, &dev->p2p_pgmap->pgmap);
		dev->p2p_pgmap = NULL;
	}
	dev->dev = NULL;
	//ida_simple_remove(&fgds_chr_minor_ida, dev->idx);
}

int fgds_cdev_add(struct cdev *cdev, struct device *cdev_device,
                   const struct file_operations *fops, struct module *owner,
                   struct fgds_dev *dev) {
	int ret;
	//ret = ida_simple_get(&fgds_chr_minor_ida, 0, MAX_DEV_NUM, GFP_KERNEL);
	//if (ret < 0)
	//	return ret;
	//dev->idx = ret;
	ret = dev_set_name(cdev_device, "fgds_dev%d", dev->idx);
	if (ret) {
		//ida_simple_remove(&fgds_chr_minor_ida, dev->idx);
		return ret;
	}
	cdev_device->devt = MKDEV(MAJOR(fgds_chr_devt), dev->idx);
	cdev_device->class = fgds_chr_class;
	device_initialize(cdev_device);
	cdev_init(cdev, fops);
	cdev->owner = owner;
	ret = cdev_device_add(cdev, cdev_device);
	//if (ret)
	//	ida_simple_remove(&fgds_chr_minor_ida, dev->idx);
	return ret;
}

int fgds_cdev_init(struct fgds_ctrl *ctrl) {
	int ret = -ENOMEM;
	int i;
	ret = alloc_chrdev_region(&fgds_chr_devt, 0, ctrl->dev_num,
								"fgds-generic");
	if (ret < 0)
		goto destroy_subsys_class;
#ifdef CLASS_CREATE_HAS_TWO_PARAMS
  	fgds_chr_class = class_create(THIS_MODULE, "fgds-generic");
#else
  	fgds_chr_class = class_create("fgds-generic");
#endif
	if (IS_ERR(fgds_chr_class)) {
		ret = PTR_ERR(fgds_chr_class);
		goto unregister_generic_fgds;
	}
	for (i = 0; i < ctrl->dev_num; i++) {
		if (!fgds_use_device(i)) {
			continue;
		}
		/* 仅对 remap 成功的 GPU 创建字符设备 */
		if (!ctrl->gpu_dev[i].remap) {
			continue;
		}
		ret = fgds_cdev_add(&ctrl->gpu_dev[i].cdev, &ctrl->gpu_dev[i].device,
							&fgds_chr_fops, THIS_MODULE, &ctrl->gpu_dev[i]);
		if (ret) {
		kfree_const(ctrl->gpu_dev[i].device.kobj.name);
		goto unregister_generic_fgds;
		}
		printk("fgds_cdev_init device:%d success!\n", i);
	}
	return 0;

unregister_generic_fgds:
  	unregister_chrdev_region(fgds_chr_devt, ctrl->dev_num);

destroy_subsys_class:
	class_destroy(fgds_chr_class);
	return ret;
}

/** 
 * @file fgds.c
 * @brief fgds-fs kernel module initialization. It will use the memory service provided by the ZONE_DEVICE to remap the GPU device's PCIe BAR memory to the kernel space, and create a character device for each GPU device.
 */
static int __init fgds_init(void) {
	int ret, i;

	// get nvidia_p2p symbols
	if (nvfs_nvidia_p2p_init()) {
		printk("Could not load nvidia_p2p* symbols\n");
		ret = -EOPNOTSUPP;
		return -1;
	}

	// Initialize the GPU information table
	nvfs_fill_gpu2peer_distance_table_once();
	gpu_count = 0;
	for (i = 0; i < MAX_DEV_NUM; i++) {
		if (gpu_info_table[i] != 0) {
			gpu_count++;
		} else {
			break;
		}
	}

	printk("fgds_gpu_count num:%d\n", gpu_count);

	if (gpu_count <= 0 || gpu_count > MAX_DEV_NUM) {
		printk("fgds_gpu_count error:%u\n", gpu_count);
		return -1;
	}
    // obtain the PCIe BAR information of each GPU device via the PCIe bus
    // and remap the GPU device's BAR memory to the kernel space.
	ret = fgds_ctrl_init(&ctrl, gpu_count);
	if (ret != 0) {
		printk("fgds_ctrl_init error:%d\n", ret);
		return -1;
	}

    // create a fgds-fs character device for each GPU device
	ret = fgds_cdev_init(&ctrl);
	if (ret) {
		printk("fgds_init error!\n");
		return -1;
	}

    // initialize the hash table to store the registered GPU memory regions
	fgds_mbuffer_init();
	return 0;
}

/**
 * @file fgds.c
 * @brief fgds-fs kernel module uninitialization. It will delete the character devices created during initialization, and unmap the remapped GPU device's BAR memory from the kernel space.
 */
static void __exit fgds_exit(void) {
	int i;
	/* 仅释放加载时 fgds_devm_memremap 成功的 GPU 的资源；失败的 GPU 不调用 fgds_cdev_del */
	for (i = 0; i < ctrl.dev_num; i++) {
		if (!ctrl.gpu_dev[i].remap) {
			/* remap 失败的 GPU：仅释放 PCI 引用 */
			if (ctrl.gpu_dev[i].dev) {
				pci_dev_put(ctrl.gpu_dev[i].dev);
				ctrl.gpu_dev[i].dev = NULL;
			}
			continue;
		}
		fgds_cdev_del(&ctrl.gpu_dev[i].cdev, &ctrl.gpu_dev[i].device, &ctrl.gpu_dev[i]);
	}

	// delete nvidia_p2p symbols
	nvfs_nvidia_p2p_exit();
	// destroy fgds character device class
	class_destroy(fgds_chr_class);
	// unregister the character device region
	unregister_chrdev_region(fgds_chr_devt, FGDS_MINORS);
	ida_destroy(&fgds_chr_minor_ida);

	printk("fgds_exit, Good bye!");
}

module_init(fgds_init);
module_exit(fgds_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("qiushi <qiushijsxs@outlook.com>");
MODULE_DESCRIPTION("NVIDIA GPU Direct Storage");
MODULE_VERSION("0.0.1");