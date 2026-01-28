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
u32 npu_num;
extern uint64_t gpu_info_table[MAX_GPU_DEVS];

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

static int fgds_devm_memremap(struct fgds_dev *phx_dev) {
	int ret = 1;
	struct dev_pagemap *pgmap;

	phx_dev->p2p_pgmap = devm_kzalloc(&phx_dev->dev->dev,  //分配的是内核内存，这里指针的意思是关联了这个设备，设备卸载时内存会自动释放
									sizeof(struct pci_p2pdma_pagemap), GFP_KERNEL);  //分配的是内核虚拟内存
	if (phx_dev->p2p_pgmap == NULL)
		return -ENOMEM;

	pgmap = &phx_dev->p2p_pgmap->pgmap;

	pgmap->range.start = phx_dev->paddr + 0x600000;
	pgmap->range.end = phx_dev->paddr + phx_dev->size - 1;

	printk("npu->pgmap->res.start is %#llx, end is %#llx\n", pgmap->range.start,
			pgmap->range.end);
	pgmap->nr_range = 1;
	pgmap->type = MEMORY_DEVICE_PCI_P2PDMA;
	phx_dev->pci_mem_va = devm_memremap_pages(&phx_dev->dev->dev, pgmap);  // 为这个设备的物理地址映射虚拟地址
	printk("npu numa is %d\n", phx_dev->dev->dev.numa_node);

	if (IS_ERR_OR_NULL(phx_dev->pci_mem_va)) {
		printk("pci_alloc_p2pmem fail! \n");
		devm_kfree(&phx_dev->dev->dev, phx_dev->p2p_pgmap);
		return -22;
	}

	printk("npu devm_memremap_pages success, addr is %#lx\n",
			(uintptr_t)phx_dev->pci_mem_va);
	phx_dev->remap = 1;
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
	u64 size;
	u16 bus, fn;

	// get the PCIe BAR information of each GPU device
	dev_ctrl->dev_num = dev_num;
	for (i = 0; i < dev_ctrl->dev_num; i++) {
        // get the PCIe BAR information of each GPU device
		bus = (gpu_info_table[i] >> 8) & 0xFF;
		fn = gpu_info_table[i] & 0xFF;
		dev_ctrl->phx_dev[i].dev = pci_get_domain_bus_and_slot(0, bus, fn);
		if (dev_ctrl->phx_dev[i].dev == NULL) {
			printk("npu%u: pci_get_domain_bus_and_slot failed\n", i);
			return -1;
		}
		for (j = 0; j < PCI_NUM_RESOURCES; j++) {
			size = pci_resource_len(dev_ctrl->phx_dev[i].dev, j);
			if (size > dev_ctrl->phx_dev[i].size){
				// get the maximum BAR size for each GPU device, which is the size of the GPU memory
				dev_ctrl->phx_dev[i].paddr = pci_resource_start(dev_ctrl->phx_dev[i].dev, j);
				dev_ctrl->phx_dev[i].size = size;
			}
		}
		dev_ctrl->phx_dev[i].idx = i;
		dev_ctrl->phx_dev[i].remap = 0;
		printk("npu%u: bus is %x, size is %llu, paddr is %#llx\n", i,
			dev_ctrl->phx_dev[i].dev->bus->number, dev_ctrl->phx_dev[i].size,
			dev_ctrl->phx_dev[i].paddr);
		if (dev_ctrl->phx_dev[i].dev->bus->number != 0x83) {//TODOwh
			continue;
		}

		// remap the GPU device's BAR memory to the kernel space
		ret = fgds_devm_memremap(&dev_ctrl->phx_dev[i]);
		if (ret)
			return ret;
	}
	return 0;
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
		printk("fgds_open %s, npu_idx is %d\n", file_name, dev_idx);
		if (dev_idx < 0 || dev_idx >= ctrl.dev_num) {
			ret = -1;
			goto out;
		}
		// save the device metadata in the file structure
		filp->private_data = &ctrl.phx_dev[dev_idx];
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
			return fgds_map_dev_addr(&map_param, map_param.n_vaddr, map_param.n_size,
									map_param.c_vaddr, map_param.c_size);
		}
		// unmap and clean up the device address mapping
		case FGDS_IOCTL_UNMAP: {
			struct fgds_ioctl_map_s map_param;
			if (copy_from_user(&map_param, argp, sizeof(struct fgds_ioctl_map_s)))
				return -EFAULT;
			fgds_map_dev_release(&map_param, map_param.n_vaddr, map_param.n_size,
								map_param.c_vaddr, map_param.c_size);
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
		devm_memunmap_pages(&dev->dev->dev, &dev->p2p_pgmap->pgmap);
		dev->pci_mem_va = NULL;
	}
	if (dev->p2p_pgmap != NULL) {
		devm_kfree(&dev->dev->dev, &dev->p2p_pgmap->pgmap);
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
	//	ida_simple_remove(&phxfs_chr_minor_ida, dev->idx);
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
		//TODOwh
		//dev_id = ida_simple_get(&fgds_chr_minor_ida, 0, MAX_DEV_NUM, GFP_KERNEL);
		if (ctrl->phx_dev[i].dev->bus->number != 0x83) {
			printk("npu%u: bus is %x, skip\n", i, ctrl->phx_dev[i].dev->bus->number);
			continue;
		}
		ret = fgds_cdev_add(&ctrl->phx_dev[i].cdev, &ctrl->phx_dev[i].device,
							&fgds_chr_fops, THIS_MODULE, &ctrl->phx_dev[i]);
		if (ret) {
		kfree_const(ctrl->phx_dev[i].device.kobj.name);
		goto unregister_generic_fgds;
		}
	}
	printk("fgds_cdev_init success!\n");
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
	npu_num = 0;
	for (i = 0; i < MAX_DEV_NUM; i++) {
		if (gpu_info_table[i] != 0) {
			npu_num++;
		} else {
			break;
		}
	}

	printk("devdrv_get_devnum num:%d\n", npu_num);

	if (npu_num <= 0 || npu_num > MAX_DEV_NUM) {
		printk("devdrv_get_devnum error:%u\n", npu_num);
		return -1;
	}
    // obtain the PCIe BAR information of each GPU device via the PCIe bus
    // and remap the GPU device's BAR memory to the kernel space.
	ret = fgds_ctrl_init(&ctrl, npu_num);
	if (ret != 0) {
		printk("npu_ctrl_init error:%d\n", ret);
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
	// delete the character devices created during initialization
	for (i = 0; i < ctrl.dev_num; i++) { //TODOwh
		if (ctrl.phx_dev[i].dev->bus->number != 0x83) {
			printk("npu%u: bus is %x, skip\n", i, ctrl.phx_dev[i].dev->bus->number);
			ctrl.phx_dev[i].dev = NULL;
			continue;
		}
		fgds_cdev_del(&ctrl.phx_dev[i].cdev, &ctrl.phx_dev[i].device, &ctrl.phx_dev[i]);
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
MODULE_DESCRIPTION("NPU/NVIDIA direct storgae");
MODULE_VERSION("0.0.1");