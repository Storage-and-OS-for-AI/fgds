#include <linux/module.h>
#include <linux/pci.h>
#include <linux/pci-p2pdma.h>
#include <linux/device.h>
#include <linux/slab.h>
#include <linux/ctype.h>

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Kai Kuang");
MODULE_DESCRIPTION("PCI P2P DMA Distance Detection Example");

static struct pci_dev *provider_dev = NULL;
static struct pci_dev *client_devs[4] = {NULL, NULL, NULL, NULL};
static int num_clients = 0;

/* 通过 PCI 厂商 ID 和设备 ID 查找设备 */
static struct pci_dev *find_pci_device(u16 vendor_id, u16 device_id, int instance)
{
    struct pci_dev *pdev = NULL;
    int found = 0;

    while ((pdev = pci_get_device(vendor_id, device_id, pdev)) != NULL) {
        if (found == instance)
            return pdev;
        found++;
    }
    return NULL;
}

/* 解析客户端设备参数字符串 - 使用内核原生 strsep */
static int parse_client_devices(const char *param)
{
    char *param_copy;
    char *token;
    char *work_str;
    int count = 0;
    unsigned int vendor, device;
    int ret;

    if (!param)
        return 0;

    /* 复制参数字符串，因为 strsep 会修改原字符串 */
    param_copy = kstrdup(param, GFP_KERNEL);
    if (!param_copy)
        return -ENOMEM;

    work_str = param_copy;
    
    /* 使用 strsep 遍历逗号分隔的 token */
    while ((token = strsep(&work_str, ",")) != NULL && count < 4) {
        char *colon_pos;
        
        /* 跳过空 token（例如连续逗号的情况） */
        if (*token == '\0')
            continue;
        
        /* 查找冒号分隔符 */
        colon_pos = strchr(token, ':');
        if (!colon_pos) {
            pr_warn("Invalid client format: %s (expected vendor:device)\n", token);
            continue;
        }
        
        /* 分割 vendor 和 device 字符串 */
        *colon_pos = '\0';
        
        /* 使用内核提供的安全十六进制转换函数 */
        ret = kstrtouint(token, 16, &vendor);
        if (ret != 0) {
            pr_warn("Invalid vendor hex format: %s\n", token);
            continue;
        }
        
        ret = kstrtouint(colon_pos + 1, 16, &device);
        if (ret != 0) {
            pr_warn("Invalid device hex format: %s\n", colon_pos + 1);
            continue;
        }
        
        /* 查找 PCI 设备 */
        client_devs[count] = find_pci_device((u16)vendor, (u16)device, 0);
        if (client_devs[count]) {
            pr_info("Found client device %d: %04x:%04x\n", 
                    count, vendor, device);
            count++;
        } else {
            pr_warn("Client device %d (%04x:%04x) not found\n", 
                    count, vendor, device);
        }
    }

    kfree(param_copy);
    return count;
}

/* 向模块传递 provider 设备参数 */
static int provider_vendor = 0;
static int provider_device = 0;
module_param(provider_vendor, int, 0444);
MODULE_PARM_DESC(provider_vendor, "Provider PCI Vendor ID (hex, e.g., 0x10de)");

module_param(provider_device, int, 0444);
MODULE_PARM_DESC(provider_device, "Provider PCI Device ID (hex, e.g., 0x1e30)");

/* 向模块传递 client 设备参数，格式：vendor1:device1,vendor2:device2,... */
static char *client_params = NULL;
module_param(client_params, charp, 0444);
MODULE_PARM_DESC(client_params, "Client PCI devices (format: vendor1:device1,vendor2:device2)");

/* 检测并打印 P2P 距离信息 */
static void check_p2p_distance(void)
{
    struct device *client_devs_dev[4];
    int i;
    int distance;
    int valid_client_count = 0;

    if (!provider_dev) {
        pr_err("Provider device not found\n");
        return;
    }

    /* 构建有效的 client device 数组 */
    for (i = 0; i < num_clients; i++) {
        if (client_devs[i]) {
            client_devs_dev[valid_client_count] = &client_devs[i]->dev;
            valid_client_count++;
        }
    }

    if (valid_client_count == 0) {
        pr_err("No valid client devices\n");
        return;
    }

    pr_info("=== PCI P2P DMA Distance Check ===\n");
    pr_info("Provider: %04x:%04x (domain:bus:slot.func = %04x:%02x:%02x.%d)\n",
            provider_dev->vendor, provider_dev->device,
            pci_domain_nr(provider_dev->bus),
            provider_dev->bus->number,
            PCI_SLOT(provider_dev->devfn),
            PCI_FUNC(provider_dev->devfn));

    for (i = 0; i < valid_client_count; i++) {
        struct pci_dev *client = to_pci_dev(client_devs_dev[i]);
        pr_info("Client %d: %04x:%04x (domain:bus:slot.func = %04x:%02x:%02x.%d)\n",
                i, client->vendor, client->device,
                pci_domain_nr(client->bus),
                client->bus->number,
                PCI_SLOT(client->devfn),
                PCI_FUNC(client->devfn));
    }

    /* 核心调用：检测 P2P DMA 距离 */
    distance = pci_p2pdma_distance_many(provider_dev, 
                                         client_devs_dev, 
                                         valid_client_count,
                                         true);  /* verbose=true，不兼容时打印警告 */

    if (distance < 0) {
        pr_info("Result: P2P DMA NOT SUPPORTED (distance = %d)\n", distance);
        pr_info("Reason: Devices are not behind the same root port or\n");
        pr_info("        the host bridge is not in the pci_p2pdma whitelist\n");
    } else {
        pr_info("Result: P2P DMA SUPPORTED (distance = %d)\n", distance);
        if (distance == 0) {
            pr_info("        Provider and client(s) are the same device (optimal)\n");
        } else {
            pr_info("        Lower distance indicates better P2P performance\n");
        }
    }
}

static int __init p2p_distance_init(void)
{
    int ret = 0;

    pr_info("PCI P2P DMA Distance Check Module loaded\n");

    /* 检查并获取 provider 设备 */
    if (provider_vendor == 0 || provider_device == 0) {
        pr_err("Please provide provider_vendor and provider_device parameters\n");
        pr_err("Example: modprobe p2p_distance provider_vendor=0x10de provider_device=0x1e30\n");
        return -EINVAL;
    }

    /* 解析 provider 参数（支持带 0x 前缀的十六进制） */
    provider_dev = find_pci_device((u16)provider_vendor, (u16)provider_device, 0);
    if (!provider_dev) {
        pr_err("Provider device %04x:%04x not found\n", 
               provider_vendor, provider_device);
        return -ENODEV;
    }

    pci_dev_get(provider_dev);
    pr_info("Found provider device: %04x:%04x\n", 
            provider_dev->vendor, provider_dev->device);

    /* 解析客户端设备 */
    if (client_params) {
        num_clients = parse_client_devices(client_params);
        if (num_clients <= 0) {
            pr_warn("No valid client devices found\n");
        }
    } else {
        pr_warn("No client devices specified. Please use client_params parameter.\n");
        pr_warn("Example: client_params=\"0x10de:0x1e30,0x8086:0x1533\"\n");
    }

    /* 执行 P2P 距离检测 */
    if (provider_dev && num_clients > 0) {
        check_p2p_distance();
    } else {
        pr_err("Insufficient devices for P2P detection\n");
        ret = -EINVAL;
        goto err_put_provider;
    }

    return 0;

err_put_provider:
    pci_dev_put(provider_dev);
    return ret;
}

static void __exit p2p_distance_exit(void)
{
    int i;

    pr_info("PCI P2P DMA Distance Check Module unloaded\n");

    if (provider_dev)
        pci_dev_put(provider_dev);

    for (i = 0; i < num_clients; i++) {
        if (client_devs[i])
            pci_dev_put(client_devs[i]);
    }
}

module_init(p2p_distance_init);
module_exit(p2p_distance_exit);
