Usage of get p2p distance

1. compile the kernel module

# make

2. get the gpu and nvme ssd device info

#　ls /dev/disk/by-path/ -l
pci-0000:10:00.0-nvme-1 -> ../../nvme0n1

# lspci -nv |grep 10:00
10:00.0 0108: 144d:a80d (prog-if 02 [NVM Express])

# nvidia-smi
|   0  NVIDIA H100 PCIe               Off |   00000000:05:00.0 Off |                    0 |
| N/A   49C    P0             90W /  350W |   42916MiB /  81559MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   1  NVIDIA H100 PCIe               Off |   00000000:1E:00.0 Off |                    0 |
| N/A   42C    P0             50W /  350W |       1MiB /  81559MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+

# lspci -nv |grep 05:00.0
05:00.0 0302: 10de:2331 (rev a1)

3. insmod and check the dmesg

# insmod p2p_distance.ko provider_vendor=0x10de provider_device=0x2331 client_params="0x144d:0xa80d"

# dmesg |tail -n 20
[185796.894004] === PCI P2P DMA Distance Check ===
[185796.894006] Provider: 10de:2331 (domain:bus:slot.func = 0000:05:00.0)
[185796.894010] Client 0: 144d:a80d (domain:bus:slot.func = 0000:10:00.0)
[185796.894021] Result: P2P DMA SUPPORTED (distance = 8)
[185796.894023]         Lower distance indicates better P2P performance
