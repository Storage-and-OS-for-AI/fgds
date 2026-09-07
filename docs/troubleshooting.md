# Troubleshooting

## Build fails

The FGDS build needs the NVIDIA driver source (`nv-reg.h` / `nv-p2p.h` under `/usr/src/nvidia-*/`) to compile the kernel module. If CMake reports that it cannot find these files:

1. Verify the driver source is installed:

   ```shell
   ls /usr/src/nvidia-*/
   ```

2. If it is missing, install or reinstall the NVIDIA driver with kernel module sources.

## Kernel module fails to load / GPU not bound

1. Loading the FGDS kernel module requires NVIDIA kernel modules to be loaded first. Run `nvidia-smi` to verify.

2. If a GPU cannot be bound, it is usually because its memory is already in use by another process. Run `sudo fuser -v /dev/nvidia*` to see which processes are using the GPUs (the PIDs listed are the processes holding the GPU); for GPU0:

   ```shell
   sudo fuser -v /dev/nvidia0
   ```

   Stop the conflicting processes (e.g. an existing training or inference job).

   Check the kernel log for error messages with `dmesg`. On Ubuntu/Debian, the messages also appear in `/var/log/syslog`; on RHEL/CentOS/Fedora, in `/var/log/messages`. On systemd-based systems, use `journalctl -k` to view kernel logs.

## Kernel module fails to unload

If `scripts/unload_fgds.sh` (which runs `sudo rmmod fgdsfs`) reports `Module fgdsfs is in use`, some process still holds an open file descriptor or mmap mapping on `/dev/fgds_devX`. Exit all processes using FGDS, then unload again.

## Technical Details (optional)

When loading, FGDS maps each GPU's PCIe BAR physical addresses into `ZONE_DEVICE` virtual memory in the kernel. If some physical pages in a GPU's PCIe BAR address range have PAT (memory caching policy) conflicts that cannot be skipped, that GPU cannot be bound. This is normally caused by those addresses already being in use by other processes.
