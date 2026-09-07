# Libfgds
Libfgds primarily provides interfaces to simplify the interaction between applications and the kernel module (fgds-fs), in order to complete the registration and unregistration of GPU buffers. This section will focus on the four fundamental interfaces provided by fgds-fs.

## 1. Driver management
### fgds_open
```c++
/**
 * @brief Open and initialize the metadata for a specific device.
 * @param deviceID: The identifier for the target device to be opened and initialized.
 * @return On success, 0 is returned.
 *         On failure, -1 is returned, and errno is set appropriately to indicate the error.
 */
int fgds_open(int deviceID);
```
This function opens the character device corresponding to the given deviceID, then initializes and stores the necessary metadata, which is required for subsequent GPU buffer registration and unregistration operations.

### fgds_close
```c++
/**
 * @brief Close the metadata for a specific device.
 * @param deviceID: The identifier for the target device to be closed.
 * @return On success, 0 is returned.
 *         On failure, -1 is returned, and errno is set appropriately to indicate the error.
 */
int fgds_close(int deviceID);
```
This operation is the reverse of fgds_open, used to close a previously opened device. It releases all metadata associated with the device and cleans up resources.

## 2. Buffer management
### fgds_regmem

```c++
/**
 * @brief Register a memory region for a specific device.
 * @param device_id: The identifier for the target device.
 * @param addr: Pointer to the memory region to be registered.
 * @param len: Length of the memory region in bytes.
 * @param target_addr: Pointer to the host-remapped address of the registered memory region.
 * @return On success, 0 is returned and target_addr is set to the registered address.
 *         On failure, -1 is returned , and errno is set appropriately to indicate the error.
 */
int fgds_regmem(int device_id, const void *addr, size_t len, void **target_addr);
```
This function is used to register a memory region to a specified device. It maps the memory region into the device's address space and returns the mapped address.

Here, we provide a pseudocode example to illustrate the core implementation process of `fgds_regmem`:
```c++
int fgds_regmem(int device_id, const void *addr, size_t len, void **target_addr) {
    // check the metadata and ensure the parameters are valid
    ...
    // get the device's metadata
    struct fgds_bdev *pb = get_fgds_bdev(device_id);
    if (pb == NULL) {
        fprintf(stderr, "%s: get_fgds_bdev fail\n", __func__);
        return -ENODEV;
    }

    // allocate a new P2P map structure
    struct fgds_p2p_map *p2p_map = (struct fgds_p2p_map *)malloc(sizeof(struct fgds_p2p_map));

    // set the mapping information
    p2p_map->n_addr = addr;
    p2p_map->length = len;
    // allocate virtual address space for the P2P map
    // the dev_fd is the file descriptor of the character device
    p2p_map->vaddrs = mmap(p2p_map->vaddrs,
                            p2p_map->length,
                            PROT_READ|PROT_WRITE,
                            MAP_SHARED,
                            pb->dev_fd,
                            0);
    // Check if the mmap operation was successful
    if ((u64)p2p_map->vaddrs == 0xffffffffffffffff) {
        fprintf(stderr, "%s: p2p_map->vaddrs mmap fail\n", __func__);
        return -EFAULT;
    }

    // Register the memory region with the kernel module using IOCTL
    int ret = __fgds_regmem(pb, (u64)p2p_map->n_addr, (u64)p2p_map->vaddrs, p2p_map->length);

    if (ret < 0) {
        fprintf(stderr, "%s: __fgds_regmem fail\n", __func__);
        munmap(p2p_map->vaddrs, p2p_map->length);
        return ret;
    }
    // Set the target address to the mapped address
    *target_addr = p2p_map->vaddrs;
    // insert the P2P map information into the global list, which is used to manage all P2P maps
    insert_fgds_mmap_node(pb, p2p_map);
}
```
### fgds_deregmem

```c++
/**
 * @brief Unregister a memory region for a specific device.
 * @param device_id: The identifier for the target device.
 * @param addr: Pointer to the memory region to be unregistered.
 * @param len: Length of the memory region in bytes.
 * @return On success, 0 is returned.
 *         On failure, -1 is returned, and errno is set appropriately to indicate the error.
 */
int fgds_deregmem(int device_id, const void *addr, size_t len);
```
This function is used to unregister a previously registered memory region from a specified device. It removes the mapping of the memory region from the device's address space. It performs the reverse operation of `fgds_regmem`, first removing the registration information from the kernel module using IOCTL, and then unmapping the user space mapping relationship using `munmap`.

## 3. Example

A complete usage example is provided in [example/example.cc](../example/example.cc), demonstrating the full lifecycle: `fgds_open` → `cudaMalloc` → `fgds_regmem` → `pread`/`pwrite` → `fgds_deregmem` → `fgds_close`.