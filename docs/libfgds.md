# Libfgds
Libfgds primarily provides interfaces to simplify the interaction between applications and the kernel module (fgds-fs), in order to complete the registration and unregistration of GPU buffers, as well as high-throughput data transfer between storage and GPU memory. This section will focus on the fundamental interfaces provided by fgds-fs.

## 1. Driver management
### fgds_open
```c++
/**
 * @brief Open and initialize the metadata for one device.
 * @param deviceID: index of the target device; must satisfy 0 <= deviceID < FGDS_MAX_DEVICES.
 * @return On success, 0 is returned.
 *         On failure, -1 is returned, and errno is set appropriately to indicate the error
 *         (EINVAL for an out-of-range deviceID, including -1; otherwise errno from
 *         open(2), e.g. ENOENT if the device node does not exist, EACCES if it cannot
 *         be opened).
 */
int fgds_open(int deviceID);
```
This function opens the character device corresponding to the given deviceID, then initializes and stores the necessary metadata, which is required for subsequent GPU buffer registration and unregistration operations.

Multi-device behavior:

- **Each device is opened independently.** `fgds_open()` is idempotent per device: calling it again for an already-opened device simply returns 0. Opening device *A* never prevents opening device *B*.
- **There is no "open all" mode.** `-1` is not a sentinel; it is rejected with `errno = EINVAL` like any other out-of-range id. To use every device the kernel exposes, enumerate `/dev/fgds_devN` yourself (e.g. `glob()`/`access()`) and call `fgds_open(n)` for each node that exists. This keeps a single, unambiguous return code: a "open all" call could not report *which* devices were opened, and would return a misleading success on partial failure. The set of device nodes may be sparse (e.g. `gpuids=[0,2]` exposes only `dev0`/`dev2`).
- **Out-of-range `deviceID`** (negative, including `-1`, or `>=` the library limit) fails with `errno = EINVAL` and is never aliased to device 0.
- A failed open does **not** mark the device as initialized, so it can be retried later.

> **Thread safety:** `fgds_open` / `fgds_close` are not internally synchronized in this version. Callers that share a process across threads must serialize open/close themselves (the PyTorch binding already does this via its per-file lock). Concurrent I/O on *different* already-opened devices is independent.

### fgds_close
```c++
/**
 * @brief Close the metadata for a specific device.
 * @param deviceID: The identifier for the target device to be closed.
 * @return On success, 0 is returned.
 *         On failure, -1 is returned, and errno is set appropriately to indicate the error
 *         (EINVAL for an out-of-range deviceID).
 */
int fgds_close(int deviceID);
```
This operation is the reverse of fgds_open, used to close a previously opened device. It releases all metadata associated with the device and cleans up resources (mmap nodes, the per-device io_uring ring and the character-device fd).

- Closing a device that is not open is a no-op and returns 0.
- `fgds_close` is **not reference counted** in this version: it always tears the device down. The caller must ensure no thread is still issuing `fgds_read` / `fgds_write` / `fgds_regmem` / `fgds_deregmem` on that device when it is closed.

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
This function is used to register a GPU memory region for a specified device. The GPU buffer already resides in the device's address space (e.g. allocated via cudaMalloc); fgds_regmem additionally maps it into the host user-space address space (via mmap on the device character device) and returns the resulting host virtual address through target_addr, so the host can use the GPU memory directly as an IO buffer.


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
This function is used to unregister a previously registered GPU memory region from a specified device. It removes the corresponding mapping of the GPU memory from the host user-space address space. It performs the reverse operation of `fgds_regmem`, first removing the registration information from the kernel module using IOCTL, and then unmapping the user space mapping relationship using `munmap`.

## 3. Data transfer

After registering a memory region, applications can use `fgds_read`/`fgds_write` to move data directly between the storage device and GPU memory, bypassing the CPU copy that traditional file IO requires. Both interfaces accept the original GPU buffer pointer passed to `fgds_regmem` and perform the transfer internally in a pipelined manner.

For large IO, the two interfaces transparently split the transfer into fixed-size sub-IO blocks and submit them through **io_uring**, which significantly boosts throughput and reduces latency compared to issuing a single synchronous `pread`/`pwrite`. 

```c++
/**
 * @brief Read nbyte bytes from file offset f_offset of the disk file identified by fid,
 *        and write them into the registered GPU buffer at offset buf_offset.
 * @param fid: The file identifier holding the fd and the deviceID of the opened GPU.
 *             deviceID must have been initialized by fgds_open.
 * @param buf: Pointer to the original GPU buffer passed to fgds_regmem, i.e. the device
 *             address that was registered. Note that this is NOT the host target_addr
 *             returned by fgds_regmem (that one is only needed for raw pread/pwrite).
 * @param buf_offset: Byte offset within buf (relative to the registered region) where data is stored.
 * @param nbyte: Number of bytes to read.
 * @param f_offset: Byte offset within the disk file where the read starts.
 * @return On success, the total number of bytes read is returned.
 *         On failure, -1 is returned, and errno is set appropriately to indicate the error.
 */
ssize_t fgds_read(fgds_fileid_t fid, void *buf, off_t buf_offset, ssize_t nbyte, off_t f_offset);
```

```c++
/**
 * @brief Write nbyte bytes from the registered GPU buffer at offset buf_offset,
 *        to file offset f_offset of the disk file identified by fid.
 * @param fid: The file identifier holding the fd and the deviceID of the opened GPU.
 *             deviceID must have been initialized by fgds_open.
 * @param buf: Pointer to the original GPU buffer passed to fgds_regmem, i.e. the device
 *             address that was registered. Note that this is NOT the host target_addr
 *             returned by fgds_regmem (that one is only needed for raw pread/pwrite).
 * @param buf_offset: Byte offset within buf (relative to the registered region) where data originates.
 * @param nbyte: Number of bytes to write.
 * @param f_offset: Byte offset within the disk file where the write starts.
 * @return On success, the total number of bytes written is returned.
 *         On failure, -1 is returned, and errno is set appropriately to indicate the error.
 */
ssize_t fgds_write(fgds_fileid_t fid, void *buf, off_t buf_offset, ssize_t nbyte, off_t f_offset);
```

These two functions perform direct GPU↔disk data movement. Internally they translate the given GPU buffer address into the host-mapped addresses established by `fgds_regmem`, and then read from or write to the disk file. When the requested `nbyte` is large, the transfer is sliced into sub-IO blocks that are submitted to the device via an io_uring ring shared per GPU device (initialized lazily on first use). This batching keeps a high number of IO requests in flight at once, filling the NVMe command pipeline to improve overall read/write throughput.

## 4. Example

A complete usage example is provided in [example/example.cc](../example/example.cc), demonstrating the full lifecycle: `fgds_open` → `cudaMalloc` → `fgds_regmem` → `pread`/`pwrite` → `fgds_deregmem` → `fgds_close`.