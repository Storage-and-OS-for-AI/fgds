# Micro Benchmark

A Python script (`scripts/micro.py`) that runs single-stream read/write tests on a specified NVMe file, comparing FGDS, GDS, and POSIX methods (the traditional approach where GPU reads/writes to NVMe go through CPU memory copies).

### Usage

```shell
python3 scripts/micro.py <gpu_id> <method> <mode> <device> <file_path> [length_gb]
```

| Argument | Meaning |
|----------|---------|
| `<gpu_id>` | GPU device ID (e.g. `0`) |
| `<method>` | `fgds` / `gds` / `posix` |
| `<mode>` | `sync` / `async` / `batch` |
| `<device>` | `nvme` / `nvmeof` |
| `<file_path>` | Test file path (at least 10 GB) |
| `[length_gb]` | Optional, total I/O length in GB (default 10) |

1. Edit `scripts/micro.py` and set the block sizes to test in the `io_sizes` array (unit: KB).
2. Install `numactl` first (the script uses it to pin I/O to a NUMA node): `sudo apt install numactl` (Ubuntu/Debian) or `sudo dnf install numactl` (RHEL/CentOS/Fedora).
3. Running the script with no arguments prints the full usage.

### Test File

To measure performance as objectively and sufficiently as possible, the test file must be at least 10 GB (the script rejects smaller files). Create one on the NVMe drive with random data — assuming the drive is mounted at `/data` and the file is named `10GB_ddrand`:

```shell
dd if=/dev/urandom of=/data/10GB_ddrand bs=1M count=10240 status=progress
```

### Examples

```shell
python3 scripts/micro.py 0 fgds sync nvme /data/10GB_ddrand
```

Run FGDS on GPU0 using the NVMe file `/data/10GB_ddrand` as the test file.

```shell
python3 scripts/micro.py 0 posix sync nvme /data/10GB_ddrand
```

Run the traditional approach (via CPU memory copy) on GPU0 using the same test file.