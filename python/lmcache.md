# FGDS Backend for LMCache

This document describes the FGDS (a GPU direct storage solution) external backend implementation for LMCache, which enables high-performance KV cache persistence using GPU direct storage technology.

## Overview

The `fgds_backend` provides an external storage backend for LMCache that leverages GPU Direct Storage (FGDS) technology via the FGDS library. It implements the `AllocatorBackendInterface` from LMCache, enabling direct data transfer between NVMe SSD and GPU memory without CPU memory copies, help to improve KV cache loading/storage performance.

### Key Features

- Implements LMCache's `StorageBackendInterface` and `AllocatorBackendInterface`
- Provides GPU Direct Storage backend similar to 'GDS' for LMCache
- Zero-copy data path between NVMe SSD and GPU memory
- Built-in thread pool for parallel I/O operations
- Automatic POSIX fallback when FGDS is not available
- Support for safetensors format for KV cache storage

## Architecture

### Core Components

| Component             | Description                                                      |
| --------------------- | ---------------------------------------------------------------- |
| `FgdsBackend`         | Main backend class implementing storage and allocator interfaces |
| `FgdsMemoryAllocator` | Custom GPU memory allocator with FGDS memory registration        |
| Metadata Management   | Handles serialization/deserialization of tensor metadata         |
| I/O Thread Pool       | Parallel I/O execution for batched operations                    |

### Data Flow

```
KV Cache (GPU VRAM)
      │
      ▼
FGDS Memory Registration (fgds_regmem)
      │
      ▼
Direct I/O (O_DIRECT) via FGDS library
      │
      ▼
NVMe SSD (safetensors format with metadata prefix)
```

### Cache Directory Structure

The backend creates a two-level directory hierarchy for efficient cache management:

```
/{fgds_path}/
  ├── {first 2 chars of hash}/
  │   ├── {next 2 chars of hash}/
  │   │   ├── {url_encoded_key}.kvcache.safetensors
  │   │   └── {url_encoded_key}.kvcache.safetensors.metadata
  │   └── ...
  └── ...
```

## Prerequisites

- Linux operating system with dmabuf API support
- NVIDIA or AMD GPU with dmabuf export capability
- FGDS kernel module loaded
- Python ≥ 3.10
- LMCache installed with version >= 0.5.4
- VLLM installed with version >= 0.28.0
- PyTorch with CUDA or ROCm support depending on GPU architecture
- NVMe SSD (recommended for best performance)

## Installation

### 1. Install FGDS Python Package

```bash
cd /path/to/fgds/python
pip install -e .
```

Or install directly:

```bash
pip install .
```

### 2. Verify Installation

```python
# Verify the backend can be imported
from fgds_backend.fgds_backend import FgdsBackend, FgdsMemoryAllocator
print("FGDS backend imported successfully")
```

## Configuration

### LMCache Configuration File

Create a YAML configuration file (e.g., `fgds.yaml`) with the following settings:

```yaml
local_cpu: false
chunk_size: 256
fgds_path: "/data/fgds_cache/"          # Path for cache storage
fgds_buffer_size: 8192                   # Buffer size in MB
max_local_disk_size: 50.0                # Max disk usage in GB
external_backends: "fgds_backend"

extra_config:
  external_backend.fgds_backend.module_path: fgds_backend.fgds_backend
  external_backend.fgds_backend.class_name: FgdsBackend
  
  # Optional settings
  use_fgds: true                         # Enable/disable FGDS (auto-detected if not set)
  disk_io_threads: 4                     # Number of I/O threads (default: 4)
  use_direct_io: true                    # Use O_DIRECT (default: true)
  max_alloc_attempts: 10                 # Max allocation retries (default: 10)
  allocation_attempt_delay_secs: 0.1     # Delay between retries in seconds
```

### Configuration Parameters

| Parameter                       | Type    | Default      | Description                                                  |
| ------------------------------- | ------- | ------------ | ------------------------------------------------------------ |
| `fgds_path`                     | string  | **required** | Directory path for storing KV cache files                    |
| `fgds_buffer_size`              | integer | **required** | Size of FGDS buffer in MB                                    |
| `max_local_disk_size`           | float   | -            | Maximum disk space usage in GB                               |
| `use_fgds`                      | boolean | auto         | Force enable/disable FGDS; auto-detected based on filesystem |
| `disk_io_threads`               | integer | 4            | Number of threads for parallel I/O; set to 0 to disable      |
| `use_direct_io`                 | boolean | true         | Use O\_DIRECT flag for file operations                       |
| `max_alloc_attempts`            | integer | 10           | Maximum number of memory allocation attempts                 |
| `allocation_attempt_delay_secs` | float   | 0.1          | Delay between allocation retries in seconds                  |

## Usage with vLLM

### Step 1: Prepare vLLM Environment

Clone and set up vLLM for benchmarking:

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install -e .
```

### Step 2: Install LMCache

```bash
conda activate vllm-env
pip install lmcache
```

### Step 3: Start vLLM Server with LMCache + FGDS

```bash
cd /path/to/fgds/python/test

LMCACHE_CONFIG_FILE="./fgds.yaml" \
LMCACHE_USE_EXPERIMENTAL=True \
CUDA_VISIBLE_DEVICES=0 \
vllm serve /path/to/your/model/ \
  --max-model-len 8192 \
  --port 8022 \
  --gpu-memory-utilization 0.65 \
  -tp 1 \
  --enforce-eager \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'
```

### Step 4: Run Benchmark Test

Use vLLM's benchmark serving script to test performance:

```bash
cd /path/to/vllm

python benchmarks/benchmark_serving.py \
  --model /path/to/your/model/ \
  --backend vllm \
  --dataset-name sonnet \
  --dataset-path benchmarks/sonnet.txt \
  --host 127.0.0.1 \
  --port 8022 \
  --max-concurrency 64 \
  --num-prompts 128
```

## Implementation Details

### Memory Allocation

The `FgdsMemoryAllocator` extends LMCache's `GPUMemoryAllocator` with:

- Automatic FGDS memory registration (`fgds_regmem`) on allocation and deregistration (`fgds_deregmem`) on deallocation

### I/O Operations

**Write Path:**

1. Pack tensor metadata (4KB reserved space)
2. Write metadata to temporary file
3. Use FGDS to write tensor data directly from GPU memory
4. Atomic rename temporary file to final path

**Read Path:**

1. Look up metadata from in-memory hot cache
2. Allocate GPU memory via FGDS allocator
3. Use FGDS to read data directly into GPU memory
4. Return MemoryObj with loaded tensor

### Filesystem Auto-Detection

The backend automatically detects the filesystem type of `fgds_path`:

- For `tmpfs`/`overlayfs`: Automatically disables FGDS and falls back to POSIX I/O
- For `ext4`/`xfs` (and other GDS-compatible filesystems): Uses FGDS for direct I/O
- This behavior can be overridden by explicitly setting `use_fgds` in extra\_config

### POSIX Fallback Mode

When FGDS is disabled or unavailable, the backend falls back to POSIX I/O:

1. Uses mmap for file access
2. Uses CUDA/HIP runtime memcpy for GPU↔host data transfer
3. Still provides functional KV caching, albeit without GPU Direct Storage acceleration

## File Format

Each KV cache entry is stored as two files:

1. **Data File** (`{key}.kvcache.safetensors`):
   - First 4KB: Reserved for metadata
   - Remainder: Raw tensor data in safetensors-compatible layout
2. **Metadata File** (`{key}.kvcache.safetensors.metadata`):
   - JSON-encoded tensor information
   - Includes: dtype, shape, data offsets, memory format, version info

## Troubleshooting

### FGDS Not Detected

**Symptom:** Log shows "FGDS disabled, using POSIX fallback"

**Possible Causes:**

- Filesystem is tmpfs/overlayfs (auto-disabled)
- FGDS kernel module not loaded
- dmabuf not supported on the system

**Solutions:**

```bash
# Check if FGDS module is loaded
lsmod | grep fgdsfs

# Load FGDS module if needed
bash /path/to/fgds/scripts/load_fgds.sh

# For tmpfs, set use_fgds: true in extra_config to override (not recommended)
```

### Memory Allocation Failures

**Symptom:** "Memory allocation failed" errors in logs

**Solutions:**

- Increase `fgds_buffer_size` in configuration
- Reduce `max-local-disk-size` or other GPU memory usage
- Check available GPU memory with `nvidia-smi` or `rocm-smi`
- Adjust `max_alloc_attempts` and `allocation_attempt_delay_secs`

### Permission Denied Errors

**Symptom:** "Permission Denied" errors when accessing cache files

**Solutions:**

- Ensure the user has read/write permissions for `fgds_path`
- Check SELinux/AppArmor settings if applicable
- Verify the directory exists and is writable:
  ```bash
  mkdir -p /data/fgds_cache
  chmod 755 /data/fgds_cache
  ```

### Performance Issues

**Symptom:** Slow KV cache load/store times

**Solutions:**

- Ensure you're using NVMe SSD (not SATA HDD/SSD)
- Verify FGDS is actually being used (check logs for "Using FGDS")
- Use a filesystem that supports O\_DIRECT (ext4, xfs)
- Adjust `disk_io_threads` based on your SSD's IOPS capability
- Clear system caches before testing:
  ```bash
  sync && echo 3 > /proc/sys/vm/drop_caches
  ```

## Performance Tips

1. **Use NVMe SSD**: FGDS is designed for high-speed NVMe storage
2. **Proper Alignment**: The allocator uses 4KB alignment automatically
3. **Thread Tuning**: Adjust `disk_io_threads` (4-16 for NVMe Gen3/Gen4)
4. **Dedicated Partition**: Use a separate partition for `fgds_path`
5. **Filesystem Choice**: Use ext4 or xfs with `noatime` mount option

## References

| Resource        | Link                                                                 |
| --------------- | -------------------------------------------------------------------- |
| LMCache Project | [github.com/LMCache/LMCache](https://github.com/LMCache/LMCache)     |
| vLLM Project    | [github.com/vllm-project/vllm](https://github.com/vllm-project/vllm) |