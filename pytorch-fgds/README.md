# pytorch-fgds

This directory contains `fgds_torch.py`, an FGDS checkpoint example, and benchmark scripts for performance comparison.

## Integrating the FGDS API into PyTorch

`torch.cuda.fgds` only requires a loadable `libfgds.so`. `fgds_torch.py` declares the
required entry points with `ctypes` itself and loads the library lazily, so the
separate `fgds` Python package (`fgds.fgds_bind`) does **not** need to be installed.

1. Build and install `libfgds.so` as described in the repo top-level docs, and make sure the dynamic linker can find it (for example, install it into a system library path, or set `LD_LIBRARY_PATH`). `libfgds.so` also needs `liburing` at runtime.
2. Verify that the library can be loaded:

```bash
python3 -c "import ctypes; ctypes.CDLL('libfgds.so'); print('libfgds.so loaded')"
```

Then integrate this directory's `fgds_torch.py` into PyTorch:

1. Copy `fgds_torch.py` from this directory into the PyTorch source tree and **rename** it to `fgds.py`:
     - `torch/cuda/fgds.py`
2. Edit `torch/cuda/__init__.py` in the PyTorch source tree and add the import:
     - `from . import fgds`

After that, you can use the FGDS API from Python via `from torch.cuda import fgds`.
`fgds.is_available()` reports whether `libfgds.so` can be loaded and exports the
expected symbols.

To use the module without modifying a PyTorch source tree, add this directory to
`sys.path` and `import fgds_torch` directly (the checkpoint demo does this as a
fallback).

## Design of `fgds_torch.py`

The Python API in `fgds_torch.py` follows the design of `torch/cuda/gds.py`:

- `gds.py` exports `gds_register_buffer`, `gds_deregister_buffer`, and `GdsFile`
- `fgds_torch.py` exports the corresponding `fgds_register_buffer`, `fgds_deregister_buffer`, and `FgdsFile`
- `GdsFile` provides `load_storage` / `save_storage`
- `FgdsFile` also provides `load_storage` / `save_storage`

The goal is to keep the usage of `torch.cuda.fgds` consistent with `torch.cuda.gds`, which makes migration and comparative testing easier.

Main functions/methods in `fgds_torch.py`:

- `_build_bindings`
    - Creates the `ctypes` bindings for `libfgds.so` (`fgds_open`, `fgds_close`, `fgds_read`, `fgds_write`, `fgds_regmem`, `fgds_deregmem`) and their argument/return types.
    - Uses `RTLD_LAZY | RTLD_LOCAL` and does not run at import time, so importing the module succeeds even when `libfgds.so` (or its CUDA/`liburing` dependencies) is unavailable.

- `_load_fgds_bind`
    - Loads and caches the `ctypes` bindings on first use; on failure raises `RuntimeError` with the loader error, and the result is cached for the process lifetime.

- `is_available`
    - Returns whether `libfgds.so` is loadable and exports the expected entry points.

- `_fgds_file_constructed` / `_fgds_file_destroyed`
    - Tracks `FgdsFile` instance counts per `device_id`.
    - Calls `fgds_open(device_id)` when the first instance is created, and `fgds_close(device_id)` when the last instance is destroyed.
    - When the last instance for a device is destroyed, releases any buffers still registered for that device before calling `fgds_close`.

- `_fgds_shutdown`
    - `atexit` fallback that closes sessions still open at interpreter shutdown while `libfgds.so` is still loaded.

- `_fgds_file_session_active`
    - Checks whether the given device already has an active FGDS session (used as a precondition check before registering/deregistering a buffer).

- `_device_lock`
    - Provides a per-device reentrant lock so session teardown cannot race I/O or buffer registration on the same device; operations on different devices are independent.

- `_cuda_device_index`
    - Resolves the CUDA device index from a `Storage` and verifies that the storage is on CUDA.

- `fgds_register_buffer`
    - Calls `fgds_regmem` to register GPU storage (requires an `FgdsFile` to have been created on the same device first, and a 64 KiB-aligned storage whose length is a positive multiple of 64 KiB).

- `fgds_deregister_buffer`
    - Calls `fgds_deregmem` to deregister GPU storage.

- `FgdsFile.__init__`
    - Opens the file (`os.O_DIRECT`), holds an FGDS session, and creates a `fgds_fileid_t` containing the fd and `device_id`.

- `FgdsFile.close` / `FgdsFile.__del__`
    - `close` releases the file fd and this instance's session reference (idempotent); `__del__` is a non-raising fallback.

- `FgdsFile.load_storage`
    - Calls `fgds_read` to read file contents into CUDA storage.

- `FgdsFile.save_storage`
    - Calls `fgds_write` to write CUDA storage contents to the file.

Sessions are per device: multiple `FgdsFile` objects for the same `device_id` share one
`fgds_open`/`fgds_close` session, while objects for different devices coexist
independently, so one process can drive several GPUs.

## Checkpoint Read/Write Demo

`example/checkpoint_demo.py` is a minimal end-to-end example that:

1. builds a synthetic `state_dict`;
2. serializes it with `torch.save` into standard PyTorch checkpoint bytes;
3. writes those bytes to disk with `FgdsFile.save_storage`;
4. reads them back with `FgdsFile.load_storage`;
5. validates the result byte-for-byte, with sha256, and with native `torch.load`.

Requirements:

- Linux with a loadable `libfgds.so` and the `fgdsfs` kernel module;
- a filesystem that supports `O_DIRECT` (not tmpfs);
- Python >= 3.8 (`fgds_torch.py` uses `from __future__ import annotations`);
- numpy (used when copying read-back bytes from GPU to host).

Run from the repository root:

```bash
python3 pytorch-fgds/example/checkpoint_demo.py \
    --path /data/fgds_checkpoint_demo.pt \
    --size-mb 256 \
    --dtype float16 \
    --device-id 0 \
    --chunk-mb 64
```

The demo uses `torch.save` bytes and therefore serializes on the CPU side. It
proves that the FGDS API can read and write a real PyTorch checkpoint, but it is
not the pure GPU-direct performance path. For direct GPU tensor storage I/O, see
`benchmark/bench_fgds_pytorch.py`.

For the multi-chunk and tail-read path, run the demo again with
`--chunk-mb 1`.

## Benchmark Directory

The `benchmark/` directory compares performance across different I/O paths (default test file size: 10GB):

- `bench_fgds_pytorch.py`
    - Benchmarks file write/read performance of `torch.cuda.fgds`.
    - Each round runs one write + one read, and prints per-round and average latency/bandwidth.

- `bench_gds_pytorch.py`
    - Benchmarks write/read performance of `torch.cuda.gds` (cuFile path).
    - Each round runs one write + one read, and prints per-round and average latency/bandwidth.

- `bench_posix_pytorch.py`
    - Benchmarks two ordinary file paths:
      - `torch.save` / `torch.load` (page-cache path)
      - `direct-io` (`O_DIRECT` raw I/O path)
    - Prints per-round and average latency/bandwidth, and performs read-back consistency checks.

- `common.py`
    - Shared configuration and helpers (for example test file path, test size, argument parsing, and chunked consistency checks).

- `run_all_io_tests.sh`
    - Runs the above benchmark scripts sequentially, with a `sleep 3` between scripts.

## Usage

Before running FGDS-related benchmarks, make sure `libfgds.so` is loadable and `fgds_torch.py` has been integrated into the PyTorch build in use (`torch.cuda.fgds` is importable). The `fgds` Python package is not required.

Enter the benchmark directory first:

```bash
cd benchmark
```

Run a single benchmark (arguments: `GPU_ID` `NUM_ROUNDS`):

```bash
python3 bench_fgds_pytorch.py 0 3
python3 bench_gds_pytorch.py 0 3
python3 bench_posix_pytorch.py 0 3
```

Run all benchmarks:

```bash
./run_all_io_tests.sh 0 3
```
