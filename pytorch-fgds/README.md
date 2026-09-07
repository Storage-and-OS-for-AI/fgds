# pytorch-fgds

This directory contains `fgds_torch.py` and benchmark scripts for performance comparison.

## Integrating the FGDS API into PyTorch

`torch.cuda.fgds` depends on the installed `fgds` Python package (which provides `fgds.fgds_bind`) and a loadable `libfgds.so`. Complete the underlying dependency setup before integration:

1. Build and install `libfgds.so` as described in the repo top-level docs, and make sure the dynamic linker can find it (for example, install it into a system library path, or set `LD_LIBRARY_PATH`).
2. Install this repo's Python package:

```bash
cd python
# Choose one:
python -m pip install .      # regular install
# python -m pip install -e .  # editable install for development
```

Verify that the package is importable:

```bash
python -c "import fgds.fgds_bind; print(fgds.fgds_bind.__file__)"
```

Then integrate this directory's `fgds_torch.py` into PyTorch:

1. Copy `fgds_torch.py` from this directory into the PyTorch source tree and **rename** it to `fgds.py`:
     - `torch/cuda/fgds.py`
2. Edit `torch/cuda/__init__.py` in the PyTorch source tree and add the import:
     - `from . import fgds`

After that, you can use the FGDS API from Python via `from torch.cuda import fgds`.

## Design of `fgds_torch.py`

The Python API in `fgds_torch.py` follows the design of `torch/cuda/gds.py`:

- `gds.py` exports `gds_register_buffer`, `gds_deregister_buffer`, and `GdsFile`
- `fgds_torch.py` exports the corresponding `fgds_register_buffer`, `fgds_deregister_buffer`, and `FgdsFile`
- `GdsFile` provides `load_storage` / `save_storage`
- `FgdsFile` also provides `load_storage` / `save_storage`

The goal is to keep the usage of `torch.cuda.fgds` consistent with `torch.cuda.gds`, which makes migration and comparative testing easier.

Main functions/methods in `fgds_torch.py`:

- `_load_fgds_bind`
    - Loads the installed `fgds.fgds_bind` via a normal `import` and caches it once; on failure, prompts to install the `fgds` package and ensure `libfgds.so` is loadable.

- `_fgds_file_constructed` / `_fgds_file_destroyed`
    - Tracks `FgdsFile` instance counts per `device_id`.
    - Calls `fgds_open(device_id)` when the first instance is created, and `fgds_close(device_id)` when the last instance is destroyed.

- `_fgds_file_session_active`
    - Checks whether the given device already has an active FGDS session (used as a precondition check before registering a buffer).

- `_cuda_device_index`
    - Resolves the CUDA device index from a `Storage` and verifies that the storage is on CUDA.

- `fgds_register_buffer`
    - Calls `fgds_regmem` to register GPU storage (requires an `FgdsFile` to have been created on the same device first).

- `fgds_deregister_buffer`
    - Calls `fgds_deregmem` to deregister GPU storage.

- `FgdsFile.__init__`
    - Opens the file (`os.O_DIRECT`), holds an FGDS session, and creates a `fgds_fileid_t`.

- `FgdsFile.__del__`
    - Releases the file fd and, when needed, decrements the session count to trigger `fgds_close`.

- `FgdsFile.load_storage`
    - Calls `fgds_read` to read file contents into CUDA storage.

- `FgdsFile.save_storage`
    - Calls `fgds_write` to write CUDA storage contents to the file.

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

Before running FGDS-related benchmarks, make sure the `fgds` Python package is installed as described above, `libfgds.so` is loadable, and `fgds_torch.py` has been integrated into the PyTorch build in use (`torch.cuda.fgds` is importable).

Enter the benchmark directory first:

```bash
cd benchmark
```

Run a single benchmark (arguments: `GPU_ID` `NUM_ROUNDS`):

```bash
python bench_fgds_pytorch.py 0 3
python bench_gds_pytorch.py 0 3
python bench_posix_pytorch.py 0 3
```

Run all benchmarks:

```bash
./run_all_io_tests.sh 0 3
```
