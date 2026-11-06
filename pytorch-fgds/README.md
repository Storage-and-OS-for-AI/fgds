# pytorch-fgds

> **TL;DR:** This directory is the out-of-tree prototype of `torch.cuda.fgds`, a
> GPUDirect Storage API for FGDS. It is a thin, Python-only wrapper around `libfgds.so`
> that mirrors the `torch.cuda.gds` write/read path. The upstream change is one new Python
> file (`torch/cuda/fgds.py`) plus one import line in `torch/cuda/__init__.py`, with no
> C++/CMake changes. `libfgds.so` is loaded lazily via `ctypes`, so there is no build-time
> dependency and official wheels work unchanged.

Proposed for PyTorch in [pytorch/pytorch#203560](https://github.com/pytorch/pytorch/issues/203560).
`fgds_torch.py` is the module that would land as `torch/cuda/fgds.py`; this directory also
contains a checkpoint read/write example and benchmark scripts.

## Proposed in-tree change

The in-tree change is additive:

| File | Change |
| --- | --- |
| `torch/cuda/fgds.py` | new file: `fgds_torch.py` copied verbatim, no native code |
| `torch/cuda/__init__.py` | add `fgds` to the existing `from . import _device_limits, gds` |
| `docs/source/cuda.md` | a section mirroring "GPUDirect Storage (prototype)", plus a `.. py:module:: torch.cuda.fgds` entry |
| `test/test_cuda.py` | tests mirroring `test_gds_*`, skipped when `fgds.is_available()` is `False` |

No C++/CMake/CI changes, no new build or third-party Python dependencies, and `torch.save` /
`torch.load` are untouched.

## Prerequisites

- Build `libfgds.so` as described in [`docs/getting-started.md`](../docs/getting-started.md).
  The build produces `build/libfgds.so` and has no install target, so point the dynamic linker at
  it (for example `LD_LIBRARY_PATH=$PWD/build`). `liburing` is also required at runtime. 
- The `fgdsfs` kernel module is loaded, and the target filesystem supports `O_DIRECT` (not `tmpfs`).
- NumPy is required by the checkpoint demo and `bench_posix_pytorch.py`; the FGDS/GDS benchmarks do not need it.

Verify that the library can be loaded:

```bash
python3 -c "import ctypes; ctypes.CDLL('libfgds.so'); print('libfgds.so loaded')"
```

## Integrating the out-of-tree FGDS API into PyTorch

`fgds_torch.py` is self-contained; the separate `fgds` Python package (`fgds.fgds_bind`)
is not required.

1. Copy `fgds_torch.py` into the PyTorch source tree as `torch/cuda/fgds.py`.
2. Add `from . import fgds` to `torch/cuda/__init__.py`.

Then use it as `from torch.cuda import fgds`. 

If the module is accepted upstream, this directory will not keep a second copy: `fgds_torch.py`
becomes a thin re-export of `torch.cuda.fgds` (or is removed), so the example and benchmarks here
run against the in-tree module and the two cannot drift.

## Design of `torch.cuda.fgds`

`fgds_torch.py` mirrors the `torch/cuda/gds.py` surface so the common write/read path looks
the same; it binds `libfgds.so` at runtime instead of linking it at build time.

### API compatibility

| Capability | `torch.cuda.gds` | `torch.cuda.fgds` | Compatibility |
| --- | --- | --- | --- |
| Availability | `is_available()` | `is_available()` | same name, runtime probe |
| Register buffer | `gds_register_buffer(s)` | `fgds_register_buffer(s)` | same signature; mandatory for I/O in FGDS, optional performance hint in GDS |
| Deregister buffer | `gds_deregister_buffer(s)` | `fgds_deregister_buffer(s)` | same signature; must run before the last `FgdsFile` for the device closes |
| File object | `GdsFile(filename, flags)` | `FgdsFile(filename, flags, device_id)` | same shape, extra `device_id` |
| Read | `load_storage(storage, offset=0)` | `load_storage(storage, offset=0)` | same signature |
| Write | `save_storage(storage, offset=0)` | `save_storage(storage, offset=0)` | same signature |
| Storage type | CUDA `UntypedStorage` | CUDA `UntypedStorage` | same type |

The common save/load path has the same shape; the file object additionally takes
`device_id`:

```python
import os

import torch
from torch.cuda import gds

path = "/data/fgds_example.bin"
src = torch.randn(16384, device="cuda:0")
f = gds.GdsFile(path, os.O_CREAT | os.O_RDWR)
gds.gds_register_buffer(src.untyped_storage())
f.save_storage(src.untyped_storage(), 0)
gds.gds_deregister_buffer(src.untyped_storage())
del f  # GdsFile closes its fd and handle in __del__

# FGDS: same shape, with device_id.
from torch.cuda import fgds

f = fgds.FgdsFile(path, os.O_CREAT | os.O_RDWR, device_id=0)
fgds.fgds_register_buffer(src.untyped_storage())
f.save_storage(src.untyped_storage(), 0)
fgds.fgds_deregister_buffer(src.untyped_storage())
f.close()
```

### Intentional differences from `torch.cuda.gds`

| Aspect | `torch.cuda.gds` | `torch.cuda.fgds` |
| --- | --- | --- |
| Binding | C++/pybind11 extension, `USE_CUFILE`, links `cuFile` (`libcufile`) | pure Python `ctypes`, lazy `dlopen` |
| Build dependency | build-time flag | none; official wheels work as-is |
| `is_available()` | compile-time constant | runtime probe, cached per process |
| Device selection | implicit (no explicit device argument) | explicit `device_id`; no `torch.cuda.set_device` needed |
| Buffer registration | `cuFile` buffer registration | `fgds_regmem` after an `FgdsFile` exists |
| File offset | `O_DIRECT`-aligned | same, validated in Python (4 KiB) |
| Session | file-handle-based (`register_handle` / `deregister_handle`) | per-device `fgds_open` / `fgds_close`, reference-counted |
| Lifetime API | `__del__` only | context manager and explicit `close()`; `__del__` and `atexit` as fallbacks |

These differences are why `torch.cuda.fgds` is a separate namespace rather than an extension
of `torch.cuda.gds`: extending `torch.cuda.gds` would require backend-specific branches in
the same API, because the two backends do not map one-to-one on session lifetime, buffer
registration, and resource management. The API stays aligned so that the common path is easy
to port and to compare in benchmarks.

### Where it sits

```text
torch.cuda.fgds  (torch/cuda/fgds.py, Python only; out-of-tree: fgds_torch.py)
  |
  |  is_available / fgds_register_buffer / fgds_deregister_buffer / FgdsFile
  v
lazy ctypes.CDLL("libfgds.so")  (no load at import time)
  |
  v
libfgds.so
  |  fgds_open / fgds_close / fgds_read / fgds_write /
  |  fgds_regmem / fgds_deregmem
  |
  +--> /dev/fgds_devN  <-->  fgdsfs kernel module  <-->  NVMe
  |
  +--> GPU memory (registered through fgds_regmem)
```

`/dev/fgds_devN` is the FGDS device node for `device_id=N`.

For comparison, `torch.cuda.gds` is Python (`torch/cuda/gds.py`) plus a C++/pybind11
extension (`torch/csrc/cuda/GdsFile.cpp`) compiled with `USE_CUFILE` and linked against
`cuFile` (`libcufile`). The FGDS wrapper provides the same surface in pure Python and loads
`libfgds.so` lazily with `ctypes`, so `import torch` never depends on `libfgds` being
installed.

### Lifecycle / call flow

```text
FgdsFile(path, flags, device_id)
  +-- validate device_id
  +-- os.open(path, flags | O_DIRECT)
  +-- first FgdsFile for device: fgds_open(device_id)
  +-- build fgds_fileid_t(fd, device_id)

fgds_register_buffer(storage)
  +-- require active FgdsFile for the storage's device
  +-- fgds_regmem(device_id, ptr, nbytes, &host); track (ptr, nbytes)

save_storage(storage, offset) / load_storage(storage, offset)
  +-- storage.device must match file.device_id
  +-- offset must be a non-negative multiple of 4 KiB
  +-- storage must be registered with the exact (ptr, nbytes)
  +-- fgds_write/read(fid, ptr, 0, nbytes, file_offset)
      under the per-device lock

close()
  +-- os.close(fd)
  +-- last FgdsFile for device:
      +-- fgds_deregmem(...) for all still-registered buffers
      +-- fgds_close(device_id)
```

### Runtime model

- **Lazy loading.** The library is loaded on first use, not at import; the result of the
  first probe is cached, and `is_available()` returns `False` if loading fails.
- **Per-device sessions.** The first `FgdsFile` for a `device_id` calls `fgds_open`, the
  last one calls `fgds_close`; an `atexit` handler closes sessions still open at shutdown.
  Buffers still registered when the session closes are deregistered first, because
  `fgds_close` does not release those kernel mappings itself.
- **Multi-GPU.** `device_id` fully determines the target device; different devices have
  independent sessions, registrations, and I/O, so one process can drive several GPUs.
- **Threading.** A per-device lock serializes transfers on a given device; operations on
  different devices are independent and may run in parallel.
- **SONAME.** `libfgds.so` is loaded by name and its required symbols are checked at
  load time. SONAME/ABI versioning and a minimum supported version remain to be
  defined (tracked as a TODO in `fgds_torch.py`).

### Public interface

```python
__all__ = [
    "is_available",
    "fgds_register_buffer",
    "fgds_deregister_buffer",
    "FgdsFile",
]
```

## Checkpoint read/write demo

`example/checkpoint_demo.py` builds a synthetic `state_dict`, serializes it with `torch.save`,
writes the bytes to disk with `FgdsFile.save_storage`, and reads them back with
`FgdsFile.load_storage`. It then checks the result byte-for-byte, loads it with native
`torch.load`, and prints SHA-256 digests.

The demo is a transport test: it shows that FGDS can move a serialized checkpoint payload,
but does not integrate FGDS with `torch.save`, `torch.load`, or
`torch.distributed.checkpoint`.

Run from the repository root:

```bash
python3 pytorch-fgds/example/checkpoint_demo.py \
    --path /data/fgds_checkpoint_demo.pt \
    --size-mb 256 \
    --dtype float16 \
    --device-id 0 \
    --chunk-mb 64
```

This demo uses `torch.save` bytes and therefore serializes on the CPU side: it validates
correctness, not the pure GPU-direct performance path (see the benchmarks below). Pass
`--chunk-mb 1` to exercise the multi-chunk and tail-read path.

## Benchmarks

The `benchmark/` directory compares different I/O paths (default test file size: 10 GiB):

| Script | I/O path | Notes |
| --- | --- | --- |
| `bench_fgds_pytorch.py` | `torch.cuda.fgds` | one write + one read per round |
| `bench_gds_pytorch.py` | `torch.cuda.gds` (`cuFile`) | one write + one read per round |
| `bench_posix_pytorch.py` | `torch.save` / `torch.load` (buffered) and POSIX `writev` / `readv` with `O_DIRECT` | two paths |

All three scripts verify round-trip correctness; `bench_posix_pytorch.py` checks both of its
paths.

Each script prints per-round and average latency/bandwidth. Run one or more benchmarks
(arguments: `GPU_ID` and `NUM_ROUNDS`):

```bash
cd pytorch-fgds/benchmark 
python3 bench_fgds_pytorch.py 0 3
python3 bench_gds_pytorch.py 0 3
python3 bench_posix_pytorch.py 0 3

# or run all of them (with a 3-second sleep between scripts)
./run_all_io_tests.sh 0 3
```

## Testing

Two standalone entry points validate the proposed module against an installed
`torch` (the installed tree needs both `torch/cuda/fgds.py` and the
`from . import fgds` line). The unittest suite skips hardware-dependent cases when
FGDS is unavailable; the e2e script requires an FGDS-enabled machine and exits
non-zero when prerequisites are missing.

| Entry point | Purpose |
| --- | --- |
| `tests/verify_fgds_e2e.py` | end-to-end self-check; prints `[PASS]`/`[FAIL]` per check and exits non-zero on failure |
| `tests/verify_fgds_unittest.py` | a standalone variant of the `TestFGDS` suite that the in-tree change adds to `test/test_cuda.py` (requires `psutil`) |

Together they cover round-trips across sizes, dtypes and offsets, persistence across
reopen, session lifetime and multi-device use, error paths, and opt-in stress cases.
Every round-trip case checks byte-exact content (`torch.equal` / `assertEqual`).

Run them from the fgds repository root:

```bash
export LD_LIBRARY_PATH="$PWD/build${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TMPDIR=/data/fgds_test                              # ext4/xfs, O_DIRECT-capable
mkdir -p "$TMPDIR"

python3 pytorch-fgds/tests/verify_fgds_unittest.py -v      # unittest suite
python3 pytorch-fgds/tests/verify_fgds_e2e.py              # end-to-end self-check
```

`verify_fgds_e2e.py` reads `FGDS_TMPDIR` (falling back to `TMPDIR`); stress cases are
opt-in via `FGDS_RUN_LARGE=1` (e2e, 1.5 GiB) and `FGDS_RUN_CONCURRENCY=1` (8 threads,
both suites). The proposed in-tree suite can be run with `python test/test_cuda.py TestFGDS -v`.