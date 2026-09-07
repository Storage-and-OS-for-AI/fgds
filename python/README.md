# Python API for FGDS

This module provides the Python API for FGDS and adds an external storage backend for lmcache.

## Features
- Use `ctypes` to wrap the FGDS API from the shared library (`libfgds.so`)
- Provide a class for FGDS file operations

## Installation

```bash
cd python
python -m pip install .
```

## Usage

- **Build `libfgds.so`** (see the repo top-level `README.md` for build steps).
- **Make sure `libfgds.so` is discoverable by the dynamic linker** (for example, install it into `/usr/lib64/`, or set `LD_LIBRARY_PATH` to the directory containing it).
- **Verify the Python package is importable**
```bash
python -c "import fgds; print(fgds.__file__)"
```
- **Run simple test for FGDS Python API**
```bash
python test/test.py -f /data/a.safetensors
```
- **Test the lmcache with FGDS backend**：See [fgds-lmcache.md](./docs/fgds-lmcache.md) for more details.

- **Test the fastsafetensors with FGDS copier**：See [fgds-fastsafetensor.md](./docs/fgds-fastsafetensor.md) for more details.