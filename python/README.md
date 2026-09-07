# Python API for FGDS

This module provides the Python API for FGDS and adds an extra storage backend for lmcache.

## Features
- Use `ctypes` to wrap the FGDS API from the shared library (`libfgds.so`)
- Provide a class for FGDS file operations

## Installation

```bash
cd python
python -m pip install .
```

## Usage

1. Build `libfgds.so` (see the repo top-level `README.md` for build steps).
2. Make sure `libfgds.so` is discoverable by the dynamic linker (for example, install it into `/usr/lib64/`, or set `LD_LIBRARY_PATH` to the directory containing it).
3. Verify the Python package is importable:

```bash
python -c "import fgds; print(fgds.__file__)"
```

4. Prepare the FGDS environment and run:

```bash
python test/test.py
```