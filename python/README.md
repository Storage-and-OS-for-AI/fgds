# Python api for Fgds 

This module provide the python api for fgds and add the extra storage backend of fgds for lmcache

## Features
- Use ctypes to wrapper the fgds api from dynamic library of libfgds.so
- Provide a class for fgds file operations.

## Installation

```bash
python setup install
```

## Usage

1. compile the libfgds.so
2. deploy the libfgds.so to /usr/lib64/ or any other library path
3. python -c "import fgds;print(fgds.__file__)"
4. prepare the fgds environment and run the test/test.py

## License
Apache-2.0 License
