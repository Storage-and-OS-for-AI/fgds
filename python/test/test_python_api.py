#!/usr/bin/env python
"""
Copyright (c) 2025-2026 KylinSoft Co., Ltd.

SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

"""

import torch
import os
import sys
import time
import ctypes
import argparse

from fgds import FgdsDriver, Fgds
from fgds.fgds_bind import fgds_regmem, fgds_deregmem

DEVICE_ID = 0  # only support the device id 0 for now.
BUF_SIZE = 1048576


class FgdsMemory:
    """Simple GPU memory wrapper using PyTorch for allocation, with FGDS registration."""

    def __init__(self, size: int, device: str = None):
        self.size = size
        self.device_id = DEVICE_ID

        if device is None:
            if torch.cuda.is_available():
                device = f"cuda:{DEVICE_ID}"
            else:
                raise RuntimeError("CUDA is not available")

        self.device = device
        # Allocate GPU memory using PyTorch (native CUDA memory allocation)
        self.tensor = torch.empty(size, dtype=torch.uint8, device=device)
        self.base_pointer = self.tensor.data_ptr()

        # Register memory with FGDS
        void_ptr = ctypes.c_void_p()
        host_ptr = ctypes.POINTER(ctypes.c_void_p)(void_ptr)
        fgds_regmem(
            self.device_id,
            ctypes.c_void_p(self.base_pointer),
            ctypes.c_size_t(self.size),
            host_ptr,
        )
        self.host_ptr = void_ptr

    def __del__(self):
        try:
            fgds_deregmem(
                self.device_id,
                ctypes.c_void_p(self.base_pointer),
                ctypes.c_size_t(self.size),
            )
        except Exception:
            pass

    def __str__(self):
        return f"FgdsMemory(device={self.device}, size={self.size}, ptr=0x{self.base_pointer:x})"


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="FGDS Python API Test")
    parser.add_argument("-f", "--file", type=str, required=True, help="Path to the data file")
    args = parser.parse_args()

    fgds_driver = FgdsDriver(DEVICE_ID)
    alloc = FgdsMemory(BUF_SIZE, device=f"cuda:{DEVICE_ID}")

    print(f"Allocated {BUF_SIZE} bytes on {alloc.device}")
    print(f"GPU pointer: 0x{alloc.base_pointer:x}")

    with Fgds(args.file, "r", use_direct_io=True, device_id=DEVICE_ID) as f:
        start = time.time()
        r = f.read(
            alloc.base_pointer,
            BUF_SIZE,
            file_offset=0,
            dev_offset=0,
        )
        load_dur = time.time() - start
        print(f"Read {r} bytes in {load_dur:.6f}s ({r / load_dur / 1024 / 1024:.2f} MiB/s)")

    # Cleanup
    del alloc
    del fgds_driver
