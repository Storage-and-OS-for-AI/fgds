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

from fgds import FgdsDriver, Fgds
from lmcache.v1.memory_management import GPUMemoryAllocator

DEVICE_ID = 0 # only support the device id 0 for now.
DATA_PATH = '/mnt/fgds/data.bin'

class FgdsMemoryAllocator(GPUMemoryAllocator):
    def __init__(self, size: int, device=None):
        from fgds.fgds_bind import fgds_regmem, fgds_deregmem

        self.fgdsBufDeReg = fgds_deregmem
        if device is None:
            if torch.cuda.is_available():
                device = f"cuda:{torch.cuda.current_device()}"
            else:
                device = "cpu:0"
        super().__init__(size, device, align_bytes=4096)
        self.size = size
        self.device = DEVICE_ID 
        self.base_pointer = self.tensor.data_ptr()
        void_ptr = ctypes.c_void_p()
        host_ptr = ctypes.POINTER(ctypes.c_void_p)(void_ptr)
        fgds_regmem(self.device, ctypes.c_void_p(self.base_pointer), ctypes.c_size_t(self.size), host_ptr)
        self.host_ptr = void_ptr

    def __del__(self):
        self.fgdsBufDeReg(self.device, ctypes.c_void_p(self.base_pointer), ctypes.c_size_t(self.size))

    def __str__(self):
        return "FgdsMemoryAllocator"


if __name__ == "__main__":


    fgds_driver = FgdsDriver(DEVICE_ID)

    shape = torch.Size([2, 36, 256, 1024])
    dtype = torch.bfloat16 
    alloc = FgdsMemoryAllocator(37748736, device="cuda:%d" % DEVICE_ID)

    
    with Fgds(DATA_PATH, "r", use_direct_io=True, device_id=DEVICE_ID) as f:
        start = time.time()
        r = f.read(
            alloc.base_pointer,
            37748736,
            file_offset=0,
            dev_offset=0,
        )
        load_du = time.time() - start
        print(f"read data take {load_du:.6f}s")

    alloc = None
    fgds_driver = None
