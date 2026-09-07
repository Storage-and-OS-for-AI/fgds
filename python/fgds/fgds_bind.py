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

import ctypes
import os

ctypes.CDLL("libcudart.so", mode=ctypes.RTLD_GLOBAL)
ctypes.CDLL("libcuda.so", mode=ctypes.RTLD_GLOBAL)
ctypes.CDLL("libfgds.so", mode=ctypes.RTLD_GLOBAL)

libfgds = ctypes.CDLL("libfgds.so")
cuda = ctypes.CDLL("libcuda.so")

class fgds_fileid_t(ctypes.Structure):
    _fields_ = [("fd", ctypes.c_int), ("deviceID", ctypes.c_int)]
class xfer_addr(ctypes.Structure):
    _fields_ = [("target_addr", ctypes.c_void_p), ("nbyte", ctypes.c_size_t)]

MAX_NR_ADDR = 4
class fgds_xfer_addr(ctypes.Structure):
    _fields_ = [("nr_xfer_addrs", ctypes.c_uint32), ("x_addrs", xfer_addr*1)]

cudaError_t = ctypes.c_int

libfgds.fgds_open.restype                   = ctypes.c_int
libfgds.fgds_close.restype                  = ctypes.c_int
libfgds.fgds_read.restype                   = ctypes.c_ssize_t
libfgds.fgds_write.restype                  = ctypes.c_ssize_t
libfgds.fgds_do_xfer_addr.restype           = ctypes.POINTER(fgds_xfer_addr)
libfgds.fgds_regmem.restype                 = ctypes.c_int
libfgds.fgds_deregmem.restype               = ctypes.c_int
libfgds.fgds_read_async.restype             = cudaError_t
libfgds.fgds_write_async.restype            = cudaError_t

CUstream = ctypes.c_void_p

libfgds.fgds_open.argtypes = [ctypes.c_int]
libfgds.fgds_close.argtypes = [ctypes.c_int]
libfgds.fgds_read.argtypes = [fgds_fileid_t, ctypes.c_void_p, ctypes.c_longlong, ctypes.c_size_t, ctypes.c_longlong]
libfgds.fgds_write.argtypes = [fgds_fileid_t, ctypes.c_void_p, ctypes.c_longlong, ctypes.c_size_t, ctypes.c_longlong]
libfgds.fgds_do_xfer_addr.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_longlong, ctypes.c_size_t]
libfgds.fgds_regmem.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_void_p)]
libfgds.fgds_deregmem.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
libfgds.fgds_read_async.argtypes = [fgds_fileid_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_longlong, ctypes.POINTER(ctypes.c_ssize_t), CUstream]
libfgds.fgds_write_async.argtypes = [fgds_fileid_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_longlong, ctypes.POINTER(ctypes.c_ssize_t), CUstream]

def _check_ret(ret, name):
    if ret < 0:
        raise RuntimeError(f"{name} failed with return code: {ret}")
def fgds_open(device_id: int) -> ctypes.c_int:
    ret = libfgds.fgds_open(device_id)
    _check_ret(ret, "fgds_open")
    return ret

def fgds_close(device_id: int) -> ctypes.c_int:
    ret = libfgds.fgds_close(device_id)
    _check_ret(ret, "fgds_close")
    return ret

def fgds_read(fid: fgds_fileid_t, buf: ctypes.c_void_p, buf_offset: ctypes.c_longlong, nbyte: ctypes.c_ssize_t, f_offset: ctypes.c_longlong) -> ctypes.c_ssize_t:
    ret = libfgds.fgds_read(fid, buf,  buf_offset, nbyte, f_offset)
    if ret < 0:
        raise RuntimeError(f"fgds_read failed with return code: {ret}")
    return ret

def fgds_write(fid: fgds_fileid_t, buf: ctypes.c_void_p, buf_offset: ctypes.c_longlong, nbyte: ctypes.c_ssize_t, f_offset: ctypes.c_longlong) -> ctypes.c_ssize_t:
    ret = libfgds.fgds_write(fid, buf, buf_offset, nbyte, f_offset)
    if ret < 0:
        raise RuntimeError(f"fgds_write failed with return code: {ret}")
    return ret

def fgds_do_xfer_addr(device_id: ctypes.c_int, buf: ctypes.c_void_p, buf_offset: ctypes.c_longlong, nbyte: ctypes.c_size_t) -> ctypes.POINTER(fgds_xfer_addr):
    return libfgds.fgds_do_xfer_addr(device_id, buf, buf_offset, nbyte)

def fgds_regmem(device_id: ctypes.c_int, gpu_buffer: ctypes.c_void_p, len: ctypes.c_size_t, target_addr: ctypes.POINTER(ctypes.c_void_p)) -> ctypes.c_int:
    ret = libfgds.fgds_regmem(device_id, gpu_buffer, len, target_addr)
    if ret < 0:
        raise RuntimeError(f"fgds_regmem failed with return code: {ret}")
    return ret

def fgds_deregmem(device_id: ctypes.c_int, addr: ctypes.c_void_p, len: ctypes.c_size_t) -> ctypes.c_int:
    ret = libfgds.fgds_deregmem(device_id, addr, len)
    if ret < 0:
        raise RuntimeError(f"fgds_deregmem failed with return code: {ret}")
    return ret

def fgds_read_async(fid: fgds_fileid_t, buf: ctypes.c_void_p, nbytes: ctypes.c_size_t, offset: ctypes.c_longlong, bytes_done: ctypes.c_ssize_t, stream: CUstream) -> cudaError_t:
    return libfgds.fgds_read_async(fid, buf, nbytes, offset, bytes_done, stream)

def fgds_write_async(fid: fgds_fileid_t, buf: ctypes.c_void_p, nbytes: ctypes.c_size_t, offset: ctypes.c_longlong, bytes_done: ctypes.c_ssize_t, stream: CUstream) -> cudaError_t:
    return libfgds.fgds_write_async(fid, buf, nbytes, offset, bytes_done, stream)
