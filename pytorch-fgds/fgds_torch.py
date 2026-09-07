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

Kylin FGDS helpers mirroring :mod:`torch.cuda.gds` (see ``fgds_design.md`` in this directory).
"""

import os
import sys
import threading
from typing import Any

import torch

from torch.types import Storage

__all__: list[str] = [
    "fgds_register_buffer",
    "fgds_deregister_buffer",
    "FgdsFile",
]

# Require an installed `fgds` package; do not mutate sys.path for source-tree imports.
_FGDS_BIND_UNAVAILABLE = (
    "fgds bindings are not available. Install the `fgds` Python package "
    "(from the repository `python/` directory: `python -m pip install .`) "
    "and ensure `libfgds.so` can be loaded (for example via LD_LIBRARY_PATH)."
)

_fgds_bind: Any = None
_fgds_import_error: BaseException | None = None


def _load_fgds_bind() -> Any:
    # Lazily import and cache installed fgds bindings, surfacing actionable errors.
    global _fgds_bind, _fgds_import_error
    if _fgds_bind is not None:
        return _fgds_bind
    if _fgds_import_error is not None:
        raise RuntimeError(_FGDS_BIND_UNAVAILABLE) from _fgds_import_error
    try:
        import fgds.fgds_bind as fb  # type: ignore[import-not-found]

        _fgds_bind = fb
        return fb
    except BaseException as e:
        _fgds_import_error = e
        raise RuntimeError(_FGDS_BIND_UNAVAILABLE) from e


_file_lock = threading.Lock()
# Count of live FgdsFile instances per device_id. fgds_open on first file; fgds_close when last.
_fgds_file_count: dict[int, int] = {}


def _fgds_file_constructed(device_id: int) -> None:
    # Open fgds session on first live FgdsFile for a device.
    fb = _load_fgds_bind()
    with _file_lock:
        c = _fgds_file_count.get(device_id, 0)
        if c == 0:
            torch.cuda.set_device(device_id)
            fb.fgds_open(device_id)
        _fgds_file_count[device_id] = c + 1


def _fgds_file_destroyed(device_id: int) -> None:
    # Close fgds session when the last live FgdsFile for a device is gone.
    fb = _load_fgds_bind()
    with _file_lock:
        c = _fgds_file_count.get(device_id, 0)
        if c <= 0:
            return
        c -= 1
        if c == 0:
            _fgds_file_count.pop(device_id, None)
            fb.fgds_close(device_id)
        else:
            _fgds_file_count[device_id] = c


def _fgds_file_session_active(device_id: int) -> bool:
    # Check whether a device currently has an active fgds session.
    with _file_lock:
        return _fgds_file_count.get(device_id, 0) > 0


def _cuda_device_index(storage: Storage) -> int:
    # Resolve storage device index and ensure storage is on CUDA.
    dev = storage.device
    if dev.type != "cuda":
        raise RuntimeError(f"fgds requires CUDA storage, got device {dev}")
    idx = dev.index
    if idx is None:
        return torch.cuda.current_device()
    return idx


def fgds_register_buffer(s: Storage) -> None:
    # Register a CUDA storage with fgds_regmem for direct fgds IO.
    """Registers GPU memory for FGDS using ``fgds_regmem`` for the storage's device.

    Must be called only **after** constructing a :class:`FgdsFile` for the same CUDA
    ``device_id`` so that ``fgds_open`` has already run. Pair with
    :func:`fgds_deregister_buffer` before the :class:`FgdsFile` is destroyed.

    Example::

        >>> # xdoctest: +SKIP("fgds filesystem and libfgds required")
        >>> # fh = FgdsFile(path, flags, device_id=0)  # opens fgds session first
        >>> t = torch.randn(1024, device="cuda")
        >>> fgds_register_buffer(t.untyped_storage())

    Args:
        s (Storage): CUDA buffer to register.
    """
    import ctypes

    device_id = _cuda_device_index(s)
    if not _fgds_file_session_active(device_id):
        raise RuntimeError(
            "fgds_register_buffer requires an active FgdsFile for this device: "
            "construct FgdsFile(..., device_id=...) first, then register buffers."
        )
    fb = _load_fgds_bind()
    torch.cuda.set_device(device_id)
    ptr = ctypes.c_void_p(s.data_ptr())
    nbytes = s.nbytes()
    target = ctypes.c_void_p()
    ret = fb.fgds_regmem(device_id, ptr, nbytes, ctypes.byref(target))
    if ret != 0:
        raise RuntimeError(f"fgds_regmem failed with code {ret}")


def fgds_deregister_buffer(s: Storage) -> None:
    # Deregister a CUDA storage previously registered with fgds_regmem.
    """Deregisters GPU memory with ``fgds_deregmem`` (does not call ``fgds_close``)."""
    import ctypes

    fb = _load_fgds_bind()
    device_id = _cuda_device_index(s)
    ptr = ctypes.c_void_p(s.data_ptr())
    nbytes = s.nbytes()
    ret = fb.fgds_deregmem(device_id, ptr, nbytes)
    if ret != 0:
        raise RuntimeError(f"fgds_deregmem failed with code {ret}")


class FgdsFile:
    # File wrapper that binds an fd to fgds read/write operations.
    r"""FGDS file IO using ``fgds_read`` / ``fgds_write`` with OS file ``fd``.

    ``fgds_open(device_id)`` is called when the first :class:`FgdsFile` for that
    device is constructed; ``fgds_close`` runs when the last such instance for that
    device is destroyed. Call :func:`fgds_register_buffer` only **after** creating
    :class:`FgdsFile`, and :func:`fgds_deregister_buffer` before this object is
    destroyed.

    Compared to :class:`torch.cuda.gds.GdsFile`, the constructor takes an extra
    ``device_id`` argument for ``fgds_open(device_id)`` and ``fgds_fileid_t.deviceID``.

    Args:
        filename (str): Path passed to :func:`os.open`.
        flags (int): Flags for :func:`os.open`; ``os.O_DIRECT`` is ORed in when available.
        device_id (int): CUDA device index for FGDS (``fgds_open`` / ``fgds_fileid_t``).

    Example::

        >>> # xdoctest: +SKIP("fgds filesystem and libfgds required")
        >>> import os
        >>> src = torch.randn(1024, device="cuda:0")
        >>> # fh = FgdsFile(f, os.O_CREAT | os.O_RDWR, device_id=0)
        >>> # fgds_register_buffer(src.untyped_storage())
        >>> # fh.save_storage(src.untyped_storage(), offset=0)
        >>> # fh.load_storage(dest.untyped_storage(), offset=0)
        >>> # fgds_deregister_buffer(src.untyped_storage())
    """

    def __init__(self, filename: str, flags: int, device_id: int) -> None:
        # Open O_DIRECT fd, ensure per-device fgds session, and build file handle id.
        if sys.platform == "win32":
            raise RuntimeError("FgdsFile is not supported on this platform.")
        self.filename = filename
        self.flags = flags
        self.device_id = int(device_id)
        self.fd: int | None = None
        self._fid: Any = None
        self._session_held = False

        torch.cuda.set_device(self.device_id)
        self.fd = os.open(filename, flags | os.O_DIRECT)  # type: ignore[attr-defined]
        try:
            _fgds_file_constructed(self.device_id)
            self._session_held = True
            fb = _load_fgds_bind()
            self._fid = fb.fgds_fileid_t(fd=self.fd, deviceID=self.device_id)
        except Exception:
            if self._session_held:
                _fgds_file_destroyed(self.device_id)
                self._session_held = False
            if self.fd is not None:
                os.close(self.fd)
                self.fd = None
            raise

    def __del__(self) -> None:
        # Best-effort resource cleanup for fd and per-device session reference.
        fd = self.fd
        dev_id = self.device_id
        held = self._session_held
        self.fd = None
        self._session_held = False
        self._fid = None
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        if held:
            try:
                _fgds_file_destroyed(dev_id)
            except RuntimeError:
                pass

    def _check_fid(self) -> None:
        # Validate that the file descriptor and fgds file id are initialized.
        if self._fid is None or self.fd is None:
            raise AssertionError("FgdsFile is not open.")

    def _check_storage_device(self, storage: Storage) -> None:
        # Ensure storage device matches the FgdsFile device_id.
        if _cuda_device_index(storage) != self.device_id:
            raise RuntimeError(
                f"Storage is on cuda:{_cuda_device_index(storage)} but FgdsFile uses device_id={self.device_id}"
            )

    def load_storage(self, storage: Storage, offset: int = 0) -> None:
        # Read bytes from file into CUDA storage via fgds_read.
        """Loads ``storage.nbytes()`` bytes from the file at ``offset`` into ``storage`` (``fgds_read``)."""
        import ctypes

        self._check_fid()
        self._check_storage_device(storage)
        torch.cuda.set_device(self.device_id)
        fb = _load_fgds_bind()
        ptr = ctypes.c_void_p(storage.data_ptr())
        n = storage.nbytes()
        ret = int(fb.fgds_read(self._fid, ptr, 0, n, offset))
        if ret != n:
            raise RuntimeError(f"fgds_read expected {n} bytes, got {ret}")

    def save_storage(self, storage: Storage, offset: int = 0) -> None:
        # Write bytes from CUDA storage to file via fgds_write.
        """Writes ``storage.nbytes()`` bytes from ``storage`` to the file at ``offset`` (``fgds_write``)."""
        import ctypes

        self._check_fid()
        self._check_storage_device(storage)
        torch.cuda.set_device(self.device_id)
        fb = _load_fgds_bind()
        ptr = ctypes.c_void_p(storage.data_ptr())
        n = storage.nbytes()
        ret = int(fb.fgds_write(self._fid, ptr, 0, n, offset))
        if ret != n:
            raise RuntimeError(f"fgds_write expected {n} bytes, got {ret}")
