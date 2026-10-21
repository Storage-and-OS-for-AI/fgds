"""Python bindings for Fast GPUDirect Storage (FGDS) via ``libfgds``.

The library is loaded lazily, so importing this module succeeds even when
``libfgds.so`` is not installed.

Sessions are per-device
-----------------------
All :class:`FgdsFile` objects for a device share one session: ``fgds_open`` runs
when the first one is constructed and ``fgds_close`` when the last one is
closed. Because sessions are per-device, one process can use several devices
at once: create :class:`FgdsFile` objects (and register buffers) independently
for each ``device_id``. ``libfgds`` selects the target device from ``device_id``
and never reads the current CUDA device, so no ``torch.cuda.device`` guard is
needed around the entry points. Deregister buffers explicitly with
:func:`fgds_deregister_buffer`; any that are still registered when the last
:class:`FgdsFile` for the device is closed are deregistered there, because
``fgds_close`` does not release their kernel mappings itself.

Thread safety
-------------
The Python layer serializes operations per device with a per-device lock, so a
session teardown (``fgds_close``, run when the last :class:`FgdsFile` for a
device is closed) cannot interleave with buffer registration or I/O in flight
on that device. Operations on different devices take different locks and are
independent, so transfers on different devices can proceed in parallel.

The lock is held for the whole transfer, so operations on one device are fully
serialized and multi-threaded I/O against a single GPU does not scale. (The
io_uring paths in ``libfgds`` already serialize internally; the direct
pread/pwrite and ``fgds_regmem`` paths are additionally serialized by this lock.
``libfgds`` also looks up its registration list without holding its own lock, so
narrowing this one to a per-device in-flight count needs an upstream fix first.)
"""

from __future__ import annotations

import atexit
import ctypes
import os
import sys
import threading
from typing import Any

import torch
from torch.types import Storage


__all__: list[str] = [
    "is_available",
    "fgds_register_buffer",
    "fgds_deregister_buffer",
    "FgdsFile",
]

# fgds_regmem only accepts buffers whose length is a positive multiple of this.
_FGDS_GPU_PAGE_SIZE = 64 * 1024

# O_DIRECT requires the file offset to be aligned; libfgds does not validate the
# offset itself and fails with a generic error (plus a stderr line), so validate
# here for a clear error message.
_FGDS_ALIGNMENT = 4096

_FGDS_UNAVAILABLE_MSG = (
    "FGDS is not available: libfgds.so could not be loaded. Install libfgds and "
    "make it discoverable by the dynamic linker (for example via LD_LIBRARY_PATH)."
)

_FGDS_ABI_MSG = (
    "FGDS is not available: libfgds.so was loaded but required symbols are "
    "missing; the installed libfgds is too old or incompatible."
)


class _fgds_fileid_t(ctypes.Structure):
    _fields_ = [("fd", ctypes.c_int), ("deviceID", ctypes.c_int)]


def _build_bindings() -> Any:
    # Only called from _load_fgds_bind, never at import time. libfgds.so leaves the
    # CUDA-runtime symbols referenced by its async entry points (for example
    # cudaLaunchHostFunc) unresolved, so bind lazily instead of requiring libcudart
    # to already be in the global symbol scope. This module never calls async entry
    # points. Required symbols are still resolved eagerly by the attribute accesses
    # below, so a missing or incompatible library still fails here.
    # TODO: resolve the libfgds SONAME policy (open question in the RFC);
    # "libfgds.so" is the development symlink name.
    # ABI assumptions to pin down with the SONAME/version contract: the two
    # c_longlong arguments below are libfgds's 64-bit off_t (fgds_read/write take
    # off_t buf_offset/f_offset). That holds on x86-64/aarch64 glibc, but a
    # libfgds built with 32-bit off_t would silently misalign the ABI.
    # RTLD_* are POSIX-only. On platforms without them CDLL("libfgds.so") fails
    # below and is_available() reports False, which is the intended behavior.
    # libfgds.so also needs liburing at runtime (it is in its DT_NEEDED), so a
    # machine with libfgds but no liburing fails to load here; the OSError text is
    # surfaced by _load_fgds_bind so that case is not mistaken for a missing
    # libfgds.
    lib = ctypes.CDLL(
        "libfgds.so",
        mode=getattr(os, "RTLD_LAZY", 0) | getattr(os, "RTLD_LOCAL", 0),
        use_errno=True,
    )

    lib.fgds_open.restype = ctypes.c_int
    lib.fgds_close.restype = ctypes.c_int
    lib.fgds_read.restype = ctypes.c_ssize_t
    lib.fgds_write.restype = ctypes.c_ssize_t
    lib.fgds_regmem.restype = ctypes.c_int
    lib.fgds_deregmem.restype = ctypes.c_int

    lib.fgds_open.argtypes = [ctypes.c_int]
    lib.fgds_close.argtypes = [ctypes.c_int]
    read_write_argtypes = [
        _fgds_fileid_t,
        ctypes.c_void_p,
        ctypes.c_longlong,
        ctypes.c_ssize_t,
        ctypes.c_longlong,
    ]
    lib.fgds_read.argtypes = read_write_argtypes
    lib.fgds_write.argtypes = read_write_argtypes
    lib.fgds_regmem.argtypes = [
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    lib.fgds_deregmem.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
    return lib


def _errno_detail() -> str:
    # With use_errno=True, ctypes saves the errno set by the most recent libfgds
    # call; surface it when non-zero so failures are diagnosable. libfgds leaves
    # errno at 0 on some paths, in which case we omit it rather than print
    # "Success".
    err = ctypes.get_errno()
    if err == 0:
        return ""
    return f" (errno {err}: {os.strerror(err)})"


_fgds_lock = threading.Lock()
_fgds_state: int = 0  # 0 = not tried, 1 = available, -1 = unavailable
_fgds_bind: Any = None
# Message for the unavailable state; recorded when the probe first fails so the
# cached re-raise keeps the OSError vs AttributeError distinction.
_fgds_error: str = _FGDS_UNAVAILABLE_MSG


def _load_fgds_bind() -> Any:
    global _fgds_state, _fgds_bind, _fgds_error
    if _fgds_state == 1:
        return _fgds_bind
    if _fgds_state == -1:
        raise RuntimeError(_fgds_error)
    with _fgds_lock:
        if _fgds_state == 1:
            return _fgds_bind
        if _fgds_state == -1:
            raise RuntimeError(_fgds_error)
        try:
            bind = _build_bindings()
        except OSError as e:
            # Keep the loader's own text: a missing transitive dependency of
            # libfgds.so (liburing, for example) also surfaces as an OSError here,
            # and "install libfgds" alone would point the user the wrong way.
            msg = f"{_FGDS_UNAVAILABLE_MSG} Loader error: {e}"
            _fgds_state = -1
            _fgds_error = msg
            raise RuntimeError(msg) from e
        except AttributeError as e:
            _fgds_state = -1
            _fgds_error = _FGDS_ABI_MSG
            raise RuntimeError(_FGDS_ABI_MSG) from e
        _fgds_bind = bind
        _fgds_state = 1
        return bind


def is_available() -> bool:
    r"""Return ``True`` if ``libfgds.so`` can be loaded and exports the expected
    entry points at runtime.

    FGDS is loaded dynamically, so this does not depend on how PyTorch was built.
    It only reports whether the library is loadable and ABI-compatible; the ``fgdsfs``
    kernel module and an ``O_DIRECT``-capable filesystem are still required to
    actually perform I/O.

    The result of the first probe is cached for the lifetime of the process: if
    the first probe fails, making the library available later (for example by
    changing ``LD_LIBRARY_PATH``) does not flip this to ``True`` without a restart.
    """
    try:
        _load_fgds_bind()
    except RuntimeError:
        return False
    return True


# Serialize each device's session separately rather than with one process-wide
# lock. A per-device lock preserves the close-vs-I/O mutual exclusion and the
# session/registration checks for the device while letting different devices run
# I/O in parallel. The locks are reentrant: a caller may hold one across a
# session-active check that re-acquires the same lock.
_file_locks_guard = threading.Lock()
_file_locks: dict[int, threading.RLock] = {}


def _device_lock(device_id: int) -> threading.RLock:
    # The guard only protects the _file_locks map, and it is released before the
    # device lock is returned, so it never sits on the I/O path and no cross-device
    # lock-ordering cycle can form.
    with _file_locks_guard:
        lock = _file_locks.get(device_id)
        if lock is None:
            lock = threading.RLock()
            _file_locks[device_id] = lock
        return lock


# Live FgdsFile count per device. fgds_open on the first, fgds_close on the last.
# Each device's entry is guarded by that device's lock.
_fgds_file_count: dict[int, int] = {}
# Buffers registered per device, as (data_ptr, nbytes). libfgds accepts a
# duplicate registration and adds a separate kernel mapping for each call, so
# track them here and reject duplicates. The same bookkeeping lets a closing
# session release whatever is still registered (see _deregister_remaining).
_fgds_registered: dict[int, set[tuple[int, int]]] = {}


def _fgds_file_constructed(device_id: int) -> None:
    # libfgds selects the device explicitly (g_dev_ctx[device_id]) and never reads
    # the current CUDA device, so no torch.cuda.device() guard is needed here or
    # around the other entry points; this also matches the C++ GdsFile path.
    bind = _load_fgds_bind()
    with _device_lock(device_id):
        count = _fgds_file_count.get(device_id, 0)
        if count == 0:
            ret = bind.fgds_open(device_id)
            if ret != 0:
                raise RuntimeError(
                    f"fgds_open failed for device {device_id} with return code "
                    f"{ret}{_errno_detail()}"
                )
        _fgds_file_count[device_id] = count + 1


def _deregister_remaining(bind: Any, device_id: int) -> None:
    # Release the registrations a closing session still holds. fgds_close only frees
    # libfgds's own bookkeeping: it neither munmaps the host mappings nor issues
    # FGDS_IOCTL_UNMAP, and the fgdsfs char device's release handler is a no-op, so
    # anything left registered here keeps the GPU pages pinned until the fgdsfs
    # module is unloaded -- even process exit would not release them. Best effort: the bookkeeping is dropped either way (a failed
    # deregister must not keep the session alive), and the fgds_deregmem return code
    # is deliberately ignored -- ctypes returns a non-zero code here instead of
    # raising, and a failed unmapping must not abort fgds_close.
    for ptr, nbytes in sorted(_fgds_registered.pop(device_id, set())):
        try:
            bind.fgds_deregmem(
                ctypes.c_int(device_id), ctypes.c_void_p(ptr), ctypes.c_size_t(nbytes)
            )
        except Exception:
            pass


def _fgds_file_destroyed(device_id: int) -> None:
    # Must stay CUDA-context free: this runs from close()/__del__, possibly at
    # shutdown. Check the count before touching the library: _fgds_shutdown (atexit)
    # drains sessions while libfgds is still loaded, and a later __del__ during
    # finalization must not call into a possibly-unloaded library.
    with _device_lock(device_id):
        count = _fgds_file_count.get(device_id, 0)
        if count <= 0:
            return
        count -= 1
        if count == 0:
            bind = _load_fgds_bind()
            _fgds_file_count.pop(device_id, None)
            _deregister_remaining(bind, device_id)
            if bind.fgds_close(device_id) != 0:
                raise RuntimeError(
                    f"fgds_close failed for device {device_id}{_errno_detail()}"
                )
        else:
            _fgds_file_count[device_id] = count


def _fgds_shutdown() -> None:
    # atexit handler: close any sessions still open while the interpreter and
    # libfgds are alive. Without this, a final FgdsFile.__del__ can run after the
    # CDLL has been finalized and call into unloaded native code.
    # Snapshot the known devices under the guard, then drain each under its own
    # lock. Every device with an open session had a lock created by
    # _fgds_file_constructed, so _file_locks is a superset of the live sessions.
    with _file_locks_guard:
        device_ids = list(_file_locks)
    for device_id in device_ids:
        with _device_lock(device_id):
            if _fgds_file_count.get(device_id, 0) <= 0:
                continue
            _fgds_file_count.pop(device_id, None)
            try:
                bind = _load_fgds_bind()
                # Match _fgds_file_destroyed: release registrations before fgds_close.
                _deregister_remaining(bind, device_id)
                bind.fgds_close(device_id)
            except Exception:
                pass


atexit.register(_fgds_shutdown)


def _fgds_file_session_active(device_id: int) -> bool:
    with _device_lock(device_id):
        return _fgds_file_count.get(device_id, 0) > 0


def _cuda_device_index(storage: Storage) -> int:
    if isinstance(storage, torch.Tensor):
        raise TypeError(
            "fgds requires a torch.UntypedStorage, got a torch.Tensor; pass "
            "tensor.untyped_storage() instead"
        )
    if not isinstance(storage, torch.UntypedStorage):
        raise TypeError(
            f"fgds requires a torch.UntypedStorage, got {type(storage).__name__}"
        )
    device = storage.device
    if device.type != "cuda":
        raise RuntimeError(f"fgds requires CUDA storage, got device {device}")
    index = device.index
    if index is None:
        return torch.cuda.current_device()
    return index


def _check_offset_aligned(offset: int) -> None:
    if offset < 0 or offset % _FGDS_ALIGNMENT != 0:
        raise RuntimeError(
            f"offset must be a non-negative multiple of {_FGDS_ALIGNMENT} bytes "
            f"(O_DIRECT), got {offset}"
        )


def _check_registered(device_id: int, storage: Storage) -> None:
    ptr, nbytes = storage.data_ptr(), storage.nbytes()
    if (ptr, nbytes) not in _fgds_registered.get(device_id, set()):
        raise RuntimeError(
            f"storage (data_ptr={ptr:#x}, nbytes={nbytes}) is not registered for "
            "this device; pass the exact storage previously given to "
            "fgds_register_buffer."
        )


def fgds_register_buffer(s: Storage) -> None:
    """Registers a CUDA storage as an FGDS buffer via ``fgds_regmem``.

    The storage length must be a positive multiple of 64 KiB and its address must
    be 64 KiB-aligned. An active :class:`FgdsFile` for the same device must exist
    first, because ``fgds_regmem`` requires the device session opened by
    ``fgds_open``.

    Registrations track the exact ``(data_ptr, nbytes)`` seen here and do not keep
    the storage alive: do not resize or free the storage while it is registered,
    and pass the same storage to ``load_storage``/``save_storage``/
    :func:`fgds_deregister_buffer` (a sub-range storage, for example from slicing a
    storage, is rejected). Pair with :func:`fgds_deregister_buffer`; a registration
    still held when the last :class:`FgdsFile` for the device is closed is released
    there. Registering the same storage more than once before deregistering it
    raises a ``RuntimeError`` (libfgds itself accepts the duplicate and maps the
    range twice).

    Example::

        >>> # xdoctest: +SKIP("fgds filesystem and libfgds required")
        >>> import os, tempfile
        >>> src = torch.randn(16384, device="cuda")  # 64 KiB
        >>> fd, path = tempfile.mkstemp()  # must be on an O_DIRECT-capable fs
        >>> os.close(fd)
        >>> with FgdsFile(path, os.O_RDWR, device_id=0) as file:
        ...     fgds_register_buffer(src.untyped_storage())
        ...     fgds_deregister_buffer(src.untyped_storage())

    Args:
        s (Storage): CUDA buffer to register.
    """
    device_id = _cuda_device_index(s)
    nbytes = s.nbytes()
    ptr = s.data_ptr()
    target = ctypes.c_void_p()
    # Hold the device lock across the session check, validation, and fgds_regmem so
    # that a concurrent fgds_close (last FgdsFile for this device closed) cannot
    # tear the session down in between. Operations on other devices are not blocked.
    # Check the session first: without an active session registration cannot
    # succeed, so this error is deterministic even when the buffer also fails the
    # size or alignment checks below.
    with _device_lock(device_id):
        if not _fgds_file_session_active(device_id):
            raise RuntimeError(
                "fgds_register_buffer requires an active FgdsFile for this device: "
                "construct FgdsFile(..., device_id=...) first, then register buffers."
            )
        if nbytes == 0 or nbytes % _FGDS_GPU_PAGE_SIZE != 0:
            raise RuntimeError(
                "fgds_register_buffer requires the storage size to be a positive "
                f"multiple of {_FGDS_GPU_PAGE_SIZE} bytes, but got {nbytes} bytes."
            )
        # fgds_regmem drills down to nvfs_nvidia_p2p_get_pages, which pins the
        # buffer at 64 KiB GPU-page granularity and requires the base address to be
        # page-aligned; libfgds validates the length but not the address, so reject
        # a misaligned base here rather than relying on the nvidia-p2p driver to
        # handle it. (The torch caching allocator only guarantees 512 B alignment.)
        if ptr % _FGDS_GPU_PAGE_SIZE != 0:
            raise RuntimeError(
                f"fgds_register_buffer requires the storage address to be aligned to "
                f"{_FGDS_GPU_PAGE_SIZE} bytes, but data_ptr()={ptr:#x} is not."
            )
        registered = _fgds_registered.setdefault(device_id, set())
        if (ptr, nbytes) in registered:
            raise RuntimeError(
                "fgds_register_buffer: this storage is already registered for "
                f"device {device_id}; deregister it first."
            )
        bind = _load_fgds_bind()
        ret = bind.fgds_regmem(
            ctypes.c_int(device_id),
            ctypes.c_void_p(ptr),
            ctypes.c_size_t(nbytes),
            ctypes.byref(target),
        )
        if ret != 0:
            raise RuntimeError(
                f"fgds_regmem failed with return code {ret}{_errno_detail()}"
            )
        registered.add((ptr, nbytes))


def fgds_deregister_buffer(s: Storage) -> None:
    """Deregisters a CUDA storage previously registered with :func:`fgds_register_buffer`.

    This wraps ``fgds_deregmem`` and does not close the device session held by
    :class:`FgdsFile`. It must be called while that session is still active: the
    last :class:`FgdsFile` for a device releases whatever is still registered when
    it is closed, so deregistering after that fails. Deregistering a storage that is
    not currently registered for the device raises a ``RuntimeError`` instead of
    silently doing nothing, so a size or address mismatch is reported rather than
    leaving a stale entry.

    Example::

        >>> # xdoctest: +SKIP("fgds filesystem and libfgds required")
        >>> import os, tempfile
        >>> src = torch.randn(16384, device="cuda")  # 64 KiB
        >>> fd, path = tempfile.mkstemp()  # must be on an O_DIRECT-capable fs
        >>> os.close(fd)
        >>> with FgdsFile(path, os.O_RDWR, device_id=0) as file:
        ...     fgds_register_buffer(src.untyped_storage())
        ...     fgds_deregister_buffer(src.untyped_storage())

    Args:
        s (Storage): CUDA buffer to deregister.
    """
    device_id = _cuda_device_index(s)
    nbytes = s.nbytes()
    ptr = s.data_ptr()
    with _device_lock(device_id):
        if not _fgds_file_session_active(device_id):
            raise RuntimeError(
                "fgds_deregister_buffer requires an active FgdsFile for this device: "
                "registrations are released when the last FgdsFile for the device is "
                "closed, so deregister before closing it."
            )
        registered = _fgds_registered.get(device_id, set())
        if (ptr, nbytes) not in registered:
            raise RuntimeError(
                f"storage (data_ptr={ptr:#x}, nbytes={nbytes}) is not registered "
                "for this device; nothing to deregister."
            )
        bind = _load_fgds_bind()
        ret = bind.fgds_deregmem(
            ctypes.c_int(device_id),
            ctypes.c_void_p(ptr),
            ctypes.c_size_t(nbytes),
        )
        if ret != 0:
            raise RuntimeError(
                f"fgds_deregmem failed with return code {ret}{_errno_detail()}"
            )
        registered.discard((ptr, nbytes))


class FgdsFile:
    r"""Wrapper around an OS file descriptor used for FGDS direct I/O.

    ``fgds_open(device_id)`` runs when the first :class:`FgdsFile` for a device is
    constructed, and ``fgds_close(device_id)`` when the last one is closed. Prefer
    :meth:`close` (or the object as a context manager) over relying on garbage
    collection, so the session is released at a known point; ``__del__`` and an
    ``atexit`` handler remain as fallbacks. Sessions are per-device, so
    :class:`FgdsFile` objects for different ``device_id`` values can coexist in one
    process and are independent of each other.

    Buffers must be registered with :func:`fgds_register_buffer` after this object is
    created: unlike the cuFile path, the FGDS transfer APIs look up host mappings
    created by ``fgds_regmem``, so every storage passed to ``load_storage``/
    ``save_storage`` must be registered first. Deregister them with
    :func:`fgds_deregister_buffer`; whatever is still registered when the last
    :class:`FgdsFile` for the device is closed is released there, since ``fgds_close``
    would otherwise leave those kernel mappings behind.

    Operations on a single device are serialized by a per-device lock, so the APIs
    are safe to call from multiple threads but concurrent transfers on the same
    device do not overlap; transfers and registration for different devices are
    independent and may run in parallel.

    Compared to :class:`torch.cuda.gds.GdsFile`, the constructor takes an additional
    ``device_id`` argument used for ``fgds_open`` and to identify the file in each
    transfer.

    Args:
        filename (str): Name of the file to open.
        flags (int): Flags to pass to ``os.open``. ``os.O_DIRECT`` is added
            automatically when available.
        device_id (int): CUDA device index for the FGDS session and file handle.

    Example::

        >>> # xdoctest: +SKIP("fgds filesystem and libfgds required")
        >>> import os, tempfile
        >>> src = torch.randn(16384, device="cuda")  # 64 KiB
        >>> dest = torch.empty_like(src)
        >>> fd, path = tempfile.mkstemp()  # must be on an O_DIRECT-capable fs
        >>> os.close(fd)
        >>> with FgdsFile(path, os.O_RDWR, device_id=0) as file:
        ...     fgds_register_buffer(src.untyped_storage())
        ...     fgds_register_buffer(dest.untyped_storage())
        ...     file.save_storage(src.untyped_storage(), offset=0)
        ...     file.load_storage(dest.untyped_storage(), offset=0)
        ...     fgds_deregister_buffer(dest.untyped_storage())
        ...     fgds_deregister_buffer(src.untyped_storage())
        ...     torch.equal(src, dest)
        True
    """

    def __init__(self, filename: str, flags: int, device_id: int) -> None:
        # Initialize every attribute before the platform check so that __del__
        # on a partially constructed object (as happens when __init__ raises)
        # does not print an "Exception ignored in __del__" traceback.
        self.filename = filename
        self.flags = flags
        self.device_id = 0
        self.fd: int | None = None
        self._fid: Any = None
        self._session_held = False

        if isinstance(device_id, bool) or not isinstance(device_id, int):
            raise TypeError(f"device_id must be an int, got {type(device_id).__name__}")
        self.device_id = device_id

        if sys.platform == "win32":
            raise RuntimeError("FgdsFile is not supported on this platform.")

        # Reject ids outside the visible CUDA devices before probing libfgds or
        # creating the file, so the error is clear and no empty file is left
        # behind. (libfgds itself returns EINVAL for ids outside its range.)
        n_devices = torch.cuda.device_count()
        if not 0 <= self.device_id < n_devices:
            raise RuntimeError(
                f"FgdsFile device_id must be in [0, {n_devices}), got {self.device_id}"
            )

        # Probe availability before opening the file: with O_CREAT, os.open would
        # otherwise leave an empty file behind when libfgds is missing.
        _load_fgds_bind()

        self.fd = os.open(filename, flags | getattr(os, "O_DIRECT", 0))
        try:
            _fgds_file_constructed(self.device_id)
            self._session_held = True
            self._fid = _fgds_fileid_t(self.fd, self.device_id)
        except BaseException:
            try:
                self.close()
            except Exception:
                # Do not let a failed cleanup mask the original exception.
                pass
            raise

    def close(self) -> None:
        """Closes the file descriptor and releases this file's session reference.

        Idempotent. ``fgds_close`` runs when the last :class:`FgdsFile` for the device
        is closed; when this is that file, buffers still registered for the device are
        deregistered first. Closing explicitly (or using the object as a context
        manager) instead of relying on garbage collection makes that point
        deterministic. Errors from ``fgds_close`` propagate out of this call.

        The whole teardown holds the device lock, so it cannot race a transfer or a
        second ``close()`` on the same device; the lock is reentrant, so the
        ``_fgds_file_destroyed`` call below can reacquire it.
        """
        with _device_lock(self.device_id):
            fd, self.fd = self.fd, None
            held, self._session_held = self._session_held, False
            self._fid = None
            try:
                if fd is not None:
                    os.close(fd)
            finally:
                if held:
                    _fgds_file_destroyed(self.device_id)

    def __enter__(self) -> "FgdsFile":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __del__(self) -> None:
        # Fallback for an object that was never closed; __del__ must never raise.
        try:
            self.close()
        except Exception:
            pass

    def _check_fid(self) -> None:
        if self._fid is None or self.fd is None:
            raise AssertionError("FgdsFile is not open.")

    def _check_storage_device(self, storage: Storage) -> None:
        index = _cuda_device_index(storage)
        if index != self.device_id:
            raise RuntimeError(
                f"Storage is on cuda:{index} but FgdsFile uses device_id={self.device_id}"
            )

    def load_storage(self, storage: Storage, offset: int = 0) -> None:
        """Loads ``storage.nbytes()`` bytes from the file at ``offset`` into ``storage``.

        This is a wrapper around ``fgds_read``. The storage must already be registered
        with :func:`fgds_register_buffer`.

        Args:
            storage (Storage): Storage to load data into.
            offset (int, optional): Offset into the file to start loading from. Must be a
                non-negative multiple of 4096 bytes (O_DIRECT requirement). (Default: 0)
        """
        self._check_storage_device(storage)
        _check_offset_aligned(offset)
        bind = _load_fgds_bind()
        nbytes = storage.nbytes()
        # Hold the device lock across the open/registration checks and fgds_read so a
        # concurrent close() (which takes the same lock) cannot close the fd or tear
        # the session down mid-transfer. Other devices are locked independently and
        # can transfer in parallel.
        with _device_lock(self.device_id):
            self._check_fid()
            _check_registered(self.device_id, storage)
            ret = bind.fgds_read(
                self._fid,
                ctypes.c_void_p(storage.data_ptr()),
                ctypes.c_longlong(0),
                ctypes.c_ssize_t(nbytes),
                ctypes.c_longlong(offset),
            )
            # Read errno while still under the lock and immediately after the call,
            # so a later ctypes call cannot overwrite the thread-local errno.
            errno_detail = _errno_detail() if ret < 0 else ""
        if ret < 0:
            raise RuntimeError(f"fgds_read failed with return code {ret}{errno_detail}")
        if ret != nbytes:
            raise RuntimeError(f"fgds_read expected {nbytes} bytes, got {ret}")

    def save_storage(self, storage: Storage, offset: int = 0) -> None:
        """Saves all bytes of ``storage`` into the file at ``offset``.

        This is a wrapper around ``fgds_write``. The storage must already be registered
        with :func:`fgds_register_buffer`.

        Args:
            storage (Storage): Storage to save data from.
            offset (int, optional): Offset into the file to start saving to. Must be a
                non-negative multiple of 4096 bytes (O_DIRECT requirement). (Default: 0)
        """
        self._check_storage_device(storage)
        _check_offset_aligned(offset)
        bind = _load_fgds_bind()
        nbytes = storage.nbytes()
        # Hold the device lock across the open/registration checks and fgds_write so a
        # concurrent close() (which takes the same lock) cannot close the fd or tear
        # the session down mid-transfer. Other devices are locked independently and
        # can transfer in parallel.
        with _device_lock(self.device_id):
            self._check_fid()
            _check_registered(self.device_id, storage)
            ret = bind.fgds_write(
                self._fid,
                ctypes.c_void_p(storage.data_ptr()),
                ctypes.c_longlong(0),
                ctypes.c_ssize_t(nbytes),
                ctypes.c_longlong(offset),
            )
            # Read errno while still under the lock and immediately after the call,
            # so a later ctypes call cannot overwrite the thread-local errno.
            errno_detail = _errno_detail() if ret < 0 else ""
        if ret < 0:
            raise RuntimeError(
                f"fgds_write failed with return code {ret}{errno_detail}"
            )
        if ret != nbytes:
            raise RuntimeError(f"fgds_write expected {nbytes} bytes, got {ret}")
