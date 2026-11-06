#!/usr/bin/env python3
"""Standalone end-to-end validation for torch.cuda.fgds on a machine with libfgds.

This deliberately does not import the PyTorch test framework, so it can run
against an installed torch even when the local repo is a newer commit.

Run from a directory that is NOT the PyTorch repo root, e.g.:

    cd /tmp && FGDS_TMPDIR=/tmp python3 verify_fgds_e2e.py

FGDS_TMPDIR must point at an O_DIRECT-capable local filesystem (ext4/xfs).
Exit code is 0 only if every check passes.
"""

import gc
import os
import sys
import tempfile
import threading

import torch
import torch.cuda.fgds as fgds

FAILS = []


def check(name, cond, extra=""):
    tag = "PASS" if cond else "FAIL"
    print(f"[{tag}] {name}" + (f" :: {extra}" if extra and not cond else ""))
    if not cond:
        FAILS.append(name)


def expect_raise(name, exc, match, fn):
    try:
        fn()
    except exc as e:
        check(name, match in str(e), repr(e))
    except BaseException as e:  # noqa: BLE001
        check(name, False, f"wrong exception {type(e).__name__}: {e}")
    else:
        check(name, False, "no exception raised")


def main():
    print("torch:", torch.__file__)
    if "site-packages" not in torch.__file__:
        print("ERROR: importing a source tree, not the installed torch")
        return 2
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return 2

    if not fgds.is_available():
        print("ERROR: fgds unavailable (libfgds.so not loadable)")
        return 2

    dev = torch.cuda.current_device()
    tmp = os.environ.get("FGDS_TMPDIR") or tempfile.gettempdir()
    print("scratch dir:", tmp, "| device:", dev)

    def rbytes(n):
        return torch.randint(0, 256, (n,), device="cuda", dtype=torch.uint8)

    def ebytes(n):
        return torch.empty(n, device="cuda", dtype=torch.uint8)

    def make_file(name):
        path = os.path.join(tmp, f"fgds_{name}_{os.getpid()}.bin")
        return fgds.FgdsFile(path, os.O_CREAT | os.O_RDWR, device_id=dev), path

    def roundtrip(nbytes):
        src, dst = rbytes(nbytes), ebytes(nbytes)
        f, path = make_file(f"rt{nbytes}")
        try:
            fgds.fgds_register_buffer(src.untyped_storage())
            fgds.fgds_register_buffer(dst.untyped_storage())
            try:
                f.save_storage(src.untyped_storage(), offset=0)
                f.load_storage(dst.untyped_storage(), offset=0)
                return torch.equal(src, dst)
            finally:
                fgds.fgds_deregister_buffer(src.untyped_storage())
                fgds.fgds_deregister_buffer(dst.untyped_storage())
        finally:
            del f
            gc.collect()
            os.unlink(path)

    for mib in (1, 16, 64):
        check(f"roundtrip {mib} MiB exact", roundtrip(mib * 1024 * 1024))

    # --- >1 GiB cross-segment mmap path (off by default; needs ~3 GiB GPU memory) ---
    if os.environ.get("FGDS_RUN_LARGE") == "1":
        check("roundtrip 1.5 GiB exact", roundtrip(int(1.5 * 1024 * 1024 * 1024)))
    else:
        print("[SKIP] 1.5 GiB roundtrip (set FGDS_RUN_LARGE=1 to run)")

    # --- multiple dtypes ---
    for dtype in (torch.float32, torch.float64, torch.int64, torch.uint8):
        numel = 1024 * 1024 // dtype.itemsize
        src = (
            torch.randn(numel, device="cuda", dtype=dtype)
            if dtype.is_floating_point
            else torch.randint(0, 128, (numel,), device="cuda", dtype=dtype)
        )
        dst = torch.empty_like(src)
        f, path = make_file(f"dt{str(dtype).split('.')[-1]}")
        try:
            fgds.fgds_register_buffer(src.untyped_storage())
            fgds.fgds_register_buffer(dst.untyped_storage())
            try:
                f.save_storage(src.untyped_storage())
                f.load_storage(dst.untyped_storage())
                check(f"dtype {dtype} exact", torch.equal(src, dst))
            finally:
                fgds.fgds_deregister_buffer(src.untyped_storage())
                fgds.fgds_deregister_buffer(dst.untyped_storage())
        finally:
            del f
            gc.collect()
            os.unlink(path)

    # --- file offsets ---
    a, b = rbytes(1024 * 1024), rbytes(1024 * 1024)
    da, db = ebytes(1024 * 1024), ebytes(1024 * 1024)
    f, path = make_file("offsets")
    try:
        for t in (a, b, da, db):
            fgds.fgds_register_buffer(t.untyped_storage())
        try:
            f.save_storage(a.untyped_storage(), offset=0)
            f.save_storage(b.untyped_storage(), offset=a.nbytes)
            f.load_storage(da.untyped_storage(), offset=0)
            f.load_storage(db.untyped_storage(), offset=a.nbytes)
            check("offsets at 0 and nbytes exact", torch.equal(a, da) and torch.equal(b, db))
        finally:
            for t in (a, b, da, db):
                fgds.fgds_deregister_buffer(t.untyped_storage())
    finally:
        del f
        gc.collect()
        os.unlink(path)

    # --- persistence across reopen ---
    # The Python layer clears per-device registration tracking when the last
    # FgdsFile for the device is destroyed, so each buffer must be deregistered
    # before that FgdsFile is destroyed.
    src, dst = rbytes(1024 * 1024), ebytes(1024 * 1024)
    path = os.path.join(tmp, f"fgds_persist_{os.getpid()}.bin")
    f = fgds.FgdsFile(path, os.O_CREAT | os.O_RDWR, device_id=dev)
    fgds.fgds_register_buffer(src.untyped_storage())
    f.save_storage(src.untyped_storage())
    fgds.fgds_deregister_buffer(src.untyped_storage())
    del f
    gc.collect()
    f2 = fgds.FgdsFile(path, os.O_RDWR, device_id=dev)
    fgds.fgds_register_buffer(dst.untyped_storage())
    f2.load_storage(dst.untyped_storage())
    check("persist across reopen", torch.equal(src, dst))
    fgds.fgds_deregister_buffer(dst.untyped_storage())
    del f2
    gc.collect()
    os.unlink(path)

    # --- session lifecycle ---
    base = fgds._fgds_file_count.get(dev, 0)
    fa, pa = make_file("sess_a")
    fb, pb = make_file("sess_b")
    check("two files -> count + 2", fgds._fgds_file_count.get(dev, 0) == base + 2)
    del fa
    gc.collect()
    check("one destroyed -> count + 1", fgds._fgds_file_count.get(dev, 0) == base + 1)
    del fb
    gc.collect()
    check("last destroyed -> count base", fgds._fgds_file_count.get(dev, 0) == base)
    os.unlink(pa)
    os.unlink(pb)

    # --- error paths ---
    s = torch.empty(16384, device="cuda").untyped_storage()
    expect_raise(
        "register without FgdsFile",
        RuntimeError,
        "requires an active FgdsFile",
        lambda: fgds.fgds_register_buffer(s),
    )
    expect_raise(
        "deregister without FgdsFile",
        RuntimeError,
        "requires an active FgdsFile",
        lambda: fgds.fgds_deregister_buffer(s),
    )

    # deregister after the last FgdsFile closed the session -> clear Python error
    closed_path = os.path.join(tmp, f"fgds_closed_{os.getpid()}.bin")
    reg = torch.empty(65536, device="cuda", dtype=torch.uint8).untyped_storage()
    fclosed = fgds.FgdsFile(closed_path, os.O_CREAT | os.O_RDWR, device_id=dev)
    fgds.fgds_register_buffer(reg)
    fgds.fgds_deregister_buffer(reg)  # release before close
    del fclosed
    gc.collect()
    expect_raise(
        "deregister after session closed",
        RuntimeError,
        "requires an active FgdsFile",
        lambda: fgds.fgds_deregister_buffer(reg),
    )
    os.unlink(closed_path)

    f, path = make_file("errs")
    bad = torch.empty(1024, device="cuda").untyped_storage()  # 4 KiB, unaligned
    expect_raise(
        "unaligned size", RuntimeError, "multiple of",
        lambda: fgds.fgds_register_buffer(bad),
    )
    empty = torch.empty(0, device="cuda").untyped_storage()
    expect_raise(
        "zero size", RuntimeError, "multiple of",
        lambda: fgds.fgds_register_buffer(empty),
    )
    if torch.cuda.device_count() > 1:
        other_dev = 1 if dev == 0 else 0
        other = torch.empty(16384, device=f"cuda:{other_dev}").untyped_storage()
        expect_raise(
            "wrong device", RuntimeError, "uses device_id",
            lambda: f.load_storage(other),
        )
    # offset must be O_DIRECT aligned (libfgds otherwise returns an opaque -1)
    st = torch.empty(65536, device="cuda", dtype=torch.uint8).untyped_storage()
    expect_raise(
        "save offset alignment", RuntimeError, "multiple of 4096",
        lambda: f.save_storage(st, offset=100),
    )
    expect_raise(
        "load offset alignment", RuntimeError, "multiple of 4096",
        lambda: f.load_storage(st, offset=100),
    )
    del f
    gc.collect()
    os.unlink(path)

    # device_id out of range is rejected by the Python layer before libfgds is
    # even loaded (libfgds itself returns EINVAL for out-of-range ids and never
    # aliases them to device 0)
    devrange_path = os.path.join(tmp, f"fgds_devrange_{os.getpid()}.bin")
    expect_raise(
        "device_id out of range", RuntimeError, "device_id must be in",
        lambda: fgds.FgdsFile(
            devrange_path, os.O_CREAT | os.O_RDWR, device_id=torch.cuda.device_count()
        ),
    )

    # --- concurrency (off by default) ---
    # torch.cuda.fgds serializes each device's session with a per-device lock, so
    # a thread cannot close a session (last FgdsFile destroyed) while another
    # thread still has registrations or I/O in flight on the same device; devices
    # are independent. Kept behind an env var so shared runners are not disturbed.
    if os.environ.get("FGDS_RUN_CONCURRENCY") == "1":
        errs = []

        def worker(_):
            try:
                roundtrip(1024 * 1024)
            except BaseException as e:  # noqa: BLE001
                errs.append(repr(e))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        check("8-thread concurrent roundtrip", not errs, errs[:2])
    else:
        print("[SKIP] 8-thread concurrent roundtrip (set FGDS_RUN_CONCURRENCY=1 to run)")

    # --- regression: normal CUDA still works ---
    a = torch.randn(4096, 4096, device="cuda")
    check("matmul finite", bool(torch.isfinite(a @ a.t()).all()))
    check("alloc/fill/sum", torch.empty(1 << 20, device="cuda").fill_(1).sum().item() == float(1 << 20))

    # --- last: unregistered destination (may abort the process if libfgds is unsafe) ---
    src2, dst2 = rbytes(1024 * 1024), ebytes(1024 * 1024)
    f, path = make_file("unreg")
    fgds.fgds_register_buffer(src2.untyped_storage())
    try:
        f.save_storage(src2.untyped_storage())
        f.load_storage(dst2.untyped_storage())  # dst2 intentionally not registered
        check("unregistered dst raises", False, "no exception raised")
    except RuntimeError:
        check("unregistered dst raises", True)
    except BaseException as e:  # noqa: BLE001
        check("unregistered dst raises", False, repr(e))
    finally:
        del f
        gc.collect()
        if os.path.exists(path):
            os.unlink(path)

    print()
    if FAILS:
        print(f"RESULT: {len(FAILS)} FAILED: {FAILS}")
        return 1
    print("RESULT: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
