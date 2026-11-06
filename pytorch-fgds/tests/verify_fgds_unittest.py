#!/usr/bin/env python3
"""Standalone unittest runner for the `TestFGDS` logic.

Unlike `test/test_cuda.py`, this imports only from `torch.testing._internal.
common_utils`, which is stable across the installed torch and this repo. That
lets it run against an installed torch without triggering the newer
`common_cuda` imports (e.g. `BF16X9_API_SUPPORTED`).

Run from a directory that is NOT the PyTorch repo root:

    cd /tmp && python3 verify_fgds_unittest.py -v

FGDS is skipped automatically when libfgds.so or an O_DIRECT filesystem is
unavailable, so this is also safe to run on machines without FGDS hardware.
"""

import os
import sys
import tempfile
import threading
import unittest

import psutil
import torch

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TemporaryFileName,
    TestCase,
)


_TEST_CUDA = torch.cuda.is_available()


@unittest.skipIf(not _TEST_CUDA, "CUDA not available, skipping tests")
class TestFGDS(TestCase):
    # fgds_regmem requires 64 KiB-multiple lengths, and O_DIRECT requires
    # aligned transfers, so use 1 MiB buffers.
    _BYTES = 1024 * 1024

    def _get_tmp_dir_fs_type(self):
        my_path = os.path.realpath(tempfile.gettempdir())
        root_type = ""
        for part in psutil.disk_partitions():
            if part.mountpoint == "/":
                root_type = part.fstype
                continue
            if part.mountpoint == my_path:
                return part.fstype
        return root_type

    def _require_fgds(self):
        if sys.platform != "linux":
            self.skipTest("FGDS requires Linux")
        if not torch.cuda.fgds.is_available():
            self.skipTest("libfgds.so is not loadable")
        # is_available() only proves libfgds.so loads, not that the fgds kernel
        # module is loaded; without it fgds_open fails and every test errors. The
        # kernel can also load while omitting a node (BAR remap failure or a sparse
        # gpuids selection), so check the node the tests will actually use.
        if not os.path.exists(f"/dev/fgds_dev{torch.cuda.current_device()}"):
            self.skipTest("fgds kernel module is not loaded for the current device")
        if self._get_tmp_dir_fs_type() not in ("ext4", "xfs"):
            self.skipTest("FGDS requires ext4/xfs for local filesystem")

    def _new_buffer(self, dtype):
        numel = self._BYTES // dtype.itemsize
        if dtype.is_floating_point:
            return torch.randn(numel, device="cuda", dtype=dtype)
        return torch.randint(0, 128, (numel,), device="cuda", dtype=dtype)

    def test_fgds_import_succeeds_without_libfgds(self):
        # Importing must succeed and is_available() must return a bool even when
        # libfgds.so is absent (R1). Do not reload the module here: reload would
        # reset the session/registration state tracked by torch.cuda.fgds.
        import torch.cuda.fgds as fgds_module

        self.assertIs(fgds_module, torch.cuda.fgds)
        self.assertIsInstance(fgds_module.is_available(), bool)

    def test_fgds_is_available(self):
        self.assertIsInstance(torch.cuda.fgds.is_available(), bool)

    def test_fgds_register_buffer_requires_fgds_file(self):
        storage = torch.empty(16384, device="cuda").untyped_storage()
        with self.assertRaisesRegex(RuntimeError, "requires an active FgdsFile"):
            torch.cuda.fgds.fgds_register_buffer(storage)

    def test_fgds_deregister_buffer_requires_fgds_file(self):
        storage = torch.empty(16384, device="cuda").untyped_storage()
        with self.assertRaisesRegex(RuntimeError, "requires an active FgdsFile"):
            torch.cuda.fgds.fgds_deregister_buffer(storage)

    def test_fgds_deregister_buffer_after_session_closed(self):
        self._require_fgds()
        device = torch.cuda.current_device()
        storage = torch.empty(16384, device="cuda").untyped_storage()
        with TemporaryFileName() as f:
            file = torch.cuda.fgds.FgdsFile(f, os.O_CREAT | os.O_RDWR, device_id=device)
            torch.cuda.fgds.fgds_register_buffer(storage)
            torch.cuda.fgds.fgds_deregister_buffer(storage)
            del file
        with self.assertRaisesRegex(RuntimeError, "requires an active FgdsFile"):
            torch.cuda.fgds.fgds_deregister_buffer(storage)

    def test_fgds_register_buffer_length_not_multiple(self):
        # Registration requires the length (not just the address) to be a 64 KiB
        # multiple; a 4 KiB storage must be rejected.
        self._require_fgds()
        device = torch.cuda.current_device()
        storage = torch.empty(1024, device="cuda").untyped_storage()  # 4 KiB
        with TemporaryFileName() as f:
            file = torch.cuda.fgds.FgdsFile(f, os.O_CREAT | os.O_RDWR, device_id=device)
            self.assertIsNotNone(file.fd)
            with self.assertRaisesRegex(RuntimeError, "multiple of"):
                torch.cuda.fgds.fgds_register_buffer(storage)

    def test_fgds_device_id_out_of_range(self):
        with TemporaryFileName() as f:
            with self.assertRaisesRegex(RuntimeError, "device_id must be in"):
                torch.cuda.fgds.FgdsFile(
                    f, os.O_CREAT | os.O_RDWR, device_id=torch.cuda.device_count()
                )

    def test_fgds_offset_alignment(self):
        self._require_fgds()
        device = torch.cuda.current_device()
        storage = torch.empty(16384, device="cuda").untyped_storage()
        with TemporaryFileName() as f:
            file = torch.cuda.fgds.FgdsFile(f, os.O_CREAT | os.O_RDWR, device_id=device)
            with self.assertRaisesRegex(RuntimeError, "multiple of 4096"):
                file.save_storage(storage, offset=100)
            with self.assertRaisesRegex(RuntimeError, "multiple of 4096"):
                file.load_storage(storage, offset=100)
            with self.assertRaisesRegex(RuntimeError, "non-negative multiple of 4096"):
                file.save_storage(storage, offset=-4096)
            with self.assertRaisesRegex(RuntimeError, "non-negative multiple of 4096"):
                file.load_storage(storage, offset=-4096)

    @parametrize("dtype", [torch.float32, torch.float64, torch.int64, torch.uint8])
    def test_fgds_dtypes(self, dtype):
        self._require_fgds()
        device = torch.cuda.current_device()
        src = self._new_buffer(dtype)
        dest = torch.empty_like(src)
        registered = []
        try:
            with TemporaryFileName() as f:
                file = torch.cuda.fgds.FgdsFile(f, os.O_CREAT | os.O_RDWR, device_id=device)
                for tensor in (src, dest):
                    torch.cuda.fgds.fgds_register_buffer(tensor.untyped_storage())
                    registered.append(tensor.untyped_storage())
                file.save_storage(src.untyped_storage(), offset=0)
                file.load_storage(dest.untyped_storage(), offset=0)
        finally:
            for storage in registered:
                torch.cuda.fgds.fgds_deregister_buffer(storage)
        self.assertEqual(src, dest)

    def test_fgds_read_write_offsets(self):
        self._require_fgds()
        device = torch.cuda.current_device()
        src1 = self._new_buffer(torch.float32)
        src2 = self._new_buffer(torch.float32)
        dest1 = torch.empty_like(src1)
        dest2 = torch.empty_like(src2)
        registered = []
        try:
            with TemporaryFileName() as f:
                file = torch.cuda.fgds.FgdsFile(f, os.O_CREAT | os.O_RDWR, device_id=device)
                for tensor in (src1, src2, dest1, dest2):
                    torch.cuda.fgds.fgds_register_buffer(tensor.untyped_storage())
                    registered.append(tensor.untyped_storage())
                file.save_storage(src1.untyped_storage(), offset=0)
                file.save_storage(src2.untyped_storage(), offset=src1.nbytes)
                file.load_storage(dest1.untyped_storage(), offset=0)
                file.load_storage(dest2.untyped_storage(), offset=src1.nbytes)
        finally:
            for storage in registered:
                torch.cuda.fgds.fgds_deregister_buffer(storage)
        self.assertEqual(src1, dest1)
        self.assertEqual(src2, dest2)

    def test_fgds_persist_across_reopen(self):
        self._require_fgds()
        device = torch.cuda.current_device()
        src = self._new_buffer(torch.float32)
        dest = torch.empty_like(src)
        with TemporaryFileName() as f:
            file = torch.cuda.fgds.FgdsFile(f, os.O_CREAT | os.O_RDWR, device_id=device)
            torch.cuda.fgds.fgds_register_buffer(src.untyped_storage())
            file.save_storage(src.untyped_storage())
            # The Python layer clears per-device registration tracking when the
            # last FgdsFile for the device is destroyed; deregister first.
            torch.cuda.fgds.fgds_deregister_buffer(src.untyped_storage())
            del file
            file2 = torch.cuda.fgds.FgdsFile(f, os.O_RDWR, device_id=device)
            torch.cuda.fgds.fgds_register_buffer(dest.untyped_storage())
            file2.load_storage(dest.untyped_storage())
            torch.cuda.fgds.fgds_deregister_buffer(dest.untyped_storage())
            del file2
        self.assertEqual(src, dest)

    def test_fgds_session_lifecycle(self):
        self._require_fgds()
        device = torch.cuda.current_device()
        base = torch.cuda.fgds._fgds_file_count.get(device, 0)
        with TemporaryFileName() as f1, TemporaryFileName() as f2:
            file1 = torch.cuda.fgds.FgdsFile(f1, os.O_CREAT | os.O_RDWR, device_id=device)
            file2 = torch.cuda.fgds.FgdsFile(f2, os.O_CREAT | os.O_RDWR, device_id=device)
            self.assertEqual(torch.cuda.fgds._fgds_file_count.get(device, 0), base + 2)
            del file1
            self.assertEqual(torch.cuda.fgds._fgds_file_count.get(device, 0), base + 1)
            del file2
        self.assertEqual(torch.cuda.fgds._fgds_file_count.get(device, 0), base)

    def test_fgds_multi_device_roundtrip(self):
        # Sessions are per-device: FgdsFile objects for different devices coexist
        # in one process, each with its own registrations and I/O.
        self._require_fgds()
        if torch.cuda.device_count() < 2:
            self.skipTest("requires 2 GPUs")
        # The test uses device_id 0 and 1. The kernel only creates nodes for the
        # GPUs selected at module load and names them by their original index, so
        # with a sparse selection (e.g. gpuids=0,2) node 1 can be absent even when
        # two nodes exist; check the ids actually used rather than the node count.
        missing = [d for d in (0, 1) if not os.path.exists(f"/dev/fgds_dev{d}")]
        if missing:
            self.skipTest(f"FGDS device node(s) missing: {missing}")

        registered = []
        # payloads is defined before the try so the FgdsFile objects (and thus
        # their sessions) stay alive through the finally cleanup below.
        payloads = []
        try:
            with TemporaryFileName() as f0, TemporaryFileName() as f1:
                for device, path in ((0, f0), (1, f1)):
                    file = torch.cuda.fgds.FgdsFile(
                        path, os.O_CREAT | os.O_RDWR, device_id=device
                    )
                    src = torch.randn(
                        self._BYTES // 4, device=f"cuda:{device}", dtype=torch.float32
                    )
                    dest = torch.empty_like(src)
                    for tensor in (src, dest):
                        torch.cuda.fgds.fgds_register_buffer(tensor.untyped_storage())
                        registered.append(tensor.untyped_storage())
                    payloads.append((file, src, dest))

                for file, src, dest in payloads:
                    file.save_storage(src.untyped_storage(), offset=0)
                    file.load_storage(dest.untyped_storage(), offset=0)
                    self.assertEqual(src, dest)

                # Deregister on both devices while both sessions are still alive.
                # Remove each entry once deregistered so the finally block only
                # retries the ones that actually failed.
                for storage in list(registered):
                    torch.cuda.fgds.fgds_deregister_buffer(storage)
                    registered.remove(storage)
        finally:
            for storage in registered:
                try:
                    torch.cuda.fgds.fgds_deregister_buffer(storage)
                except RuntimeError:
                    # Do not let a failed cleanup mask the original exception.
                    pass

    def test_fgds_file_fails_cleanly_when_unavailable(self):
        # Mirrors the main-repo TestFGDS: without libfgds the constructor must fail
        # with a clear error instead of a raw loader error.
        if torch.cuda.fgds.is_available():
            self.skipTest("libfgds present; failure path differs")
        with TemporaryFileName() as f:
            with self.assertRaisesRegex(RuntimeError, "FGDS is not available"):
                torch.cuda.fgds.FgdsFile(f, os.O_CREAT | os.O_RDWR, device_id=0)

    def test_fgds_failed_construction_leaves_no_file(self):
        # libfgds is probed before os.open(O_CREAT), so a failed construction must
        # not leave an empty file on disk.
        if torch.cuda.fgds.is_available():
            self.skipTest("libfgds present; failure path differs")
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "fgds_leak.bin")
            with self.assertRaisesRegex(RuntimeError, "FGDS is not available"):
                torch.cuda.fgds.FgdsFile(path, os.O_CREAT | os.O_RDWR, device_id=0)
            self.assertFalse(os.path.exists(path))

    def test_fgds_requires_cuda_storage(self):
        # A CPU storage is rejected before libfgds is loaded.
        storage = torch.empty(16384).untyped_storage()
        with self.assertRaisesRegex(RuntimeError, "requires CUDA storage"):
            torch.cuda.fgds.fgds_register_buffer(storage)
        with self.assertRaisesRegex(RuntimeError, "requires CUDA storage"):
            torch.cuda.fgds.fgds_deregister_buffer(storage)

    def test_fgds_rejects_tensor_instead_of_storage(self):
        # Passing a Tensor must give a TypeError that points at untyped_storage().
        tensor = torch.empty(16384, device="cuda")
        with self.assertRaisesRegex(TypeError, "untyped_storage"):
            torch.cuda.fgds.fgds_register_buffer(tensor)
        with self.assertRaisesRegex(TypeError, "untyped_storage"):
            torch.cuda.fgds.fgds_deregister_buffer(tensor)

    def test_fgds_duplicate_register_rejected(self):
        self._require_fgds()
        device = torch.cuda.current_device()
        storage = torch.empty(16384, device="cuda").untyped_storage()
        with TemporaryFileName() as f:
            file = torch.cuda.fgds.FgdsFile(f, os.O_CREAT | os.O_RDWR, device_id=device)
            try:
                torch.cuda.fgds.fgds_register_buffer(storage)
                with self.assertRaisesRegex(RuntimeError, "already registered"):
                    torch.cuda.fgds.fgds_register_buffer(storage)
            finally:
                # file is still alive; deregister needs its session open.
                torch.cuda.fgds.fgds_deregister_buffer(storage)

    def test_fgds_load_save_requires_registered_buffer(self):
        self._require_fgds()
        device = torch.cuda.current_device()
        storage = torch.empty(16384, device="cuda").untyped_storage()
        with TemporaryFileName() as f:
            file = torch.cuda.fgds.FgdsFile(f, os.O_CREAT | os.O_RDWR, device_id=device)
            with self.assertRaisesRegex(RuntimeError, "not registered"):
                file.save_storage(storage)
            with self.assertRaisesRegex(RuntimeError, "not registered"):
                file.load_storage(storage)

    def test_fgds_deregister_unregistered_rejected(self):
        # Deregistering a storage that is not registered (or whose size no longer
        # matches) must fail fast instead of silently doing nothing.
        self._require_fgds()
        device = torch.cuda.current_device()
        storage = torch.empty(16384, device="cuda").untyped_storage()
        with TemporaryFileName() as f:
            file = torch.cuda.fgds.FgdsFile(f, os.O_CREAT | os.O_RDWR, device_id=device)
            self.assertIsNotNone(file.fd)
            with self.assertRaisesRegex(RuntimeError, "not registered"):
                torch.cuda.fgds.fgds_deregister_buffer(storage)

    def test_fgds_storage_device_mismatch(self):
        self._require_fgds()
        if torch.cuda.device_count() < 2:
            self.skipTest("requires 2 GPUs")
        device = torch.cuda.current_device()
        other = 1 if device == 0 else 0
        storage = torch.empty(16384, device=f"cuda:{other}").untyped_storage()
        with TemporaryFileName() as f:
            file = torch.cuda.fgds.FgdsFile(f, os.O_CREAT | os.O_RDWR, device_id=device)
            with self.assertRaisesRegex(RuntimeError, "uses device_id"):
                file.load_storage(storage)
            with self.assertRaisesRegex(RuntimeError, "uses device_id"):
                file.save_storage(storage)

    def test_fgds_read_write_tensors(self):
        self._require_fgds()
        device = torch.cuda.current_device()
        src = self._new_buffer(torch.float32)
        dest = torch.empty_like(src)
        registered = []
        try:
            with TemporaryFileName() as f:
                file = torch.cuda.fgds.FgdsFile(
                    f, os.O_CREAT | os.O_RDWR, device_id=device
                )
                for tensor in (src, dest):
                    torch.cuda.fgds.fgds_register_buffer(tensor.untyped_storage())
                    registered.append(tensor.untyped_storage())
                file.save_storage(src.untyped_storage(), offset=0)
                file.load_storage(dest.untyped_storage(), offset=0)
        finally:
            # Keep the local file alive until here; deregister needs its session open.
            for storage in registered:
                torch.cuda.fgds.fgds_deregister_buffer(storage)
        self.assertEqual(src, dest)

    def test_fgds_small_buffer_roundtrip(self):
        # 64 KiB is the smallest registerable size and exercises the direct
        # (pread/pwrite) path. The 1 MiB buffers in the other tests exercise the
        # io_uring path for reads (>= 512 KiB) while their writes stay on the
        # direct path (< 2 MiB).
        self._require_fgds()
        device = torch.cuda.current_device()
        src = torch.randn(16384, device="cuda", dtype=torch.float32)  # 64 KiB
        dest = torch.empty_like(src)
        registered = []
        try:
            with TemporaryFileName() as f:
                file = torch.cuda.fgds.FgdsFile(
                    f, os.O_CREAT | os.O_RDWR, device_id=device
                )
                for tensor in (src, dest):
                    torch.cuda.fgds.fgds_register_buffer(tensor.untyped_storage())
                    registered.append(tensor.untyped_storage())
                file.save_storage(src.untyped_storage(), offset=0)
                file.load_storage(dest.untyped_storage(), offset=0)
        finally:
            # Keep the local file alive until here; deregister needs its session open.
            for storage in registered:
                torch.cuda.fgds.fgds_deregister_buffer(storage)
        self.assertEqual(src, dest)

    @unittest.skipUnless(
        os.environ.get("FGDS_RUN_CONCURRENCY") == "1",
        "concurrent FGDS stress test; set FGDS_RUN_CONCURRENCY=1 to run",
    )
    def test_fgds_concurrent(self):
        self._require_fgds()
        device = torch.cuda.current_device()
        errors = []

        def worker():
            try:
                src = self._new_buffer(torch.float32)
                dest = torch.empty_like(src)
                with TemporaryFileName() as f:
                    file = torch.cuda.fgds.FgdsFile(f, os.O_CREAT | os.O_RDWR, device_id=device)
                    torch.cuda.fgds.fgds_register_buffer(src.untyped_storage())
                    torch.cuda.fgds.fgds_register_buffer(dest.untyped_storage())
                    try:
                        file.save_storage(src.untyped_storage())
                        file.load_storage(dest.untyped_storage())
                        if not torch.equal(src, dest):
                            errors.append("data mismatch")
                    finally:
                        torch.cuda.fgds.fgds_deregister_buffer(src.untyped_storage())
                        torch.cuda.fgds.fgds_deregister_buffer(dest.untyped_storage())
            except BaseException as e:  # noqa: BLE001
                errors.append(repr(e))

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])


instantiate_parametrized_tests(TestFGDS)

if __name__ == "__main__":
    run_tests()
