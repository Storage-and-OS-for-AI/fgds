#!/usr/bin/env python3
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

Minimal FGDS checkpoint read/write demo.

The demo creates a synthetic state_dict, serializes it with torch.save() to
standard PyTorch checkpoint bytes, writes those bytes with FgdsFile.save_storage(),
reads them back with FgdsFile.load_storage(), and finally validates the result
byte-for-byte and with torch.load().

This is a correctness/example path, not a pure GPU-direct performance path:
torch.save() serializes on the CPU side and FGDS moves the resulting bytes.
For the direct GPU tensor storage I/O path, see
pytorch-fgds/benchmark/bench_fgds_pytorch.py.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import os
import sys
from pathlib import Path
from typing import Any

ALIGN = 64 * 1024
DEFAULT_PATH = "/data/fgds_checkpoint_demo.pt"
DEFAULT_SIZE_MB = 256
DEFAULT_CHUNK_MB = 64
DEFAULT_NUM_TENSORS = 4
DEFAULT_DTYPE = "float16"
DTYPES = ("float16", "float32", "bfloat16")


def _round_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _tensor_to_bytes(tensor: Any) -> bytes:
    """Copy a CPU/GPU uint8 tensor to a Python bytes object."""
    cpu_tensor = tensor.detach().to(device="cpu", non_blocking=False)
    try:
        return cpu_tensor.numpy().tobytes()
    except RuntimeError as exc:
        raise RuntimeError(
            "numpy is required by the checkpoint read path; "
            "install numpy or use a PyTorch build with numpy support"
        ) from exc


def _load_fgds_module() -> Any:
    try:
        from torch.cuda import fgds

        return fgds
    except ImportError:
        # The demo can run directly from the source tree even when fgds_torch.py
        # has not been copied into the PyTorch source tree yet.
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        import fgds_torch as fgds

        return fgds


class AlignedStaging:
    """A 64 KiB-aligned GPU staging storage with a 64 KiB-multiple length.

    The extra allocation guarantees that a suitably aligned sub-storage always
    exists.  Registration and all FGDS I/O use ``storage``; ``view`` is only a
    tensor view used to fill/inspect the bytes.
    """

    def __init__(self, nbytes: int, device: str) -> None:
        import torch

        if nbytes <= 0:
            raise ValueError(f"nbytes must be positive, got {nbytes}")

        nbytes = _round_up(nbytes, ALIGN)
        self.raw = torch.empty(nbytes + ALIGN, dtype=torch.uint8, device=device)
        base_storage = self.raw.untyped_storage()
        offset = (-base_storage.data_ptr()) % ALIGN

        if offset + nbytes > base_storage.nbytes():
            raise RuntimeError(
                "internal error: aligned staging slice exceeds the raw storage"
            )

        self.storage = base_storage[offset : offset + nbytes]
        # as_strided() gives a uint8 tensor view at the same aligned address.
        # It avoids relying on Tensor.set_() accepting an UntypedStorage.
        self.view = torch.as_strided(
            self.raw, (nbytes,), (1,), offset
        )

        if not isinstance(self.storage, torch.UntypedStorage):
            raise RuntimeError(
                "UntypedStorage slicing did not return an UntypedStorage; "
                "this PyTorch version does not support the required storage view"
            )
        if self.storage.data_ptr() != base_storage.data_ptr() + offset:
            raise RuntimeError(
                "UntypedStorage slicing did not preserve the expected address"
            )
        if self.storage.data_ptr() % ALIGN != 0:
            raise RuntimeError(
                f"staging address is not {ALIGN}-byte aligned: "
                f"{self.storage.data_ptr():#x}"
            )
        if self.storage.nbytes() != nbytes:
            raise RuntimeError(
                f"staging length is {self.storage.nbytes()}, expected {nbytes}"
            )
        if self.view.data_ptr() != self.storage.data_ptr():
            raise RuntimeError("staging tensor view does not start at storage base")
        if self.view.numel() != nbytes:
            raise RuntimeError(
                f"staging view has {self.view.numel()} elements, expected {nbytes}"
            )

    @property
    def nbytes(self) -> int:
        return self.storage.nbytes()


def _make_state_dict(
    size_mb: int,
    dtype: Any,
    num_tensors: int,
    device: str,
    seed: int,
) -> dict[str, Any]:
    import torch

    if size_mb < 1:
        raise ValueError("size_mb must be >= 1")
    if num_tensors < 1:
        raise ValueError("num_tensors must be >= 1")

    total_bytes = size_mb * 1024 * 1024
    elem_size = torch.empty((), dtype=dtype).element_size()
    total_elems = total_bytes // elem_size
    if total_elems < num_tensors:
        raise ValueError(
            f"size_mb={size_mb} is too small for num_tensors={num_tensors}"
        )

    base = total_elems // num_tensors
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    state: dict[str, Any] = {}
    for index in range(num_tensors):
        if index < num_tensors - 1:
            count = base
        else:
            count = total_elems - base * (num_tensors - 1)
        # torch.rand() produces values in [0, 1) and never produces NaN/Inf.
        # A distinct generator plus different lengths makes every tensor
        # distinguishable, so byte-order/cross-tensor bugs cannot hide behind
        # identical values.
        state[f"tensor_{index}"] = torch.rand(
            count, dtype=dtype, device=device, generator=generator
        )
    return state


def _fgds_write_bytes(
    fgds: Any,
    path: str,
    data: bytes,
    device_id: int,
    chunk_bytes: int,
) -> dict[str, int]:
    import torch

    if not data:
        raise ValueError("checkpoint payload is empty")
    if chunk_bytes <= 0 or chunk_bytes % ALIGN != 0:
        raise ValueError(f"chunk_bytes must be a positive multiple of {ALIGN}")

    device = f"cuda:{device_id}"
    stats = {"fgds_bytes": 0, "fgds_calls": 0}

    with fgds.FgdsFile(
        path, os.O_CREAT | os.O_RDWR | os.O_TRUNC, device_id=device_id
    ) as f:
        staging = AlignedStaging(chunk_bytes, device)
        # Registration requires an active FgdsFile session for this device.
        fgds.fgds_register_buffer(staging.storage)
        try:
            for offset in range(0, len(data), chunk_bytes):
                count = min(chunk_bytes, len(data) - offset)
                # Keep the bytearray alive while torch.frombuffer() and the
                # H2D copy use its memory.
                host_buffer = bytearray(data[offset : offset + count])
                host_tensor = torch.frombuffer(host_buffer, dtype=torch.uint8)
                staging.view[:count].copy_(host_tensor)
                del host_tensor, host_buffer
                if count < staging.nbytes:
                    staging.view[count:].zero_()

                # FGDS does not know about PyTorch CUDA streams.  The H2D copy
                # above uses the default stream, so synchronize before the
                # host-side FGDS write.
                torch.cuda.synchronize(device_id)
                f.save_storage(staging.storage, offset=offset)
                stats["fgds_bytes"] += staging.nbytes
                stats["fgds_calls"] += 1

            # The last block is written padded to the staging size.  Truncate
            # back to the exact checkpoint length.
            os.ftruncate(f.fd, len(data))
            os.fsync(f.fd)
        finally:
            fgds.fgds_deregister_buffer(staging.storage)

    return stats


def _fgds_read_bytes(
    fgds: Any,
    path: str,
    device_id: int,
    chunk_bytes: int,
) -> tuple[bytearray, dict[str, int]]:
    import torch

    if chunk_bytes <= 0 or chunk_bytes % ALIGN != 0:
        raise ValueError(f"chunk_bytes must be a positive multiple of {ALIGN}")

    total = os.path.getsize(path)
    aligned = total - total % ALIGN
    if aligned == 0:
        raise RuntimeError(
            f"checkpoint is {total} bytes, smaller than {ALIGN}; "
            "FGDS read would not run"
        )

    # Preallocate once; assigning into slices avoids the extra extend()/bytes()
    # copies that would otherwise dominate memory for large checkpoints.
    out = bytearray(total)
    stats = {"fgds_bytes": 0, "fgds_calls": 0, "pread_bytes": 0}
    device = f"cuda:{device_id}"

    with fgds.FgdsFile(path, os.O_RDONLY, device_id=device_id) as f:
        main_size = min(chunk_bytes, aligned)
        staging = AlignedStaging(main_size, device)
        fgds.fgds_register_buffer(staging.storage)
        try:
            offset = 0
            while offset + main_size <= aligned:
                f.load_storage(staging.storage, offset=offset)
                torch.cuda.synchronize(device_id)
                out[offset : offset + main_size] = _tensor_to_bytes(
                    staging.view[:main_size]
                )
                stats["fgds_bytes"] += main_size
                stats["fgds_calls"] += 1
                offset += main_size

            if offset < aligned:
                # The remaining full 64 KiB block is smaller than one chunk.
                # Give it its own staging buffer so the read never crosses EOF.
                count = aligned - offset
                tail = AlignedStaging(count, device)
                fgds.fgds_register_buffer(tail.storage)
                try:
                    f.load_storage(tail.storage, offset=offset)
                    torch.cuda.synchronize(device_id)
                    out[offset : offset + count] = _tensor_to_bytes(
                        tail.view[:count]
                    )
                    stats["fgds_bytes"] += count
                    stats["fgds_calls"] += 1
                finally:
                    fgds.fgds_deregister_buffer(tail.storage)
        finally:
            fgds.fgds_deregister_buffer(staging.storage)

    if aligned < total:
        # The final <64 KiB tail cannot be registered as an FGDS buffer without
        # reading past EOF.  Read it with a normal pread().
        fd = os.open(path, os.O_RDONLY)
        try:
            tail_data = os.pread(fd, total - aligned, aligned)
        finally:
            os.close(fd)
        if len(tail_data) != total - aligned:
            raise RuntimeError(
                f"short pread: got {len(tail_data)} bytes, "
                f"expected {total - aligned}"
            )
        out[aligned:total] = tail_data
        stats["pread_bytes"] = len(tail_data)

    if stats["fgds_bytes"] + stats["pread_bytes"] != total:
        raise RuntimeError(
            "internal error: bytes read via FGDS + pread do not match file size"
        )
    expected_calls = (aligned + chunk_bytes - 1) // chunk_bytes
    if stats["fgds_calls"] != expected_calls:
        raise RuntimeError(
            "internal error: expected "
            f"{expected_calls} FGDS reads, got {stats['fgds_calls']}"
        )

    return out, stats


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write and read a standard PyTorch checkpoint with FGDS.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--path",
        default=DEFAULT_PATH,
        help="checkpoint path; must be on an O_DIRECT-capable filesystem",
    )
    parser.add_argument(
        "--size-mb",
        type=int,
        default=DEFAULT_SIZE_MB,
        help="approximate state_dict size in MiB",
    )
    parser.add_argument(
        "--dtype",
        choices=DTYPES,
        default=DEFAULT_DTYPE,
        help="tensor dtype",
    )
    parser.add_argument(
        "--device-id",
        type=int,
        default=0,
        help="CUDA device index used for FgdsFile and GPU allocations",
    )
    parser.add_argument(
        "--chunk-mb",
        type=int,
        default=DEFAULT_CHUNK_MB,
        help="FGDS transfer chunk size in MiB (multiple of 64 KiB)",
    )
    parser.add_argument(
        "--num-tensors",
        type=int,
        default=DEFAULT_NUM_TENSORS,
        help="number of tensors in the synthetic state_dict",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for reproducible tensor contents",
    )
    return parser


def _run(args: argparse.Namespace) -> int:
    if importlib.util.find_spec("numpy") is None:
        print(
            "FAIL: numpy is required by the checkpoint read path",
            file=sys.stderr,
        )
        return 1

    import torch

    if args.size_mb < 1:
        print("FAIL: --size-mb must be >= 1", file=sys.stderr)
        return 1
    if args.chunk_mb < 1:
        print("FAIL: --chunk-mb must be >= 1", file=sys.stderr)
        return 1
    if args.num_tensors < 1:
        print("FAIL: --num-tensors must be >= 1", file=sys.stderr)
        return 1

    chunk_bytes = args.chunk_mb * 1024 * 1024
    if chunk_bytes % ALIGN != 0:
        print(
            f"FAIL: --chunk-mb must produce a multiple of {ALIGN} bytes",
            file=sys.stderr,
        )
        return 1

    path = os.path.abspath(os.path.expanduser(args.path))
    parent = os.path.dirname(path)
    if not os.path.isdir(parent):
        print(
            f"FAIL: parent directory does not exist: {parent}",
            file=sys.stderr,
        )
        return 1
    if os.path.isdir(path):
        print(f"FAIL: checkpoint path is a directory: {path}", file=sys.stderr)
        return 1

    if not torch.cuda.is_available():
        print("FAIL: CUDA is not available", file=sys.stderr)
        return 1

    device_count = torch.cuda.device_count()
    if args.device_id < 0 or args.device_id >= device_count:
        print(
            f"FAIL: --device-id must be in [0, {device_count}), "
            f"got {args.device_id}",
            file=sys.stderr,
        )
        return 1

    fgds = _load_fgds_module()
    if not fgds.is_available():
        print(
            "FAIL: FGDS is not available; check libfgds.so, LD_LIBRARY_PATH, "
            "the fgdsfs kernel module, and the target filesystem",
            file=sys.stderr,
        )
        return 1

    print("note: --path must be on an O_DIRECT-capable filesystem (not tmpfs)")
    device = f"cuda:{args.device_id}"
    dtype = getattr(torch, args.dtype)

    state_dict = _make_state_dict(
        args.size_mb,
        dtype,
        args.num_tensors,
        device,
        args.seed,
    )

    # Make sure the random GPU tensors are fully materialized before torch.save
    # reads their storages for serialization.
    torch.cuda.synchronize(args.device_id)
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    payload = buffer.getvalue()
    del buffer

    print(f"checkpoint tensors: {len(state_dict)}, dtype={args.dtype}")
    print(f"checkpoint payload: {len(payload)} bytes")

    write_stats = _fgds_write_bytes(
        fgds, path, payload, args.device_id, chunk_bytes
    )
    file_bytes = os.path.getsize(path)
    print(f"checkpoint file: {path} ({file_bytes} bytes)")
    print(
        f"FGDS write: {write_stats['fgds_bytes']} B in "
        f"{write_stats['fgds_calls']} calls"
    )
    if file_bytes != len(payload):
        raise RuntimeError(
            f"file size mismatch: wrote {file_bytes} bytes, "
            f"payload is {len(payload)} bytes"
        )
    expected_write_calls = (len(payload) + chunk_bytes - 1) // chunk_bytes
    if write_stats["fgds_calls"] != expected_write_calls:
        raise RuntimeError(
            "internal error: expected "
            f"{expected_write_calls} FGDS writes, got {write_stats['fgds_calls']}"
        )
    if write_stats["fgds_bytes"] < len(payload):
        raise RuntimeError("internal error: checkpoint bytes did not go through FGDS")

    read_back, read_stats = _fgds_read_bytes(
        fgds, path, args.device_id, chunk_bytes
    )
    print(
        f"FGDS read:  {read_stats['fgds_bytes']} B in "
        f"{read_stats['fgds_calls']} calls"
    )
    print(f"pread tail: {read_stats['pread_bytes']} B")

    if read_back != payload:
        raise RuntimeError("byte mismatch: FGDS read-back != torch.save payload")

    print("sha256 src:", hashlib.sha256(payload).hexdigest()[:16])
    print("sha256 dst:", hashlib.sha256(read_back).hexdigest()[:16])
    print("byte compare: OK")

    loaded = torch.load(
        io.BytesIO(read_back),
        map_location=device,
        weights_only=True,
    )
    if set(loaded.keys()) != set(state_dict.keys()):
        raise RuntimeError("torch.load() returned different tensor names")

    for name, expected in state_dict.items():
        actual = loaded[name]
        if actual.dtype != expected.dtype or actual.shape != expected.shape:
            raise RuntimeError(f"metadata mismatch: {name}")
        if not torch.equal(actual, expected):
            raise RuntimeError(f"content mismatch: {name}")

    expected_keys = set(state_dict.keys())
    del read_back, loaded, payload, state_dict

    # Also verify that the file written by FGDS opens directly with torch.load.
    loaded_from_path = torch.load(
        path, map_location="cpu", weights_only=True
    )
    if set(loaded_from_path.keys()) != expected_keys:
        raise RuntimeError("direct torch.load() returned different tensor names")
    del loaded_from_path

    print("torch.load: OK")
    print("PASS: FGDS checkpoint round-trip matches")
    print(
        "note: this demo serializes on CPU; for the pure GPU-direct performance "
        "path see pytorch-fgds/benchmark/bench_fgds_pytorch.py"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    if sys.platform == "win32":
        print("SKIP: FgdsFile is not supported on Windows", file=sys.stderr)
        return 1

    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return _run(args)
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
