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

Standalone POSIX benchmark: torch.save/load and direct-io for multiple rounds.

Run from outside the raw source tree, e.g.:
  cd /tmp && python /path/to/agent_space/bench_posix_pytorch.py
  cd /tmp && python /path/to/agent_space/bench_posix_pytorch.py 1   # use cuda:1
"""

from __future__ import annotations

import mmap
import os
import sys
import time

from common import TEST_NBYTES, TEST_PATH, cuda_tensors_equal_chunked, parse_cuda_cli_args


_DIO_CHUNK_BYTES = 4 * 1024 * 1024


def _print_stats(tag: str, write_times: list[float], read_times: list[float], gib: float) -> None:
    avg_w = sum(write_times) / len(write_times)
    avg_r = sum(read_times) / len(read_times)
    print(
        f"{tag} AVG ({len(write_times)} rounds)  WRITE  latency={avg_w * 1000:.3f} ms, "
        f"bandwidth={gib / avg_w:.3f} GiB/s"
    )
    print(
        f"{tag} AVG ({len(read_times)} rounds)  READ   latency={avg_r * 1000:.3f} ms, "
        f"bandwidth={gib / avg_r:.3f} GiB/s"
    )


def benchmark_torch_save_load(src, path: str, device_id: int, num_rounds: int, gib: float):
    import torch

    write_times: list[float] = []
    read_times: list[float] = []

    for r in range(1, num_rounds + 1):
        t0 = time.perf_counter()
        torch.save(src, path)
        t1 = time.perf_counter()
        w = t1 - t0

        t2 = time.perf_counter()
        dest = torch.load(path, map_location=f"cuda:{device_id}", weights_only=False)
        t3 = time.perf_counter()
        rd = t3 - t2

        write_times.append(w)
        read_times.append(rd)

        print(
            f"[torch.save/load] Round {r}/{num_rounds}  WRITE  latency={w * 1000:.3f} ms, "
            f"bandwidth={gib / w:.3f} GiB/s"
        )
        print(
            f"[torch.save/load] Round {r}/{num_rounds}  READ   latency={rd * 1000:.3f} ms, "
            f"bandwidth={gib / rd:.3f} GiB/s"
        )

    _print_stats("[torch.save/load]", write_times, read_times, gib)
    return dest


def benchmark_direct_io(src, dest, path: str, device_id: int, num_rounds: int, gib: float) -> None:
    import torch

    dio_path = f"{path}.direct"
    fd = os.open(dio_path, os.O_CREAT | os.O_RDWR | os.O_DIRECT, 0o644)
    os.ftruncate(fd, TEST_NBYTES)

    stage_w = torch.empty(_DIO_CHUNK_BYTES, dtype=torch.uint8, pin_memory=True)
    stage_r = torch.empty(_DIO_CHUNK_BYTES, dtype=torch.uint8, pin_memory=True)
    # mmap gives page-aligned backing memory suitable for O_DIRECT buffers.
    io_w = mmap.mmap(-1, _DIO_CHUNK_BYTES, access=mmap.ACCESS_WRITE)
    io_r = mmap.mmap(-1, _DIO_CHUNK_BYTES, access=mmap.ACCESS_WRITE)
    mv_w = memoryview(io_w)
    mv_r = memoryview(io_r)

    write_times: list[float] = []
    read_times: list[float] = []

    try:
        for r in range(1, num_rounds + 1):
            os.lseek(fd, 0, os.SEEK_SET)
            t0 = time.perf_counter()
            for off in range(0, TEST_NBYTES, _DIO_CHUNK_BYTES):
                n = min(_DIO_CHUNK_BYTES, TEST_NBYTES - off)
                cpu_w = stage_w[:n]
                cpu_w.copy_(src[off : off + n], non_blocking=True)
                torch.cuda.synchronize(device_id)
                mv_w[:n] = memoryview(cpu_w.numpy())
                done = 0
                while done < n:
                    wrote = os.writev(fd, [mv_w[done:n]])
                    if wrote <= 0:
                        raise RuntimeError(f"direct write failed at offset {off + done}")
                    done += wrote
            os.fsync(fd)
            t1 = time.perf_counter()
            w = t1 - t0

            os.lseek(fd, 0, os.SEEK_SET)
            t2 = time.perf_counter()
            for off in range(0, TEST_NBYTES, _DIO_CHUNK_BYTES):
                n = min(_DIO_CHUNK_BYTES, TEST_NBYTES - off)
                done = 0
                while done < n:
                    got = os.readv(fd, [mv_r[done:n]])
                    if got <= 0:
                        raise EOFError(f"direct read EOF at offset {off + done}")
                    done += got
                cpu_r = stage_r[:n]
                cpu_r.copy_(torch.frombuffer(mv_r[:n], dtype=torch.uint8), non_blocking=False)
                dest[off : off + n].copy_(cpu_r, non_blocking=True)
            torch.cuda.synchronize(device_id)
            t3 = time.perf_counter()
            rd = t3 - t2

            write_times.append(w)
            read_times.append(rd)

            print(
                f"[direct-io] Round {r}/{num_rounds}  WRITE  latency={w * 1000:.3f} ms, "
                f"bandwidth={gib / w:.3f} GiB/s"
            )
            print(
                f"[direct-io] Round {r}/{num_rounds}  READ   latency={rd * 1000:.3f} ms, "
                f"bandwidth={gib / rd:.3f} GiB/s"
            )
    finally:
        os.close(fd)
        mv_w.release()
        mv_r.release()
        io_w.close()
        io_r.close()

    _print_stats("[direct-io]", write_times, read_times, gib)


def main(argv: list[str] | None = None) -> int:
    import torch

    device_id, num_rounds = parse_cuda_cli_args(
        argv, description="POSIX (torch.save/torch.load) benchmark"
    )

    path = TEST_PATH
    gib = TEST_NBYTES / (1024**3)

    try:
        src = torch.randint(0, 256, (TEST_NBYTES,), dtype=torch.uint8, device=f"cuda:{device_id}")
        dest_direct = torch.empty(TEST_NBYTES, dtype=torch.uint8, device=f"cuda:{device_id}")

        os.truncate(path, TEST_NBYTES)
        dest_cache = benchmark_torch_save_load(src, path, device_id, num_rounds, gib)
        print()
        benchmark_direct_io(src, dest_direct, path, device_id, num_rounds, gib)

        ok_cache = cuda_tensors_equal_chunked(src, dest_cache)
        ok_direct = cuda_tensors_equal_chunked(src, dest_direct)
        if ok_cache and ok_direct:
            print("PASS: posix round-trip read/write matches")
            return 0
        if not ok_cache:
            print("FAIL: torch.save/load tensor mismatch", file=sys.stderr)
        if not ok_direct:
            print("FAIL: direct-io tensor mismatch", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"FAIL: {type(e).__name__}: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
