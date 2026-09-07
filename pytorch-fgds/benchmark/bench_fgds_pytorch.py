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

Standalone FGDS benchmark: torch.cuda.fgds write/read for multiple rounds.

Run from outside the raw source tree, e.g.:
  cd /tmp && python /path/to/agent_space/bench_fgds_pytorch.py
  cd /tmp && python /path/to/agent_space/bench_fgds_pytorch.py 1   # use cuda:1
"""

from __future__ import annotations

import os
import sys
import time

from common import TEST_NBYTES, TEST_PATH, cuda_tensors_equal_chunked, parse_cuda_cli_args


def main(argv: list[str] | None = None) -> int:
    import torch
    from torch.cuda import fgds

    if sys.platform == "win32":
        print("SKIP: FgdsFile is not supported on Windows", file=sys.stderr)
        return 0

    device_id, num_rounds = parse_cuda_cli_args(argv, description="FGDS read/write benchmark")
    path = TEST_PATH
    gib = TEST_NBYTES / (1024**3)

    try:
        src = torch.randint(0, 256, (TEST_NBYTES,), dtype=torch.uint8, device=f"cuda:{device_id}")
        dest = torch.empty(TEST_NBYTES, dtype=torch.uint8, device=f"cuda:{device_id}")

        os.truncate(path, TEST_NBYTES)

        fh = fgds.FgdsFile(path, os.O_CREAT | os.O_RDWR, device_id=device_id)
        try:
            fgds.fgds_register_buffer(src.untyped_storage())
            fgds.fgds_register_buffer(dest.untyped_storage())
            print(
                f"Registered buffers: {src.nbytes} bytes ({gib:.1f} GiB) on cuda:{device_id}"
            )

            write_times: list[float] = []
            read_times: list[float] = []

            for r in range(1, num_rounds + 1):
                t0 = time.perf_counter()
                fh.save_storage(src.untyped_storage(), offset=0)
                t1 = time.perf_counter()
                w = t1 - t0

                t2 = time.perf_counter()
                fh.load_storage(dest.untyped_storage(), offset=0)
                t3 = time.perf_counter()
                rd = t3 - t2

                write_times.append(w)
                read_times.append(rd)

                print(
                    f"Round {r}/{num_rounds}  WRITE  latency={w * 1000:.3f} ms, "
                    f"bandwidth={gib / w:.3f} GiB/s"
                )
                print(
                    f"Round {r}/{num_rounds}  READ   latency={rd * 1000:.3f} ms, "
                    f"bandwidth={gib / rd:.3f} GiB/s"
                )

            avg_w = sum(write_times) / len(write_times)
            avg_r = sum(read_times) / len(read_times)
            print(
                f"AVG ({num_rounds} rounds)  WRITE  latency={avg_w * 1000:.3f} ms, "
                f"bandwidth={gib / avg_w:.3f} GiB/s"
            )
            print(
                f"AVG ({num_rounds} rounds)  READ   latency={avg_r * 1000:.3f} ms, "
                f"bandwidth={gib / avg_r:.3f} GiB/s"
            )

            fgds.fgds_deregister_buffer(src.untyped_storage())
            fgds.fgds_deregister_buffer(dest.untyped_storage())
            print("deregister_buffer done")
        finally:
            del fh

        ok = cuda_tensors_equal_chunked(src, dest)
        if ok:
            print("PASS: fgds round-trip read/write matches")
            return 0
        print("FAIL: tensor mismatch", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"FAIL: {type(e).__name__}: {e}", file=sys.stderr)
        raise
    finally:
        pass


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
