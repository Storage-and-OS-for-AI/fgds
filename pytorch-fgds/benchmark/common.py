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

from __future__ import annotations

import argparse
import sys

TEST_PATH = "/data/10GB_ddrand"
TEST_NBYTES = 10 * 1024 * 1024 * 1024

# torch.equal on full N-byte tensors allocates an N-byte bool mask on CUDA; compare in chunks.
_VERIFY_CHUNK_BYTES = 64 * 1024 * 1024


def cuda_tensors_equal_chunked(a, b, chunk_bytes: int = _VERIFY_CHUNK_BYTES) -> bool:
    import torch

    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    flat_a = a.flatten()
    flat_b = b.flatten()
    n = flat_a.numel()
    step = max(1, chunk_bytes // flat_a.element_size())
    for i in range(0, n, step):
        j = min(i + step, n)
        if not torch.equal(flat_a[i:j], flat_b[i:j]):
            return False
    return True


def parse_cuda_cli_args(
    argv: list[str] | None,
    *,
    description: str,
    default_gpu_id: int = 0,
    default_num_rounds: int = 3,
) -> tuple[int, int]:
    """
    Parse CLI args and set the active CUDA device.

    Returns:
        (device_id, num_rounds)
    """
    import torch

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "gpu_id",
        nargs="?",
        type=int,
        default=default_gpu_id,
        metavar="GPU_ID",
        help="CUDA device index (default: %(default)s)",
    )
    parser.add_argument(
        "num_rounds",
        nargs="?",
        type=int,
        default=default_num_rounds,
        metavar="NUM_ROUNDS",
        help="Number of write+read rounds (default: %(default)s)",
    )
    args = parser.parse_args(argv)

    if args.num_rounds <= 0:
        parser.error("NUM_ROUNDS must be a positive integer")

    if not torch.cuda.is_available():
        print("FAIL: CUDA not available", file=sys.stderr)
        raise SystemExit(1)

    n_gpu = torch.cuda.device_count()
    if args.gpu_id < 0 or args.gpu_id >= n_gpu:
        print(
            f"FAIL: invalid gpu_id {args.gpu_id}, need 0 <= gpu_id < {n_gpu}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    torch.cuda.set_device(args.gpu_id)
    return args.gpu_id, args.num_rounds

