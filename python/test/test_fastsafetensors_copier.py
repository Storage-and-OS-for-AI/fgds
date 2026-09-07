#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import time

import torch

from fastsafetensors import fastsafe_open


def test_loading_time(loader_name, filenames, device, use_fgds=False):
    print(f"Testing {loader_name}...")
    start_time = time.time()

    with fastsafe_open(filenames=filenames, device=device, use_fgds=use_fgds) as f:
        keys = f.keys()
        print(f"Found {len(keys)} tensors")

        tensors = {}
        for key in keys:
            tensor = f.get_tensor(key)
            tensors[key] = tensor

    end_time = time.time()
    loading_time = end_time - start_time
    print(f"{loader_name} loading time: {loading_time:.4f} seconds")
    return loading_time


def collect_safetensor_files(directory):
    items = os.listdir(directory)
    filenames = [
        os.path.join(directory, f)
        for f in sorted(items)
        if f.endswith(".safetensors")
    ]
    for filename in filenames:
        if not os.path.exists(filename):
            print(f"File not found: {filename}")
            raise FileNotFoundError(filename)
    return filenames


def main():
    parser = argparse.ArgumentParser(description="FGDS/GDS loading benchmark")
    parser.add_argument(
        "--model-dir",
        "-d",
        action="append",
        default=[],
        help="Model directory containing .safetensors files (can be repeated)",
    )
    parser.add_argument(
        "--use-fgds",
        action="store_true",
        default=False,
        help="Use FGDS (Fast GPU Direct Storage) copier instead of GDS",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        default=False,
        help="Run both GDS and FGDS and compare times",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="CUDA device to use (default: cuda:0)",
    )
    parser.add_argument(
        "--drop-cache",
        action="store_true",
        default=False,
        help="Drop filesystem cache before each run (requires root /proc/sys/vm/drop_caches)",
    )
    args = parser.parse_args()

    model_dirs = args.model_dir or ["/data/Qwen-32B"]

    for edir in model_dirs:
        if args.drop_cache:
            time.sleep(2)
            os.system("sync && echo 3 > /proc/sys/vm/drop_caches")

        print(f"\n{'=' * 60}")
        print(f"Model directory: {edir}")
        print(f"{'=' * 60}")

        filenames = collect_safetensor_files(edir)
        print(f"Found {len(filenames)} safetensors files")

        if not torch.cuda.is_available():
            print("GPU not available, skipping GPU test.")
            continue

        device = args.device
        print(f"Using device: {device}")

        if args.compare:
            gds_time = test_loading_time("GDS", filenames, device, use_fgds=False)
            if args.drop_cache:
                time.sleep(2)
                os.system("sync && echo 3 > /proc/sys/vm/drop_caches")
            fgds_time = test_loading_time("FGDS", filenames, device, use_fgds=True)

            print("\n" + "=" * 60)
            print("Performance comparison:")
            print(f"  GDS:  {gds_time:.4f} seconds")
            print(f"  FGDS: {fgds_time:.4f} seconds")
            if fgds_time < gds_time:
                speedup = gds_time / fgds_time
                print(f"  FGDS is {speedup:.2f}x faster")
            else:
                slowdown = fgds_time / gds_time
                print(f"  FGDS is {slowdown:.2f}x slower")
        else:
            mode = "FGDS" if args.use_fgds else "GDS"
            test_loading_time(mode, filenames, device, use_fgds=args.use_fgds)


if __name__ == "__main__":
    main()
