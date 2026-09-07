import os
import subprocess
import sys
import re
import time
import shlex

from logger import Log

result_path = ""
FILE_PATH = "/mnt/fgds/test.data"
SUBDIR = "fgds"
MIN_FILE_SIZE_GB = 10
MIN_FILE_SIZE_BYTES = MIN_FILE_SIZE_GB * 1024 * 1024 * 1024
DEFAULT_LENGTH_GB = 10
# 1M 
MB = 1024
io_sizes = [4096]
#io_sizes = [4, 64, 128, 256, 512, 1024, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 24576, 32768, 65536, 131072, 262144, 524288, 1048576]
#threads = [1, 2, 4, 8, 16, 32, 64, 128]
threads = [1]
batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]

read_write = ["read", "write"]
#read_write = ["read"]
file_path = os.path.dirname(os.path.realpath(__file__))
micro_exec = os.path.join(file_path, "..", "build", "bin", "microbenchmark")

class test_config:
    def __init__(self):
        self.multi_size = False
        self.multi_thread = False
        self.multi_batch = False
        self.async_mode = 0
        self.xfer_mode = 0
        self.device_id = 0
        self.length_gb = DEFAULT_LENGTH_GB
    
    def reset(self):
        self.multi_size = False
        self.multi_thread = False
        self.multi_batch = False

pattern = r"(?:Average IO bandwidth|Average IO latency|95th percentile latency|99th percentile latency|99.9th percentile latency):\s*([\d.]+)"

def validate_file_size(path):
    if not os.path.isfile(path):
        Log.error(f"Test file not found: {path}")
        sys.exit(1)
    file_size = os.path.getsize(path)
    if file_size < MIN_FILE_SIZE_BYTES: # 要求所使用的测试文件大小至少为10GB
        file_size_gb = file_size / (1024 ** 3)
        Log.error(
            f"Test file size {file_size_gb:.2f}GB is less than minimum "
            f"{MIN_FILE_SIZE_GB}GB: {path}"
        )
        sys.exit(1)

def run_bench(rw="read", io_size=4, thread=1, batch_size=16, file_path_=FILE_PATH,
              async_mode=0, xfer_mode=0, device_id=0, length_gb=DEFAULT_LENGTH_GB):
    length_arg = f"{length_gb}G"
    if batch_size > 64:
        return f"{micro_exec} -f {file_path_} -l {length_arg} -s {io_size}k -t {thread} -i {batch_size} -m {rw} -a {async_mode} -d {device_id} -x {xfer_mode}"
    return f"numactl -N 0 {micro_exec} -f {file_path_} -l {length_arg} -s {io_size}k -t {thread} -i 1 -m {rw} -a {async_mode} -d {device_id} -x {xfer_mode}"

def parse_result(result):
    matches = re.findall(pattern, result)
    matches = [float(match) for match in matches]
    return matches[0], matches[1], matches[2], matches[3], matches[4]

def x_thread_y_size_z_batch(config: test_config):
    f = open(result_path, "a+")
    print(f"result_path: {result_path}")
    io_size_iter = io_sizes if config.multi_size == True else [4]
    thread_iter = threads if config.multi_thread  == True else [1]
    batch_size_iter = batch_sizes if config.multi_batch == True else [1]
    f.write("gpuid-length_gb-thread-rw-io_size-batch_size,bandwidth,latency,p95_latency,p99_latency,p999_latency\n")
    f.write(f"async_mode: {config.async_mode}, xfer_mode: {config.xfer_mode}\n")
    for rw in read_write:
        for io_size in io_size_iter:
            for thread in thread_iter:
                for batch_size in batch_size_iter:
                    # subprocess.run("echo 3 | sudo tee /proc/sys/vm/drop_caches", shell=True)
                    cmdline = run_bench(rw=rw, io_size=io_size, thread=thread, batch_size=batch_size,
                                        async_mode=config.async_mode, xfer_mode=config.xfer_mode,
                                        device_id=config.device_id, file_path_=FILE_PATH,
                                        length_gb=config.length_gb)
                    Log.info(f"Run {cmdline}")
                    result = subprocess.check_output(cmdline, shell=True).decode()
                    Log.info(result)
                    bandwidth, latency, p95_latency, p99_latency, p999_latency = parse_result(result)
                    f.write(f"{config.device_id}-{config.length_gb}-{thread}-{rw}-{io_size}-{batch_size},{bandwidth},{latency},{p95_latency},{p99_latency},{p999_latency}\n")
                    f.flush()

def run_perf_cpu(pid: int):
    return subprocess.Popen("")


if __name__ == "__main__":
    if len(sys.argv) not in (6, 7):
        Log.error("Usage: python micro.py <device_id> <xfer_mode> <mode> <device_type> <file_path> [length_gb]")
        Log.info("device_id: GPU device id for -d (e.g. 0)")
        Log.info("xfer_mode: fgds, gds, posix")
        Log.info("mode: sync, async, batch")
        Log.info("device_type: nvme, nvmeof")
        Log.info(f"file_path: test file path (must be at least {MIN_FILE_SIZE_GB}GB)")
        Log.info(f"length_gb: optional, IO length for -l (default {DEFAULT_LENGTH_GB}, i.e. -l {DEFAULT_LENGTH_GB}G)")
        sys.exit(1)

    device_id = int(sys.argv[1])             # GPU device id for -d
    xfer_mode_str  = sys.argv[2].lower()     # fgds / gds / posix
    run_mode_str   = sys.argv[3].lower()     # sync / async / batch
    device_type_str = sys.argv[4].lower()    # 0 - nvme / 1 - nvmeof
    FILE_PATH = sys.argv[5]
    length_gb = DEFAULT_LENGTH_GB
    if len(sys.argv) == 7:
        try:
            length_gb = int(sys.argv[6])
        except ValueError:
            Log.error(f"Invalid length_gb: {sys.argv[6]}, must be a positive integer")
            sys.exit(1)
        if length_gb <= 0:
            Log.error(f"Invalid length_gb: {length_gb}, must be a positive integer")
            sys.exit(1)

    validate_file_size(FILE_PATH)

    xfer_mode_map  = {"fgds": 0, "gds": 1, "posix": 2}
    run_mode_map   = {"sync": 0, "async": 1, "batch": 2}
    device_type_map = { "nvme": "0", "nvmeof": "1" }

    
    if xfer_mode_str not in xfer_mode_map:
        Log.error("Invalid xfer_mode. Must be 'fgds' or 'gds' or 'posix'.")
        sys.exit(1)
    if run_mode_str not in run_mode_map:
        Log.error("Invalid mode. Must be 'sync', 'async' or 'batch'.")
        sys.exit(1)
    if device_type_str not in device_type_map:
        Log.error("Invalid device type. Must be 0 (nvme) or 1 (nvmeof).")
        sys.exit(1)

    xfer_mode   = xfer_mode_map[xfer_mode_str]   # 0 / 1 / 2
    async_mode  = run_mode_map[run_mode_str]     # 0 / 1 / 2
    device_type = device_type_map[device_type_str]           # 0 / 1 
    
    result_dir = os.path.join(file_path, "results", "latency", xfer_mode_str)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    result_path = os.path.join(result_dir, f"{run_mode_str}_{device_type_str}.txt")

    config = test_config()
    config.reset()
    config.device_id = device_id
    config.async_mode = async_mode
    config.xfer_mode = xfer_mode
    config.length_gb = length_gb
    
    # change the config to test different modes
    # for example, config.multi_thread indicates that multi-threading testing will be conducted
    config.multi_size = True
    config.multi_thread = True
    x_thread_y_size_z_batch(config)
    config.reset()
