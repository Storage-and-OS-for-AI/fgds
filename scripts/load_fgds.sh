#!/bin/bash
# 从项目根目录 config.json 读取 use_all_gpus 和 gpuids，加载 fgds 内核模块并传入对应参数。
# 用法:
#   scripts/load_fgds.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG="$ROOT_DIR/config.json"
MODULE_NAME="fgdsfs"

MODULE_DIR="$ROOT_DIR/build/module"
echo "MODULE_DIR: $MODULE_DIR"

if [ ! -f "$CONFIG" ]; then
	echo "config.json not found at $CONFIG"
	exit 1
fi

# 使用 Python 解析 JSON，避免依赖 jq
eval "$(python3 - "$CONFIG" << 'PY'
import json, sys
with open(sys.argv[1]) as f:
    c = json.load(f)
use = 1 if c.get("use_all_gpus", False) else 0
ids = c.get("gpuids", [0])
if not isinstance(ids, list) or not all(isinstance(x, int) and not isinstance(x, bool) for x in ids):
    # 经 eval 执行：exit 1 由 eval 自身承担，set -e 才能中断脚本
    print('echo "error: gpuids must be a list of integers" >&2; exit 1')
else:
    gpuids_str = ",".join(str(x) for x in ids)
    print("use_all_gpus=%d" % use)
    print("gpuids_param=%s" % gpuids_str)
PY
)"

MOD_PATH="$MODULE_DIR/${MODULE_NAME}.ko"
if [ ! -f "$MOD_PATH" ]; then
	echo "Module not found at $MOD_PATH, build the module first."
	cd $MODULE_DIR && make
	[ $? -ne 0 ] && exit 1
fi

echo "MOD_PATH: $MOD_PATH"

# use_all_gpus=1 时使用所有 GPU，不需要传 gpuids 参数；
# 特别地，当 config.json 中 gpuids 为空数组时，传空串会触发内核
# module_param_array 对空值的 -EINVAL，因此这里直接省略该参数。
# use_all_gpus=0 时按 config.json 中 gpuids 指定的索引加载；若 gpuids
# 为空，属于非法配置，在此前置拦截，避免走到 insmod 的晦涩报错。
if [ "$use_all_gpus" = "1" ]; then
	echo "Loading fgds with use_all_gpus=1 (use all GPUs)"
	sudo insmod "$MOD_PATH" use_all_gpus=1
else
	if [ -z "$gpuids_param" ]; then
		echo "config error: use_all_gpus=false but gpuids is empty, please specify at least one GPU index" >&2
		exit 1
	fi
	echo "Loading fgds with use_all_gpus=0 gpuids=$gpuids_param"
	sudo insmod "$MOD_PATH" use_all_gpus=0 gpuids="$gpuids_param"
fi

grep -q "^$MODULE_NAME " /proc/modules && echo "$MODULE_NAME is loaded." || echo "warning: $MODULE_NAME is not loaded after insmod." >&2
