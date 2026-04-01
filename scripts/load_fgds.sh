#!/bin/bash
# 从项目根目录 config.json 读取 use_all_gpus 和 gpuids，加载 fgds 内核模块并传入对应参数。
# 用法: scripts/load_fgds.sh  或 从项目根目录执行 ./scripts/load_fgds.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG="$ROOT_DIR/config.json"
MODULE_NAME="fgdsfs"
INSTALL_DIR="/opt/fgds"

if [ -d "$INSTALL_DIR/module" ]; then
  MODULE_DIR="$INSTALL_DIR/module"
else
  MODULE_DIR="$ROOT_DIR/build/module"
fi
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
echo "Loading fgds with use_all_gpus=$use_all_gpus gpuids=$gpuids_param"
sudo insmod "$MOD_PATH" use_all_gpus="$use_all_gpus" gpuids="$gpuids_param"
