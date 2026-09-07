#!/bin/bash
# 重载 fgds 内核模块：先卸载再加载，等效于 rmmod + insmod。
# 复用了 unload_fgds.sh 与 load_fgds.sh，因此加载参数仍由 config.json 决定。
# 用法:
#   scripts/reload_fgds.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

UNLOAD="$SCRIPT_DIR/unload_fgds.sh"
LOAD="$SCRIPT_DIR/load_fgds.sh"

for s in "$UNLOAD" "$LOAD"; do
	if [ ! -f "$s" ]; then
		echo "error: required script not found: $s" >&2
		exit 1
	fi
done

echo "Reloading fgds module..."
echo "== unload =="
bash "$UNLOAD"
echo "== load =="
bash "$LOAD"
echo "Reload finished."
