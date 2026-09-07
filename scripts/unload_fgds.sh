#!/bin/bash
# 卸载 fgds 内核模块。与 load_fgds.sh 成对使用。
# 用法:
#   scripts/unload_fgds.sh

set -e
MODULE_NAME="fgdsfs"

if grep -q "^$MODULE_NAME " /proc/modules; then
	echo "Unloading fgds module: $MODULE_NAME"
	sudo rmmod "$MODULE_NAME"
	echo "Unloaded $MODULE_NAME"
else
	echo "$MODULE_NAME module is not loaded."
fi
