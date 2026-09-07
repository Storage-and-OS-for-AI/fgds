cmake_minimum_required (VERSION 3.18)
project (libfgds-benchmarks)

file(GLOB safetensor_src "${benchmarks_root}/safetensor/*.cc")

make_benchmark (safetensor safetensor "${safetensor_src}")
