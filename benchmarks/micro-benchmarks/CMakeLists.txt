cmake_minimum_required (VERSION 3.18)
project (libfgds-benchmarks)

make_benchmark (microbenchmark microbenchmark "main.cc;gds_test.cc;fgds_test.cc;posix_test.cc")
