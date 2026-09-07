cmake_minimum_required (VERSION 3.18)
project (libfgds LANGUAGES CUDA C CXX)

set (CMAKE_C_STANDARD 99)
set (CMAKE_CXX_STANDARD 20)
set (CMAKE_CUDA_STANDARD 20)
set (CMAKE_CUDA_ARCHITECTURES 90)

find_package (Threads REQUIRED)
find_package (CUDAToolkit 12.4 REQUIRED)
find_library (CUFILE_LIBRARIES cufile ${CUDAToolkit_LIBRARY_DIR})
find_library (DL_LIB dl REQUIRED)

# 动态链接系统 liburing：写入 libfgds.so 的 DT_NEEDED，供 Python 加载时自动解析。
# 构建机需 liburing-devel（或 liburing-dev）；运行机需 liburing 运行时库。
find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBURING REQUIRED IMPORTED_TARGET liburing)
message(STATUS "liburing include dirs: ${LIBURING_INCLUDE_DIRS}")
message(STATUS "liburing libraries: ${LIBURING_LINK_LIBRARIES}")

set (NVIDIA "" CACHE PATH "Path to Nvidia driver source")
set (KERNEL "/lib/modules/${CMAKE_SYSTEM_VERSION}/build" CACHE PATH "Path to kernel source, or module directory")

set (nvidia_archs "90" CACHE STRING "NVIDIA compute architectures")
set (no_module false CACHE BOOL "Do not build kernel module")
set (no_cuda false CACHE BOOL "Do not build library with CUDA support")
set (BUILD_SHARED_LIBS true CACHE BOOL "Build shared library, set to false for static")

if (no_smartio)
    set (no_smartio_samples true)
    set (no_smartio_benchmarks true)
endif ()

# Set global compiler flags
set (GPU_ARCHS "" CACHE INTERNAL "" FORCE)
foreach (arch ${nvidia_archs})
    string (CONCAT GPU_ARCHS ${GPU_ARCHS} " -gencode arch=compute_${arch},code=sm_${arch}")
endforeach ()

set (convert_warnings implicit-function-declaration incompatible-pointer-types int-conversion return-type unused-function unused-variable)
set (flags "")
foreach (flag ${convert_warnings})
    string (APPEND flags " -Werror=${flag}")
endforeach()

# C_FLAGS
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra ${flags}" CACHE INTERNAL "" FORCE)
set (CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_DEBUG} -g -DDEBUG" CACHE INTERNAL "" FORCE)
set (CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O2" CACHE INTERNAL "" FORCE)
# CXX_FLAGS
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -std=c++20" CACHE INTERNAL "" FORCE)
set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG " CACHE INTERNAL "" FORCE)
set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2 -std=c++20" CACHE INTERNAL "" FORCE)

if (no_cc)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D__NO_COHERENCE__")
    set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -D__NO_COHERENCE__")
    set(CCUDA_NVCC_FLAGS "${CMAKE_C_FLAGS} -D__NO_COHERENCE__")
endif ()

include_directories ("${PROJECT_SOURCE_DIR}/libfgds/include"
                    "${CUDAToolkit_INCLUDE_DIRS}")

# Try to find Nvidia driver
if (CUDAToolkit_FOUND AND NOT no_cuda)

    # Try to find Nvidia headers (will choose the newest version)
    file (GLOB_RECURSE driver_paths "/usr/src/nvidia-*/nv-reg.h")
    string (REPLACE "nv-reg.h" "" driver_paths "${driver_paths}")
    message(STATUS "driver path is ${driver_paths}")
    if (driver_paths)
        list (REVERSE driver_paths)
    endif ()

    find_path (driver_include NAMES "nv-p2p.h" PATHS "${NVIDIA}" ${driver_paths})
    set (driver_dir "${driver_include}")

    # Find Nvidia driver symbols
    if (NOT EXISTS "${driver_dir}/Module.symvers")
        get_filename_component (driver_dir "${driver_dir}" DIRECTORY)
    endif ()

    if ("${driver_dir}" STREQUAL "")
        message(STATUS "No NVIDIA driver symbols found")
    else ()
        message(STATUS "Using NVIDIA driver found in ${driver_dir}")
    endif ()
endif ()

set (install_dir "/opt/fgds")
if (NOT no_module)
    set (module_root "${PROJECT_SOURCE_DIR}/module")
    set (module_output "${CMAKE_BINARY_DIR}/module")
else()
    # rpm builds do not generate kernel modules but provide users a module directory so that they can build them themselves.
    set (module_root "${install_dir}/module")
    set (module_output "${install_dir}/module")
endif ()

# for Makefile.in, do not delete
set (module_ccflags "-I${driver_include}")
configure_file ("${PROJECT_SOURCE_DIR}/module/Makefile.in" "${CMAKE_BINARY_DIR}/module/Makefile" @ONLY)
message (STATUS "Configuring module build in ${module_output}")

# # Module source files (skip when no_module=ON)
if (NOT no_module)
    add_custom_target(
        modules
        ALL
        COMMAND make
        WORKING_DIRECTORY ${module_output}
        COMMENT "Building kernel module"
    )

    add_custom_target(
        clean_modules
        COMMAND make clean
        WORKING_DIRECTORY ${module_output}
        COMMENT "Cleaning kernel module"
    )

    add_custom_target(
        insmod
        COMMAND sudo make insmod
        WORKING_DIRECTORY ${module_output}
        COMMENT "Inserting kernel module"
    )

    add_custom_target(
        rmmod
        COMMAND sudo make rmmod
        WORKING_DIRECTORY ${module_output}
        COMMENT "Removing kernel module"
    )
else ()
    message (STATUS "Skipping kernel module build (no_module=ON)")
endif ()


# Library source files
set(libfgds_root "${PROJECT_SOURCE_DIR}/libfgds")
file(GLOB libfgds_source "${libfgds_root}/*.cc")
file(GLOB libfgds_include "${libfgds_root}/include/*.h")

add_library(libfgds ${libfgds_source} ${libfgds_include})
set_target_properties (libfgds PROPERTIES OUTPUT_NAME "fgds")
# PkgConfig::LIBURING（PUBLIC）：
# 1) 编译 libfgds 时带上 liburing 头文件；
# 2) 链接时写入 DT_NEEDED(liburing.so)，供 Python 加载 libfgds.so 时自动解析；
# 3) 传递给依赖 libfgds 的目标（部分 benchmark 也直接调用了 io_uring API）。
target_include_directories(libfgds PRIVATE "${libfgds_root}" "${libfgds_root}/include" CUDA::cudart)
target_link_libraries(libfgds PUBLIC PkgConfig::LIBURING)
# Set install locations
install (FILES ${libfgds_include} DESTINATION "include")
install (TARGETS libfgds DESTINATION "lib")


# Make target depend on pthreads
macro (set_multithread target)
    target_link_libraries (${target} Threads::Threads)
endmacro ()

# Benchmarks
set (benchmarks_root "${PROJECT_SOURCE_DIR}/benchmarks")
set (benchmarks_utils "${benchmarks_root}/utils/")
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

# Set preferred ordering for sample targets
# Make benchmark target
# We assume all benchmarks are CUDA programs
macro (make_benchmark target binary_name files)

    add_executable (${target}) # Ugly bugly
    target_sources (${target} PRIVATE ${files})

    add_dependencies (${target} libfgds)
    # liburing 的 include/link 已由 libfgds PUBLIC PkgConfig::LIBURING 传递；
    # 此处再显式链接一次，保证 benchmark 自身直接调用的 io_uring_* 也能正确解析。
    target_include_directories(${target} PRIVATE ${benchmarks_utils} ${libfgds_root}/include)
    target_link_libraries (${target} libfgds ${CUFILE_LIBRARIES} CUDA::cudart cuda PkgConfig::LIBURING ${DL_LIB} Threads::Threads)
    set_target_properties (${target} PROPERTIES OUTPUT_NAME "${binary_name}")

    list (APPEND benchmark_targets "${target}")
    set (benchmark_targets "${benchmark_targets}" PARENT_SCOPE)

endmacro ()

add_subdirectory ("${PROJECT_SOURCE_DIR}/benchmarks/breakdown")
add_subdirectory ("${PROJECT_SOURCE_DIR}/benchmarks/end-to-end")
add_subdirectory ("${PROJECT_SOURCE_DIR}/benchmarks/kvcache")
add_subdirectory ("${PROJECT_SOURCE_DIR}/benchmarks/micro-benchmarks")
add_subdirectory ("${PROJECT_SOURCE_DIR}/benchmarks/safetensor")

add_subdirectory ("${PROJECT_SOURCE_DIR}/example/")
add_subdirectory ("${PROJECT_SOURCE_DIR}/tests/")

