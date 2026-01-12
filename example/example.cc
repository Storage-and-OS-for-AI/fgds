#include <sys/types.h>
#include <unistd.h>
#include <cstddef>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <iostream>
#include <cerrno>
#include "fgds.h"
#include <ctime>

const char *file_path = "/data3/10GB_ddrand";
static int device_id = 4;
static size_t io_size = 1 * (1 << 20); // 
static size_t buff_size = 10ULL * (1 << 30); // 10GB

#define check_cudaruntimecall(fn) \
	do { \
		cudaError_t res = fn; \
		if (res != cudaSuccess) { \
			const char *str = cudaGetErrorName(res); \
			std::cerr << "cuda runtime api call failed " << #fn \
				<<  __LINE__ << ":" << str << std::endl; \
			std::cerr << "EXITING program!!!" << std::endl; \
			exit(1); \
		} \
	} while(0)

// macro definition
#define point_offset(ptr, offset) reinterpret_cast<void*>(reinterpret_cast<uint64_t>(ptr) + offset)

struct timespec get_elapsed_timespec(struct timespec start, struct timespec end) {
    struct timespec elapsed;

    elapsed.tv_sec = end.tv_sec - start.tv_sec;
    elapsed.tv_nsec = end.tv_nsec - start.tv_nsec;

    // 处理纳秒借位
    if (elapsed.tv_nsec < 0) {
        elapsed.tv_sec -= 1;
        elapsed.tv_nsec += 1000000000;
    }
    return elapsed;
}

double timespec_to_double(const struct timespec& ts) {
	constexpr double NANOSECONDS_TO_SECONDS = 1e-9;
    return static_cast<double>(ts.tv_sec) +
           static_cast<double>(ts.tv_nsec) * NANOSECONDS_TO_SECONDS;
}

double cal_bw(size_t bytes, double seconds) {
 	if (seconds <= 0.0) return 0.0;  // 避免除以0
    
    // 1 GB = 1024^3 = 1073741824 字节
    constexpr double BYTES_PER_GB = 1024.0 * 1024.0 * 1024.0;
    
    double gigabytes = static_cast<double>(bytes) / BYTES_PER_GB;
    return gigabytes / seconds;
}

double cal_bw(size_t bytes, const struct timespec& ts) {
	double seconds = timespec_to_double(ts);
	cal_bw(bytes, seconds);
}

void add_timespec(struct timespec& total, const struct timespec& addend) {
    total.tv_sec += addend.tv_sec;
    total.tv_nsec += addend.tv_nsec;

    // 处理纳秒进位（超过10^9纳秒=1秒）
    if (total.tv_nsec >= 1000000000L) {
        total.tv_sec += 1;
        total.tv_nsec -= 1000000000L;
    }
}

int test_posix_loop() {
    int file_fd, ret;
    struct timespec prog_start, prog_end;
    void *gpu_buffer = NULL;
    void *data_buffer = NULL;
    size_t  data_size = 10ULL * 1024 * 1024 * 1024; // 10GB

    file_fd = open("/data3/10GB_ddrand",  O_CREAT | O_RDWR | O_DIRECT, 0644);
    if (file_fd < 0) {
        perror("Open file error");
        return -1;
    }

    check_cudaruntimecall(cudaSetDevice(4));
    check_cudaruntimecall(cudaMalloc(&gpu_buffer, buff_size));
    check_cudaruntimecall(cudaMemset(gpu_buffer, 0x00, buff_size));
    check_cudaruntimecall(cudaStreamSynchronize(0));

    ret = posix_memalign(&data_buffer, 4096, buff_size);
    if (ret != 0) {
        data_buffer = NULL;
        printf("buffer alloc error");
		return -1;
    }

    size_t read_bytes = 0;
    ssize_t result;
    struct timespec io_start, io_readfd_end, io_end;
	struct timespec total_read_fd_elapsed {}; 
	struct timespec total_cudamemcpy_elapsed {}; 

    clock_gettime(CLOCK_MONOTONIC, &prog_start);
	int read_count = 0;

    while(read_bytes < data_size) {
        clock_gettime(CLOCK_MONOTONIC, &io_start); 

        result = pread(file_fd, point_offset(data_buffer, read_bytes), io_size, read_bytes);
        if (result == 0) {
            // End of file reached
            printf("read file end\n");
            break;
        }
        if (result != io_size) {
            std::cerr << "read_thread error, result is " << result << ", size is " << io_size << std::endl;
            return NULL;           
        }
        clock_gettime(CLOCK_MONOTONIC, &io_readfd_end);

        struct timespec readfd_elapsed = get_elapsed_timespec(io_start, io_readfd_end);
		add_timespec(total_read_fd_elapsed, readfd_elapsed);
//		printf("readfd_elapsed:%f , total_read_fd_elapsed:%f\n", timespec_to_double(readfd_elapsed), timespec_to_double(total_read_fd_elapsed));

        check_cudaruntimecall(cudaMemcpy(
            point_offset(gpu_buffer, read_bytes),
            point_offset(data_buffer, read_bytes),
            io_size, cudaMemcpyHostToDevice));
        check_cudaruntimecall(cudaStreamSynchronize(0));
        clock_gettime(CLOCK_MONOTONIC, &io_end);

        struct timespec cudamemcpy_elapsed = get_elapsed_timespec(io_readfd_end, io_end);
		add_timespec(total_cudamemcpy_elapsed, cudamemcpy_elapsed);

 /*       printf("read %zu bytes from file to memory takes %ld.%09ld; cudaMemcpy %zu bytes, takes %ld.%09ld\n", 
            result, readfd_elapsed.tv_sec, readfd_elapsed.tv_nsec,
            io_size, cudamemcpy_elapsed.tv_sec, cudamemcpy_elapsed.tv_nsec); */

        read_bytes += result;
		read_count++;
    }
    clock_gettime(CLOCK_MONOTONIC, &prog_end);

    struct timespec total_elapsed = get_elapsed_timespec(prog_start, prog_end);
	double total_cost = timespec_to_double(total_elapsed), total_readfd_cost = timespec_to_double(total_read_fd_elapsed), total_cudamemcpy_cost = timespec_to_double(total_cudamemcpy_elapsed);
    printf("io_size:%zu bytes(%zuMB), buffer_size:%zu bytes(%zuGB), posix read %zu bytes(%zuGB), read_count:%d,total cost:%fs, total read_fd cost:%fs, total cudamemcpy cost:%fs, total read_fd+cudamemcpy cost:%fs, left cost:%fs\n",
		io_size, io_size/1024/1024, buff_size, buff_size/1024/1024/1024, data_size, data_size/1024/1024/1024, 
		read_count, total_cost, total_readfd_cost, total_cudamemcpy_cost, 
		total_readfd_cost + total_cudamemcpy_cost, total_cost - total_readfd_cost - total_cudamemcpy_cost);
	printf("total read bw:%fGB/s, total read_fd bw:%fGB/s, total cudaMemcpy bw:%fGB/s\n", 
		cal_bw(data_size, total_elapsed), cal_bw(data_size, total_read_fd_elapsed), cal_bw(data_size, total_cudamemcpy_elapsed));

	free(data_buffer);
	check_cudaruntimecall(cudaFree(gpu_buffer));
	close(file_fd);
	return 0;
}

int test_posix_once() {
    int file_fd, ret;
    void *gpu_buffer = NULL;
    void *data_buffer = NULL;

    file_fd = open("/data3/10GB_ddrand",  O_CREAT | O_RDWR | O_DIRECT, 0644);
    if (file_fd < 0) {
        perror("Open file error");
        return -1;
    }

    check_cudaruntimecall(cudaSetDevice(4));
    check_cudaruntimecall(cudaMalloc(&gpu_buffer, buff_size));
    check_cudaruntimecall(cudaMemset(gpu_buffer, 0x00, buff_size));
    check_cudaruntimecall(cudaStreamSynchronize(0));

    ret = posix_memalign(&data_buffer, 4096, buff_size);
    if (ret != 0) {
        data_buffer = NULL;
        printf("buffer alloc error");
        return -1;
    }

    ssize_t result;
    struct timespec io_start, io_readfd_end, io_end;
	io_size = 1UL * 1024 * 1024 * 1024; // 1GB.  一次读如果大于1GB,会报错,读到的实际字节数<要求读到的字节数

    clock_gettime(CLOCK_MONOTONIC, &io_start);
	result = pread(file_fd, point_offset(data_buffer, 0), io_size, 0);
	if (result != io_size) {
		std::cerr << "read_thread error, result is " << result << ", size is " << io_size << std::endl;
		return NULL;
	}
	clock_gettime(CLOCK_MONOTONIC, &io_readfd_end);

	struct timespec total_read_fd_elapsed = get_elapsed_timespec(io_start, io_readfd_end);

	check_cudaruntimecall(cudaMemcpy(
		point_offset(gpu_buffer, 0),
		point_offset(data_buffer, 0),
		io_size, cudaMemcpyHostToDevice));
	check_cudaruntimecall(cudaStreamSynchronize(0));
	clock_gettime(CLOCK_MONOTONIC, &io_end);

	struct timespec total_cudamemcpy_elapsed = get_elapsed_timespec(io_readfd_end, io_end);
	struct timespec total_elapsed = get_elapsed_timespec(io_start, io_end);
	double total_cost = timespec_to_double(total_elapsed), total_readfd_cost = timespec_to_double(total_read_fd_elapsed), total_cudamemcpy_cost = timespec_to_double(total_cudamemcpy_elapsed);

	printf("io_size:%zu bytes(%zuMB), buffer_size:%zu bytes(%zuGB), posix read %zu bytes(%zuGB), total cost:%fs, total read_fd cost:%fs, total cudamemcpy cost:%fs, total read_fd+cudamemcpy cost:%fs, left cost:%fs\n",
	io_size, io_size/1024/1024, buff_size, buff_size/1024/1024/1024, io_size, io_size/1024/1024/1024, total_cost, total_readfd_cost, total_cudamemcpy_cost,
        total_readfd_cost + total_cudamemcpy_cost, total_cost - total_readfd_cost - total_cudamemcpy_cost);

	printf("total read bw:%fGB/s, total read_fd bw:%fGB/s, total cudaMemcpy bw:%fGB/s\n",
    	cal_bw(io_size, total_elapsed), cal_bw(io_size, total_read_fd_elapsed), cal_bw(io_size, total_cudamemcpy_elapsed));	

   free(data_buffer);
   check_cudaruntimecall(cudaFree(gpu_buffer));
   close(file_fd);
   return 0;
}

int fgds_demo() {
    void *gpu_buffer, *target_addr;
    int ret;
    int file_fd;
    ssize_t result; 

    file_fd = open(file_path, O_CREAT | O_RDWR | O_DIRECT, 0644);

    printf("fgds init start\n");
    ret = fgds_open(device_id);
    printf("fgds init ret: %d\n", ret);

    if (ret != 0) {
        printf("fgds init failed: %d\n", ret);
        return 1;
    }   
    cudaMalloc(&gpu_buffer, io_size);
    cudaMemset(gpu_buffer, 0x00, io_size);
    cudaStreamSynchronize(0);
    printf("fgds regmem start\n");
    // target_addr for register buffer less than 1GB
    ret = fgds_regmem(device_id, gpu_buffer, io_size, &target_addr);
    printf("fgds regmem ret: %d\n", ret);
    if (ret) {
        printf("fgds regmem failed: %d\n", ret);
        return 1;
    }
    printf("fgds regmem target_addr: %p\n", target_addr);
    result = pread(file_fd, target_addr, io_size, 0);
    printf("fgds pread ret: %ld\n", result);
    if (result < 0) {
        perror("Read file error");
        return 1;
    }
    printf("fgds pread done\n");
    ret = fgds_deregmem(device_id, gpu_buffer, io_size);
    printf("fgds unregmem ret: %d\n", ret);
    if (ret) {
        printf("fgds unregmem failed: %d\n", ret);
        return 1;
    }

    cudaFree(gpu_buffer);
    fgds_close(device_id);
    close(file_fd);
}

int main() {
	cudaSetDevice(device_id);
	fgds_demo();
}
