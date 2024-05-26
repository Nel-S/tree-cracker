#ifndef __NON_CUDA_SUPPORT_CUH
#define __NON_CUDA_SUPPORT_CUH

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda.h>
#include <stdexcept>

#ifndef CUDA_VERSION
	#define CUDA_IS_PRESENT false
	#include <pthread.h>

	#define __device__
	#define __host__
	#define __global__
	#define __managed__
	// typedef struct {
	//     int x;
	// } tempStruct;
	// tempStruct blockDim, blockIdx, threadIdx;

	#define cudaError_t int
	#define cudaSuccess 0

	// Returns the number of 1s in a 32-bit integer.
	// (The unintuitive name is from NVIDIA, not me.)
	int __popc(uint32_t x) {
		int count = 0;
		for (; static_cast<bool>(x); x >>= 1) count += static_cast<int>(x & 1);
		return count;
	}

	pthread_mutex_t __mutex;
	// Atomically adds the specified value to the value stored in the specified address. Returns the original value in the address.
	unsigned long long atomicAdd(unsigned long long *address, const int value) {
		unsigned long long __temp = *address;
		pthread_mutex_lock(&__mutex);
		*address += static_cast<unsigned long long>(value);
		pthread_mutex_unlock(&__mutex);
		return __temp;
	}
	// Atomically ORs the specified value to the value stored in the specified address. Returns the original value in the address.
	unsigned long long atomicOr(uint32_t *address, const uint32_t value) {
		uint32_t __temp = *address;
		pthread_mutex_lock(&__mutex);
		*address |= value;
		pthread_mutex_unlock(&__mutex);
		return __temp;
	}

	pthread_t *threads;
	size_t __numberOfThreads;
	cudaError_t cudaDeviceSynchronize() {
		for (size_t i = 0; i < __numberOfThreads; ++i) pthread_join(threads[i], NULL);
		return cudaSuccess;
	}
	char *cudaGetErrorString(cudaError_t error) {
		return "";
	}
	cudaError_t cudaGetLastError() {
		return cudaSuccess;
	}
	cudaError_t cudaGetSymbolAddress(void** devPtr, const void* symbol) {
		// devPtr = &symbol;
		memcpy(devPtr, symbol, sizeof(symbol));
		return cudaSuccess;
	}

	#define cudaMemsetAsync memset
	enum cudaMemcpyKind {cudaMemcpyDefault};
	cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind) {
		memcpy(dst, src, count);
		return cudaSuccess;
	}
	#define THROW_EXCEPTION(DEFAULT_VALUE_IF_CUDA, ERROR_MESSAGE_IF_NOT_CUDA) throw std::invalid_argument(ERROR_MESSAGE_IF_NOT_CUDA)
	#define FILTER_RETURN return NULL
#else
	#define CUDA_IS_PRESENT true
	#define THROW_EXCEPTION(DEFAULT_VALUE_IF_CUDA, ERROR_MESSAGE_IF_NOT_CUDA) return DEFAULT_VALUE_IF_CUDA
	#define FILTER_RETURN return
#endif

#define ABORT(...) { \
	std::fprintf(stderr, __VA_ARGS__); \
	std::abort(); \
}

#define TRY_CUDA(expr) __tryCuda(expr, __FILE__, __LINE__)

void __tryCuda(cudaError_t error, const char *file, uint64_t line) {
	if (error == cudaSuccess) return;
	ABORT("CUDA error at %s:%" PRIu64 ": %s\n", file, line, cudaGetErrorString(error));
}

#endif