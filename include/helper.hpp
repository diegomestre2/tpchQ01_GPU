#pragma once
#include <cuda_runtime.h>
#include <cub/cub.cuh>


#if !defined(__global__)
#define __global__
#endif

void get_device_properties(){
    int32_t device_cnt = 0;
    cudaGetDeviceCount(&device_cnt);
    cudaDeviceProp device_prop;

    for (int i = 0; i < device_cnt; i++) {
        cudaGetDeviceProperties(&device_prop, i);
        std::cout << "+-------------------------------------------------------------------------------+\n";
        printf("|  Device id: %d\t", i);
        printf("  Device name: %s\t", device_prop.name);
        printf("  Compute capability: %d.%d\n", device_prop.major, device_prop.minor);
        std::cout << std::endl;
        printf("|  Memory Clock Rate [KHz]: %d\n",
               device_prop.memoryClockRate);
        printf("|  Memory Bus Width [bits]: %d\n",
               device_prop.memoryBusWidth);
        printf("|  Peak Memory Bandwidth [GB/s]: %f\n",
               2.0*device_prop.memoryClockRate*(device_prop.memoryBusWidth/8)/1.0e6);
        printf("|  L2 size [KB]: %d\n",
               device_prop.l2CacheSize/1024);
        printf("|  Shared Memory per Block Size [KB]: %lu\n",
              device_prop.sharedMemPerBlock/1024);
        std::cout << std::endl;
        printf("|  Number of SMs: %d\n",
               device_prop.multiProcessorCount);
        printf("|  Max. number of threads per SM: %d\n",
               device_prop.maxThreadsPerMultiProcessor);
        printf("|  Concurrent kernels: %d\n",
               device_prop.concurrentKernels);
        printf("|  warpSize: %d\n",
               device_prop.warpSize);
        printf("|  maxThreadsPerBlock: %d\n",
               device_prop.maxThreadsPerBlock);
        printf("|  maxThreadsDim[0]: %d\n",
               device_prop.maxThreadsDim[0]);
        printf("|  maxGridSize[0]: %d\n",
               device_prop.maxGridSize[0]);
        printf("|  pageableMemoryAccess: %d\n",
               device_prop.pageableMemoryAccess);
        printf("|  concurrentManagedAccess: %d\n",
               device_prop.concurrentManagedAccess);
        printf("|  Number of async. engines: %d\n",
               device_prop.asyncEngineCount);
        std::cout << "+-------------------------------------------------------------------------------+\n";
    }
}

__device__
static uint32_t warp_size = 32;


/// returns the global id of the executing thread
__device__ inline uint32_t
global_thread_id() {
  return blockIdx.x * blockDim.x + threadIdx.x;
}

__device__ inline uint32_t
global_size() {
  return gridDim.x + blockDim.x;
}

/// returns the block id in [0, grid_size)
__device__ inline uint32_t
block_id() {
  return blockIdx.x;
}

/// returns the thread id within the current block: [0, block_size)
__device__ inline uint32_t
block_local_thread_id() {
  return threadIdx.x;
}

/// returns the warp id within the current block
/// the id is in [0, u), where u = block_size / warp_size
__device__ inline uint32_t
block_local_warp_id() {
  return block_local_thread_id() / warp_size;
}

/// returns the warp id (within the entire grid)
__device__ inline uint32_t
global_warp_id() {
  return global_thread_id() / warp_size;
}

/// returns the thread id [0,32) within the current warp
__device__ inline uint32_t
warp_local_thread_id() {
  return block_local_thread_id() % warp_size;
}


// taken from: https://codeyarns.com/2011/03/02/how-to-do-error-checking-in-cuda/
#define cuda_check_error()    __cuda_check_error( __FILE__, __LINE__ )
inline void
__cuda_check_error(const char *file, const int line ) {
  cudaError err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "cuda_check_error() failed at %s:%i : %s\n",
            file, line, cudaGetErrorString(err) );
    exit(-1);
  }
}
