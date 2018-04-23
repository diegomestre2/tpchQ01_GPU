#pragma once
#include <cuda_runtime.h>
#include <cub/cub/cub.cuh>


#if !defined(__global__)
#define __global__
#endif


static u32 warp_size = 32;


/// returns the global id of the executing thread
__device__ inline u32
global_thread_id() {
  return blockIdx.x * blockDim.x + threadIdx.x;
}

__device__ inline u32
global_size() {
  return gridDim.x + blockDim.x;
}

/// returns the block id in [0, grid_size)
__device__ inline u32
block_id() {
  return blockIdx.x;
}

/// returns the thread id within the current block: [0, block_size)
__device__ inline u32
block_local_thread_id() {
  return threadIdx.x;
}

/// returns the warp id within the current block
/// the id is in [0, u), where u = block_size / warp_size
__device__ inline u32
block_local_warp_id() {
  return block_local_thread_id() / warp_size;
}

/// returns the warp id (within the entire grid)
__device__ inline u32
global_warp_id() {
  return global_thread_id() / warp_size;
}

/// returns the thread id [0,32) within the current warp
__device__ inline u32
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
