#pragma once

// #include <stdio.h>
// #include <iostream>
// #include <cooperative_groups.h>
#include "helper.hpp"
#include "constants.hpp"
#include "../expl_comp_strat/common.hpp"
#include "data_types.h"
#include <cuda_runtime_api.h>

using namespace cooperative_groups;

#ifdef __CUDACC__
#define __fhd__  __forceinline__ __host__ __device__
#define __fd__   __forceinline__ __device__
#else
#define __fhd__ inline
#define __fd__  inline
#endif

// Annoyingly, CUDA - upto and including version 9.2 - provide atomic
// operation wrappers for unsigned int and unsigned long long int, but
// not for the in-between type of unsigned long int. So - we
// have to make our own. Don't worry about the condition checks -
// they are optimied away at compile time and the result is basically
// a single PTX instruction (provided the value is available)
__fd__ unsigned long int atomicAdd(unsigned long int *address, unsigned long int val)
{
	if (sizeof (unsigned long int) == sizeof(unsigned long long int)) {
		return atomicAdd(reinterpret_cast<unsigned long long int*>(address), val);
	}
	else if (sizeof (unsigned long int) == sizeof(unsigned int)) {
		return  atomicAdd(reinterpret_cast<unsigned int*>(address), val);
	}
	else return 0;
}

namespace cuda{


/*

__global__
void print() {
    // who am I?
    int wid = global_warp_id();
    int lid = warp_local_thread_id();
    printf(" Global Warp: %d Local Warp: %d \n", wid, lid);
}

__device__
int atomicAggInc(int *ctr) {
    auto g = coalesced_threads();
    int warp_res;
    if(g.thread_rank() == 0)
        warp_res = atomicAdd(ctr, g.size());
    return g.shfl(warp_res, 0) + g.thread_rank();
}

__global__
void filter_k(int *dst, int *nres, const int *src, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i >= n)
        return;
    if(src[i] > 0)
        dst[atomicAggInc(nres)] = src[i];
}

__inline__ __device__
int warpReduceSum(int val) {
    for (int offset = warp_size / 2; offset > 0; offset /= 2)
        val += __shfl_down(val, offset);
    return val;
}

__inline __device__
int reduce_sum(thread_group t_group, int *temp, int value){

    int lane = t_group.thread_rank();

    // Each iteration halves the number of active threads
    // Each thread adds its partial sum[i] to sum[lane + i]
    for(size_t i = t_group.size() / 2; i > 0; i /= 2){

        temp[lane] = value;
        t_group.sync(); // wait for all threads to store
        if(lane < i )
            value += temp[lane + i];
        t_group.sync(); // wait for all threads to load
    }

    return value;
}

__inline__ __device__
int blockReduceSum(int val) {

    static __shared__ int shared[32]; // Shared mem for 32 partial sums
    int lane = warp_local_thread_id();
    int wid = block_local_warp_id();

    val = cuda::warpReduceSum(val);     // Each warp performs partial reduction

    if (lane == 0) shared[wid] = val; // Write reduced value to shared memory

    __syncthreads();              // Wait for all partial reductions

    //read from shared memory only if that warp existed
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

    if (wid==0) val = cuda::warpReduceSum(val); //Final reduce within first warp

    return val;
}
*/
// The following would be better placed in a proper bit vector class having per-element proxies.

template <typename Integer>
__fhd__ constexpr Integer get_bit_range(
    const Integer& value,
    unsigned       start_bit,
    unsigned       num_bits) noexcept
{
    return (value >> start_bit) & ((1 << num_bits) - 1);
}

template <unsigned LogBitsPerValue, typename Index>
__fhd__ bit_container_t get_bit_resolution_element(
    bit_container_t  bit_container,
    Index            element_index_within_container)
{
    enum {
        bits_per_value = 1 << LogBitsPerValue,
        elements_per_container = bits_per_bit_container / bits_per_value,
    };
    // TODO: This can be improved with some PTX intrinsics
    return get_bit_range<Index>(bit_container,
        bits_per_value * element_index_within_container, bits_per_value);

}

template <unsigned LogBitsPerValue, typename Index>
__fhd__ bit_container_t get_bit_resolution_element(
    const bit_container_t* __restrict__ data,
    Index                               element_index)
{
    enum {
        bits_per_value = 1 << LogBitsPerValue,
        elements_per_container = bits_per_bit_container / bits_per_value
    };
    auto index_of_container = element_index / elements_per_container;
    auto index_within_container = element_index % elements_per_container;
    auto bit_container = data[index_of_container];
    return get_bit_resolution_element<LogBitsPerValue, Index>(bit_container, index_within_container);
}



} // namespace cuda

#undef __fhd__
#undef __fd__
