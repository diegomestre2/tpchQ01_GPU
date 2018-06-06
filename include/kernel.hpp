#pragma once

#include <stdio.h>
#include <cooperative_groups.h>
#include "helper.hpp"
#include "dates.hpp"
#include "constants.hpp"
#include "../expl_comp_strat/common.hpp"

using namespace cooperative_groups;
using u64_t = unsigned long long int;

using SHIPDATE_TYPE = int;
using DISCOUNT_TYPE = int64_t;
using EXTENDEDPRICE_TYPE = int64_t;
using TAX_TYPE = int64_t;
using QUANTITY_TYPE = int64_t;
using RETURNFLAG_TYPE = char;
using LINESTATUS_TYPE = char;

using SHIPDATE_TYPE_SMALL = int;
using DISCOUNT_TYPE_SMALL = int64_t;
using EXTENDEDPRICE_TYPE_SMALL = int64_t;
using TAX_TYPE_SMALL = int64_t;
using QUANTITY_TYPE_SMALL = int64_t;
using RETURNFLAG_TYPE_SMALL = uint8_t;
using LINESTATUS_TYPE_SMALL = uint8_t;

#define SHIPDATE_MIN 727563

namespace cuda{




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
    __device__
    int thread_sum(int *input, int n){

        int sum = 0;
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;

        for(; index < n; index += stride){

            int in = ((int*)input)[index];
            sum += in;
        }

        return sum;
    }


    __global__ 
    void deviceReduceKernel(int *in, int* out, int N) {
        int sum = 0;
        //reduce multiple elements per thread
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
            i < N; 
            i += blockDim.x * gridDim.x) {
                sum += in[i];
        }
        printf("%d\n", sum);
        sum = cuda::blockReduceSum(sum);
        if (block_local_thread_id() == 0){
            printf("%d %d\n",blockIdx.x, sum);
            out[blockIdx.x] = sum;
        }
    }

    // __device__
    // int thread_sum(int *input, AggrHashTableKey *temp, int n){

    //     int sum = 0;
    //     int index = blockIdx.x * blockDim.x + threadIdx.x;
    //     int stride = blockDim.x * gridDim.x;

    //     for(; index < n / 4; index += stride){

    //             int4 in = ((int4*)input)[index];
    //             sum += in.x + in.y + in.z + in.w;
    //     }

    //     return sum;
    // }

    __inline__ __device__ 
    idx_t magic_hash(char rf, char ls) {
        return (((rf - 'A')) - (ls - 'F'));
    }
}


// A,F,37734107,56586554400.73,53758257134.8700,55909065222.827692,25.522005853257337,38273.12973462185,0.049985295838398135,1478493
// N,F,991417,1487504710.38,1413082168.0541,1469649223.194375,25.51647192052298,38284.46776084826,0.050093426674216325,38854
// N,O,74476040,111701729697.74,106118230307.6056,110367043872.497010,25.502226769584993,38249.117988908765,0.049996586053704065,2920374
// R,F,37719753,56568041380.90,53741292684.6040,55889619119.831932,25.50579361269077,38250.85462609964,0.050009405830126356,1478870


/*
    65, 70 -> 0
    78, 70 -> 1
    78, 79 -> 2
    82, 70 -> 3
    */