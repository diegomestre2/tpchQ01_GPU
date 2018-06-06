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

    __global__
    void naive_tpchQ01(SHIPDATE_TYPE *shipdate, DISCOUNT_TYPE *discount, EXTENDEDPRICE_TYPE *extendedprice, TAX_TYPE *tax, 
        RETURNFLAG_TYPE *returnflag, LINESTATUS_TYPE *linestatus, QUANTITY_TYPE *quantity, AggrHashTable *aggregations, size_t cardinality){

        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < cardinality && shipdate[i] <= 729999){//todate_(2, 9, 1998)) {
            const auto disc = discount[i];
            const auto price = extendedprice[i];
            const auto disc_1 = Decimal64::ToValue(1, 0) - disc;
            const auto tax_1 = tax[i] + Decimal64::ToValue(1, 0);
            const auto disc_price = Decimal64::Mul(disc_1, price);
            const auto charge = Decimal64::Mul(disc_price, tax_1);
            const idx_t idx = returnflag[i] << 8 | linestatus[i];
            atomicAdd((u64_t*)&(aggregations[idx].sum_quantity), (u64_t) quantity[i]);
            atomicAdd((u64_t*)&(aggregations[idx].sum_base_price), (u64_t)price);
            auto old = atomicAdd((u64_t*)&(aggregations[idx].sum_charge), charge);
            if (old + charge < charge) {
                atomicAdd((u64_t*)&(aggregations[idx].sum_charge) + 1, 1);
            }

            auto old_2 = atomicAdd((u64_t*)&(aggregations[idx].sum_disc_price), disc_price);
            if (old_2 + disc_price < disc_price) {
                atomicAdd((u64_t*)&(aggregations[idx].sum_disc_price) + 1, 1);
            }
            atomicAdd((u64_t*)&(aggregations[idx].sum_disc), (u64_t)disc);
            atomicAdd((u64_t*)&(aggregations[idx].count), (u64_t)1);
            
        }
    }

    __device__
    int thread_sum(int *input, AggrHashTableKey *temp, int n){

        int sum = 0;
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;

        for(; index < n / 4; index += stride){

                int4 in = ((int4*)input)[index];
                sum += in.x + in.y + in.z + in.w;
        }

        return sum;
    }

    __inline__ __device__ 
    idx_t magic_hash(char rf, char ls) {
        return (((rf - 'A')) - (ls - 'F'));
    }

    __global__
    void global_ht_tpchQ01(
        SHIPDATE_TYPE *shipdate,
        DISCOUNT_TYPE *discount,
        EXTENDEDPRICE_TYPE *extendedprice,
        TAX_TYPE *tax,
        RETURNFLAG_TYPE *returnflag,
        LINESTATUS_TYPE *linestatus,
        QUANTITY_TYPE *quantity,
        AggrHashTable *aggregations,
        u64_t cardinality) {

        u64_t i = VALUES_PER_THREAD * (blockIdx.x * blockDim.x + threadIdx.x);
        u64_t end = min((u64_t)cardinality, i + VALUES_PER_THREAD);
        for(; i < end; ++i) {
            if (shipdate[i] <= 729999 - SHIPDATE_MIN) {
                const int disc = discount[i];
                const int price = extendedprice[i];
                const int disc_1 = Decimal64::ToValue(1, 0) - disc;
                const int tax_1 = tax[i] + Decimal64::ToValue(1, 0);
                const int disc_price = Decimal64::Mul(disc_1, price);
                const int charge = Decimal64::Mul(disc_price, tax_1);
                const idx_t idx = magic_hash(returnflag[i], linestatus[i]);
                
                atomicAdd(&aggregations[idx].sum_quantity, (u64_t) quantity[i] * 100);
                atomicAdd(&aggregations[idx].sum_base_price, (u64_t) price);
                atomicAdd(&aggregations[idx].sum_charge, (u64_t) charge);
                atomicAdd(&aggregations[idx].sum_disc_price, (u64_t) disc_price);
                atomicAdd(&aggregations[idx].sum_disc, (u64_t) disc);
                atomicAdd(&aggregations[idx].count, (u64_t) 1);
            }
        }
    }

    __global__
    void thread_local_tpchQ01(
        SHIPDATE_TYPE *shipdate,
        DISCOUNT_TYPE *discount,
        EXTENDEDPRICE_TYPE *extendedprice,
        TAX_TYPE *tax,
        RETURNFLAG_TYPE *returnflag,
        LINESTATUS_TYPE *linestatus,
        QUANTITY_TYPE *quantity,
        AggrHashTable *aggregations,
        u64_t cardinality) {

        constexpr uint8_t RETURNFLAG_MASK[] = { 0x03, 0x0C, 0x30, 0xC0 };
        constexpr uint8_t LINESTATUS_MASK[] = { 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80 };

        constexpr size_t N = 8;
        AggrHashTableLocal agg[N];
        memset(agg, 0, sizeof(AggrHashTableLocal) * N);

        u64_t i = VALUES_PER_THREAD * (blockIdx.x * blockDim.x + threadIdx.x);
        u64_t end = min((u64_t)cardinality, i + VALUES_PER_THREAD);
        for(; i < end; ++i) {
            if (shipdate[i] <= 729999 - SHIPDATE_MIN) {
                const int disc = discount[i];
                const int price = extendedprice[i];
                const int disc_1 = Decimal64::ToValue(1, 0) - disc;
                const int tax_1 = tax[i] + Decimal64::ToValue(1, 0);
                const int disc_price = Decimal64::Mul(disc_1, price);
                const int charge = Decimal64::Mul(disc_price, tax_1);
                const uint8_t rflag = (returnflag[i / 4] & RETURNFLAG_MASK[i % 4]) >> (2 * (i % 4));
                const uint8_t lstatus = (linestatus[i / 8] & LINESTATUS_MASK[i % 8]) >> (i % 8);
                const uint8_t idx = rflag + 4 * lstatus;
                
                agg[idx].sum_quantity   += quantity[i] * 100;
                agg[idx].sum_base_price += price;
                agg[idx].sum_charge     += charge;
                agg[idx].sum_disc_price += disc_price;
                agg[idx].sum_disc       += disc;
                agg[idx].count          += 1;
            }
        }
        // final aggregation
        for (i = 0; i < N; ++i) {
            if (!agg[i].count) {
                continue;
            }
            atomicAdd(&aggregations[i].sum_quantity, (u64_t) agg[i].sum_quantity);
            atomicAdd(&aggregations[i].sum_base_price, (u64_t) agg[i].sum_base_price);
            if (atomicAdd(&aggregations[i].sum_charge, (u64_t) agg[i].sum_charge) < agg[i].sum_charge) {
                atomicAdd(&aggregations[i].sum_charge_hi, 1);
            }
            if (atomicAdd(&aggregations[i].sum_disc_price, (u64_t) agg[i].sum_disc_price) < agg[i].sum_disc_price) {
                atomicAdd(&aggregations[i].sum_disc_price_hi, 1);
            }
            atomicAdd(&aggregations[i].sum_disc, (u64_t) agg[i].sum_disc);
            atomicAdd(&aggregations[i].count, (u64_t) agg[i].count);
        }
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