#include <stdio.h>
#include "helper.hpp"
#include "dates.hpp"
#include "../expl_comp_strat/common.hpp"
#include <cooperative_groups.h>

using namespace cooperative_groups;
using u64_t = unsigned long long int;
namespace cuda{

#define magic_hash(rf, ls) ((rf - 'A') * 16 + (ls - 'F'))

    __global__
    void print() {
        // who am I?
        int wid = global_warp_id();
        int lid = warp_local_thread_id();
        printf(" Global Warp: %d Local Warp: %d \n", wid, lid);
    }
    /*__device__ 
    int atomicAggInc_primitives(int *ctr) {
        unsigned int active = __activemask();
        int leader = __ffs(active) - 1;
        int change = __popc(active);
        unsigned int rank = __popc(active & __lanemask_lt());
        int warp_res;
        if(rank == 0)
            warp_res = atomicAdd(ctr, change);
        warp_res = __shfl_sync(active, warp_res, leader);
        return warp_res + rank;
    }*/

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
    void naive_tpchQ01(int *shipdate, int64_t *discount, int64_t *extendedprice, int64_t *tax, 
        char *returnflag, char *linestatus, int64_t *quantity, AggrHashTable *aggregations, size_t cardinality){

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
#if 0
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

    __global__
    void thread_local_tpchQ01(int *shipdate, int *discount, int *extendedprice, int *tax, 
        char *returnflag, char *linestatus, int *quantity, AggrHashTable *aggregations, size_t cardinality) {

        __shared__ u64_t t_quant[18];
        __shared__ u64_t t_base[18];
        __shared__ u64_t t_charge[18];
        __shared__ u64_t t_disc_price[18];
        __shared__ u64_t t_disc[18];
        __shared__ u64_t t_count[18];
        //if(threadIdx.x == 0)
        //    for(int i = 0; i!= 18; ++i)hashtable[i] = {0};
        //__syncthreads();
        int i = 32 * blockIdx.x * blockDim.x + threadIdx.x;
        int end = min((int)cardinality, i + 32);

        for(; i < end; i++) {

            if (i < cardinality && shipdate[i] <= 729999){//todate_(2, 9, 1998)) {
                const auto disc = discount[i];
                const auto price = extendedprice[i];
                const auto disc_1 = Decimal64::ToValue(1, 0) - disc;
                const auto tax_1 = tax[i] + Decimal64::ToValue(1, 0);
                const auto disc_price = Decimal64::Mul(disc_1, price);
                const auto charge = Decimal64::Mul(disc_price, tax_1);
                const idx_t idx = magic_hash(returnflag[i], linestatus[i]);
                
                //if(idx < 0 || idx > 17)
                    {printf(" idx%d l%d r%d i%d \n",idx,linestatus[i], returnflag[i], i);}
                t_quant[idx] += (u64_t) quantity[i];
                t_base[idx] += price;
                t_charge[idx] += charge;

                t_disc_price[idx] += disc_price;

                t_disc[idx] += disc;
                t_count[idx] += 1;
            }

        }
        // __syncthreads();
    }
#endif
#if 0
    __global__
    void thread_local_tpchQ01_old(int *shipdate, int *discount, int *extendedprice, int *tax, 
        char *returnflag, char *linestatus, int *quantity, AggrHashTable *aggregations, size_t cardinality) {

        __shared__ AggrHashTableKey hashtable[18];
        //if(threadIdx.x == 0)
        //    for(int i = 0; i!= 18; ++i)hashtable[i] = {0};
        //__syncthreads();
        int i = blockIdx.x * blockDim.x + threadIdx.x;

        if(i >= cardinality)
            return;

        if (i < cardinality && shipdate[i] <= 729999){//todate_(2, 9, 1998)) {
            const auto disc = discount[i];
            const auto price = extendedprice[i];
            const auto disc_1 = Decimal64::ToValue(1, 0) - disc;
            const auto tax_1 = tax[i] + Decimal64::ToValue(1, 0);
            const auto disc_price = Decimal64::Mul(disc_1, price);
            const auto charge = Decimal64::Mul(disc_price, tax_1);
            const idx_t idx = magic_hash(linestatus[i], returnflag[i]);
            (hashtable[idx].sum_quantity) += quantity[i];
            (hashtable[idx].sum_base_price) += (u64_t)price;
            auto old = (hashtable[idx].sum_charge) += charge;
            //if (old + charge < charge) {
            //    (hashtable[idx].sum_charge) +=  1;
            //}

            auto old_2 = (hashtable[idx].sum_disc_price) += disc_price;
            //if (old_2 + disc_price < disc_price) {
            //    (hashtable[idx].sum_disc_price) += 1;
            //}
            (hashtable[idx].sum_disc) += (u64_t)disc;
            (hashtable[idx].count) += (u64_t)1;
        }
        // __syncthreads();

    //}
}
#endif

}


/*
# A|F|37734107.0|56586554400.73|5375825713487.0|559090652228276.92|1478493
# N|F|991417.0|1487504710.38|141308216805.41|14696492231943.75|38854
# N|O|76633518.0|114935210409.19|10918959189747.20|1135610242630137.82|3004998
# R|F|37719753.0|56568041380.90|5374129268460.40|558896191198319.32|1478870

# A|F | 18862170.0 | 28284439012.17 | 2687056106082.24 | 279448898345577.64|1478493
# N|F | 495416.0 | 741540899.0 | 70453555245.0 | 7324666930399.20|38854
# N|O | 38311390.0 | 57463978052.78 | 5459204398489.66 | 567767112994873.0|3004998
# R|F | 18875614.0 | 28301392770.17 | 2688523518267.66 | 279609427976298.35|1478870
*/

/*
    65, 70 -> 0
    78, 70 -> 1
    78, 79 -> 2
    82, 70 -> 3
    */