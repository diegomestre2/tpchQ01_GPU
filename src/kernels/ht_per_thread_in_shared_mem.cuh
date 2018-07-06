#pragma once

#include "kernel.hpp"
#include "constants.hpp"
#include "data_types.h"
#include "bit_operations.h"

#ifdef __CUDACC__
#define __fhd__  __forceinline__ __host__ __device__
#define __fd__   __forceinline__ __device__
#else
#define __fhd__ inline
#define __fd__  inline
#endif

namespace cuda {

enum {
    max_threads_per_block_for_per_thread_shared_mem = 128
};

template <unsigned NumThreadsPerBlock = max_threads_per_block_for_per_thread_shared_mem>
    // this must be a nice number w.r.t. the number of shared memory banks, and not too high, otherwise
    // NVCC will complain about too much shared memory use!
__global__
void thread_in_shared_mem_ht_tpchQ01(
    sum_quantity_t*          __restrict__ sum_quantity,
    sum_base_price_t*        __restrict__ sum_base_price,
    sum_discounted_price_t*  __restrict__ sum_discounted_price,
    sum_charge_t*            __restrict__ sum_charge,
    sum_discount_t*          __restrict__ sum_discount,
    cardinality_t*           __restrict__ record_count,
    const ship_date_t*       __restrict__ shipdate,
    const discount_t*        __restrict__ discount,
    const extended_price_t*  __restrict__ extended_price,
    const tax_t*             __restrict__ tax,
    const quantity_t*        __restrict__ quantity,
    const return_flag_t*     __restrict__ return_flag,
    const line_status_t*     __restrict__ line_status,
    cardinality_t                         num_tuples)
{
    __shared__ sum_quantity_t         sums_of_quantity         [num_potential_groups * NumThreadsPerBlock];
    __shared__ sum_base_price_t       sums_of_base_price       [num_potential_groups * NumThreadsPerBlock];
    __shared__ sum_discounted_price_t sums_of_discounted_price [num_potential_groups * NumThreadsPerBlock];
    __shared__ sum_charge_t           sums_of_charge           [num_potential_groups * NumThreadsPerBlock];
    __shared__ sum_discount_t         sums_of_discount         [num_potential_groups * NumThreadsPerBlock];
    __shared__ cardinality_t          record_counts            [num_potential_groups * NumThreadsPerBlock];
        // The layout of each of these arrays is such, that all of a warp's data within is located
        // in two shared memory banks (for 64-bit data) or one back (for 32-bit data). This can
        // be possibly improved by splitting the 64-bit values into pairs of 32-bit values for
        // writing, but: 1. I don't have the time right now. 2. Not sure how much of a benefit
        // this will be.
        //
        // Also, this can be switched to dynamic shared memory, but I don't have the time right now

    cardinality_t stride = (blockDim.x * gridDim.x); //Grid-Stride
    auto warp_index = threadIdx.x / warp_size;
    auto lane_index = threadIdx.x % warp_size;
    auto thread_base_offset_in_sums = lane_index + warp_index * num_potential_groups * warp_size;
            // this relies heavily on the identity of warp_size and the number of shared memory banks

    auto thread_sums_of_quantity          = sums_of_quantity         + thread_base_offset_in_sums;
    auto thread_sums_of_base_price        = sums_of_base_price       + thread_base_offset_in_sums;
    auto thread_sums_of_charge            = sums_of_charge           + thread_base_offset_in_sums;
    auto thread_sums_of_discounted_price  = sums_of_discounted_price + thread_base_offset_in_sums;
    auto thread_sums_of_discount          = sums_of_discount         + thread_base_offset_in_sums;
    auto thread_record_counts             = record_counts            + thread_base_offset_in_sums;

    // initialize the shared memory

    for(int group_index = 0; group_index < num_potential_groups; group_index ++) {
        thread_sums_of_quantity        [group_index * warp_size] = 0;
        thread_sums_of_base_price      [group_index * warp_size] = 0;
        thread_sums_of_charge          [group_index * warp_size] = 0;
        thread_sums_of_discounted_price[group_index * warp_size] = 0;
        thread_sums_of_discount        [group_index * warp_size] = 0;
        thread_record_counts           [group_index * warp_size] = 0;
    }

    for(cardinality_t i = (blockIdx.x * blockDim.x + threadIdx.x); i < num_tuples; i += stride) {

        if (shipdate[i] <= threshold_ship_date) {
            // TODO: Some of these calculations could work on uint32_t
            auto line_quantity         = quantity[i];
            auto line_discount         = discount[i];
            auto line_price            = extended_price[i];
            auto line_discount_factor  = Decimal64::ToValue(1, 0) - line_discount;
            auto line_discounted_price = Decimal64::Mul(line_discount_factor, line_price);
            auto line_tax_factor       = tax[i] + Decimal64::ToValue(1, 0);
            auto line_charge           = Decimal64::Mul(line_discounted_price, line_tax_factor);
            auto line_return_flag      = return_flag[i];
            auto line_status_          = line_status[i];

            int group_index =
                (encode_return_flag(line_return_flag) << line_status_bits) + encode_line_status(line_status_);

            thread_sums_of_quantity        [group_index * warp_size] += line_quantity;
            thread_sums_of_base_price      [group_index * warp_size] += line_price;
            thread_sums_of_charge          [group_index * warp_size] += line_charge;
            thread_sums_of_discounted_price[group_index * warp_size] += line_discounted_price;
            thread_sums_of_discount        [group_index * warp_size] += line_discount;
            thread_record_counts           [group_index * warp_size] ++;
        }
    }

    // final aggregation

    // These manual casts are really unbecoming. We need a wrapper...
    #pragma unroll
    for (int group_index = 0; group_index < num_potential_groups; ++group_index) {
        atomicAdd( & sum_quantity        [group_index], thread_sums_of_quantity        [group_index * warp_size]);
        atomicAdd( & sum_base_price      [group_index], thread_sums_of_base_price      [group_index * warp_size]);
        atomicAdd( & sum_charge          [group_index], thread_sums_of_charge          [group_index * warp_size]);
        atomicAdd( & sum_discounted_price[group_index], thread_sums_of_discounted_price[group_index * warp_size]);
        atomicAdd( & sum_discount        [group_index], thread_sums_of_discount        [group_index * warp_size]);
        atomicAdd( & record_count        [group_index], thread_record_counts           [group_index * warp_size]);
    }
}

template <unsigned NumThreadsPerBlock = max_threads_per_block_for_per_thread_shared_mem>
    // this must be a nice number w.r.t. the number of shared memory banks, and not too high, otherwise
    // NVCC will complain about too much shared memory use!
__global__
void thread_in_shared_mem_ht_tpchQ01_compressed(
    sum_quantity_t*                      __restrict__ sum_quantity,
    sum_base_price_t*                    __restrict__ sum_base_price,
    sum_discounted_price_t*              __restrict__ sum_discounted_price,
    sum_charge_t*                        __restrict__ sum_charge,
    sum_discount_t*                      __restrict__ sum_discount,
    cardinality_t*                       __restrict__ record_count,
    const compressed::ship_date_t*       __restrict__ shipdate,
    const compressed::discount_t*        __restrict__ discount,
    const compressed::extended_price_t*  __restrict__ extended_price,
    const compressed::tax_t*             __restrict__ tax,
    const compressed::quantity_t*        __restrict__ quantity,
    const bit_container_t*               __restrict__ return_flag,
    const bit_container_t*               __restrict__ line_status,
    cardinality_t                                     num_tuples)
{
    __shared__ sum_quantity_t         sums_of_quantity         [num_potential_groups * NumThreadsPerBlock];
    __shared__ sum_base_price_t       sums_of_base_price       [num_potential_groups * NumThreadsPerBlock];
    __shared__ sum_discounted_price_t sums_of_discounted_price [num_potential_groups * NumThreadsPerBlock];
    __shared__ sum_charge_t           sums_of_charge           [num_potential_groups * NumThreadsPerBlock];
    __shared__ sum_discount_t         sums_of_discount         [num_potential_groups * NumThreadsPerBlock];
    __shared__ cardinality_t          record_counts            [num_potential_groups * NumThreadsPerBlock];
        // The layout of each of these arrays is such, that all of a warp's data within is located
        // in two shared memory banks (for 64-bit data) or one back (for 32-bit data). This can
        // be possibly improved by splitting the 64-bit values into pairs of 32-bit values for
        // writing, but: 1. I don't have the time right now. 2. Not sure how much of a benefit
        // this will be.
        //
        // Also, this can be switched to dynamic shared memory, but I don't have the time right now

    auto warp_index = threadIdx.x / warp_size;
    auto lane_index = threadIdx.x % warp_size;
    auto thread_base_offset_in_sums = lane_index + warp_index * num_potential_groups * warp_size;
            // this relies heavily on the identity of warp_size and the number of shared memory banks

    auto thread_sums_of_quantity          = sums_of_quantity         + thread_base_offset_in_sums;
    auto thread_sums_of_base_price        = sums_of_base_price       + thread_base_offset_in_sums;
    auto thread_sums_of_charge            = sums_of_charge           + thread_base_offset_in_sums;
    auto thread_sums_of_discounted_price  = sums_of_discounted_price + thread_base_offset_in_sums;
    auto thread_sums_of_discount          = sums_of_discount         + thread_base_offset_in_sums;
    auto thread_record_counts             = record_counts            + thread_base_offset_in_sums;

    // initialize the shared memory

    for(int group_index = 0; group_index < num_potential_groups; group_index ++) {
        thread_sums_of_quantity        [group_index * warp_size] = 0;
        thread_sums_of_base_price      [group_index * warp_size] = 0;
        thread_sums_of_charge          [group_index * warp_size] = 0;
        thread_sums_of_discounted_price[group_index * warp_size] = 0;
        thread_sums_of_discount        [group_index * warp_size] = 0;
        thread_record_counts           [group_index * warp_size] = 0;
    }

    cardinality_t stride = (blockDim.x * gridDim.x); //Grid-Stride
    for(cardinality_t i = (blockIdx.x * blockDim.x + threadIdx.x); i < num_tuples; i += stride) {
        if (shipdate[i] <= compressed_threshold_ship_date) {
            // TODO: Some of these calculations could work on uint32_t
            auto line_quantity         = quantity[i];
            auto line_discount         = discount[i];
            auto line_price            = extended_price[i];
            auto line_discount_factor  = Decimal64::ToValue(1, 0) - line_discount;
            auto line_discounted_price = Decimal64::Mul(line_discount_factor, line_price);
            auto line_tax_factor       = tax[i] + Decimal64::ToValue(1, 0);
            auto line_charge           = Decimal64::Mul(line_discounted_price, line_tax_factor);
            auto line_return_flag      = get_bit_resolution_element<log_return_flag_bits, cardinality_t>(return_flag, i);
            auto line_status_          = get_bit_resolution_element<log_line_status_bits, cardinality_t>(line_status, i);

            int group_index = (line_return_flag << line_status_bits) + line_status_;

            thread_sums_of_quantity        [group_index * warp_size] += line_quantity;
            thread_sums_of_base_price      [group_index * warp_size] += line_price;
            thread_sums_of_charge          [group_index * warp_size] += line_charge;
            thread_sums_of_discounted_price[group_index * warp_size] += line_discounted_price;
            thread_sums_of_discount        [group_index * warp_size] += line_discount;
            thread_record_counts           [group_index * warp_size] ++;
        }
    }

    // final aggregation

    // These manual casts are really unbecoming. We need a wrapper...
    #pragma unroll
    for (int group_index = 0; group_index < num_potential_groups; ++group_index) {
        atomicAdd( & sum_quantity        [group_index], thread_sums_of_quantity        [group_index * warp_size]);
        atomicAdd( & sum_base_price      [group_index], thread_sums_of_base_price      [group_index * warp_size]);
        atomicAdd( & sum_charge          [group_index], thread_sums_of_charge          [group_index * warp_size]);
        atomicAdd( & sum_discounted_price[group_index], thread_sums_of_discounted_price[group_index * warp_size]);
        atomicAdd( & sum_discount        [group_index], thread_sums_of_discount        [group_index * warp_size]);
        atomicAdd( & record_count        [group_index], thread_record_counts           [group_index * warp_size]);
    }

}

template <unsigned NumThreadsPerBlock = max_threads_per_block_for_per_thread_shared_mem>
    // this must be a nice number w.r.t. the number of shared memory banks, and not too high, otherwise
    // NVCC will complain about too much shared memory use!
__global__
void thread_in_shared_mem_ht_tpchQ01_pushdown_compressed(
    sum_quantity_t*                      __restrict__ sum_quantity,
    sum_base_price_t*                    __restrict__ sum_base_price,
    sum_discounted_price_t*              __restrict__ sum_discounted_price,
    sum_charge_t*                        __restrict__ sum_charge,
    sum_discount_t*                      __restrict__ sum_discount,
    cardinality_t*                       __restrict__ record_count,
    const uint8_t*                       __restrict__ shipdate,
    const compressed::discount_t*        __restrict__ discount,
    const compressed::extended_price_t*  __restrict__ extended_price,
    const compressed::tax_t*             __restrict__ tax,
    const compressed::quantity_t*        __restrict__ quantity,
    const bit_container_t*               __restrict__ return_flag,
    const bit_container_t*               __restrict__ line_status,
    cardinality_t                                     num_tuples)
 
{
    __shared__ sum_quantity_t         sums_of_quantity         [num_potential_groups * NumThreadsPerBlock];
    __shared__ sum_base_price_t       sums_of_base_price       [num_potential_groups * NumThreadsPerBlock];
    __shared__ sum_discounted_price_t sums_of_discounted_price [num_potential_groups * NumThreadsPerBlock];
    __shared__ sum_charge_t           sums_of_charge           [num_potential_groups * NumThreadsPerBlock];
    __shared__ sum_discount_t         sums_of_discount         [num_potential_groups * NumThreadsPerBlock];
    __shared__ cardinality_t          record_counts            [num_potential_groups * NumThreadsPerBlock];
        // The layout of each of these arrays is such, that all of a warp's data within is located
        // in two shared memory banks (for 64-bit data) or one back (for 32-bit data). This can
        // be possibly improved by splitting the 64-bit values into pairs of 32-bit values for
        // writing, but: 1. I don't have the time right now. 2. Not sure how much of a benefit
        // this will be.
        //
        // Also, this can be switched to dynamic shared memory, but I don't have the time right now

    auto warp_index = threadIdx.x / warp_size;
    auto lane_index = threadIdx.x % warp_size;
    auto thread_base_offset_in_sums = lane_index + warp_index * num_potential_groups * warp_size;
            // this relies heavily on the identity of warp_size and the number of shared memory banks

    auto thread_sums_of_quantity          = sums_of_quantity         + thread_base_offset_in_sums;
    auto thread_sums_of_base_price        = sums_of_base_price       + thread_base_offset_in_sums;
    auto thread_sums_of_charge            = sums_of_charge           + thread_base_offset_in_sums;
    auto thread_sums_of_discounted_price  = sums_of_discounted_price + thread_base_offset_in_sums;
    auto thread_sums_of_discount          = sums_of_discount         + thread_base_offset_in_sums;
    auto thread_record_counts             = record_counts            + thread_base_offset_in_sums;

    // initialize the shared memory

    for(int group_index = 0; group_index < num_potential_groups; group_index ++) {
        thread_sums_of_quantity        [group_index * warp_size] = 0;
        thread_sums_of_base_price      [group_index * warp_size] = 0;
        thread_sums_of_charge          [group_index * warp_size] = 0;
        thread_sums_of_discounted_price[group_index * warp_size] = 0;
        thread_sums_of_discount        [group_index * warp_size] = 0;
        thread_record_counts           [group_index * warp_size] = 0;
    }

    constexpr uint8_t SHIPDATE_MASK[] = { 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80 };

    cardinality_t stride = (blockDim.x * gridDim.x); //Grid-Stride
    for(cardinality_t i = (blockIdx.x * blockDim.x + threadIdx.x); i < num_tuples; i += stride) {
        if (shipdate[i / 8] & SHIPDATE_MASK[i % 8]) {
            // TODO: Some of these calculations could work on uint32_t
            auto line_quantity         = quantity[i];
            auto line_discount         = discount[i];
            auto line_price            = extended_price[i];
            auto line_discount_factor  = Decimal64::ToValue(1, 0) - line_discount;
            auto line_discounted_price = Decimal64::Mul(line_discount_factor, line_price);
            auto line_tax_factor       = tax[i] + Decimal64::ToValue(1, 0);
            auto line_charge           = Decimal64::Mul(line_discounted_price, line_tax_factor);
            auto line_return_flag      = get_bit_resolution_element<log_return_flag_bits, cardinality_t>(return_flag, i);
            auto line_status_          = get_bit_resolution_element<log_line_status_bits, cardinality_t>(line_status, i);

            int group_index = (line_return_flag << line_status_bits) + line_status_;

            thread_sums_of_quantity        [group_index * warp_size] += line_quantity;
            thread_sums_of_base_price      [group_index * warp_size] += line_price;
            thread_sums_of_charge          [group_index * warp_size] += line_charge;
            thread_sums_of_discounted_price[group_index * warp_size] += line_discounted_price;
            thread_sums_of_discount        [group_index * warp_size] += line_discount;
            thread_record_counts           [group_index * warp_size] ++;
        }
    }

    // final aggregation

    // These manual casts are really unbecoming. We need a wrapper...
    #pragma unroll
    for (int group_index = 0; group_index < num_potential_groups; ++group_index) {
        atomicAdd( & sum_quantity        [group_index], thread_sums_of_quantity        [group_index * warp_size]);
        atomicAdd( & sum_base_price      [group_index], thread_sums_of_base_price      [group_index * warp_size]);
        atomicAdd( & sum_charge          [group_index], thread_sums_of_charge          [group_index * warp_size]);
        atomicAdd( & sum_discounted_price[group_index], thread_sums_of_discounted_price[group_index * warp_size]);
        atomicAdd( & sum_discount        [group_index], thread_sums_of_discount        [group_index * warp_size]);
        atomicAdd( & record_count        [group_index], thread_record_counts           [group_index * warp_size]);
    }
}


} // namespace cuda
