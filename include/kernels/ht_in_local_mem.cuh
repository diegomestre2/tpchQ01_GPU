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

__global__
void in_local_mem_ht_tpchQ01(
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
    sum_quantity_t         thread_sum_quantity         [num_potential_groups] = { 0 };
    sum_base_price_t       thread_sum_base_price       [num_potential_groups] = { 0 };
    sum_discounted_price_t thread_sum_discounted_price [num_potential_groups] = { 0 };
    sum_charge_t           thread_sum_charge           [num_potential_groups] = { 0 };
    sum_discount_t         thread_sum_discount         [num_potential_groups] = { 0 };
    cardinality_t          thread_record_count         [num_potential_groups] = { 0 };


    cardinality_t stride = (blockDim.x * gridDim.x); //Grid-Stride
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

            thread_sum_quantity        [group_index] += line_quantity;
            thread_sum_base_price      [group_index] += line_price;
            thread_sum_charge          [group_index] += line_charge;
            thread_sum_discounted_price[group_index] += line_discounted_price;
            thread_sum_discount        [group_index] += line_discount;
            thread_record_count        [group_index] ++;
        }
    }

    // final aggregation

    // These manual casts are really unbecoming. We need a wrapper...
    #pragma unroll
    for (int group_index = 0; group_index < num_potential_groups; ++group_index) {
        atomicAdd( (unsigned long long* ) & sum_quantity        [group_index], thread_sum_quantity        [group_index]);
        atomicAdd( (unsigned long long* ) & sum_base_price      [group_index], thread_sum_base_price      [group_index]);
        atomicAdd( (unsigned long long* ) & sum_charge          [group_index], thread_sum_charge          [group_index]);
        atomicAdd( (unsigned long long* ) & sum_discounted_price[group_index], thread_sum_discounted_price[group_index]);
        atomicAdd( (unsigned long long* ) & sum_discount        [group_index], thread_sum_discount        [group_index]);
        atomicAdd(                        & record_count        [group_index], thread_record_count        [group_index]);
    }
}

 __global__
void in_local_mem_ht_tpchQ01_compressed(
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
    sum_quantity_t         thread_sum_quantity         [num_potential_groups] = { 0 };
    sum_base_price_t       thread_sum_base_price       [num_potential_groups] = { 0 };
    sum_discounted_price_t thread_sum_discounted_price [num_potential_groups] = { 0 };
    sum_charge_t           thread_sum_charge           [num_potential_groups] = { 0 };
    sum_discount_t         thread_sum_discount         [num_potential_groups] = { 0 };
    cardinality_t          thread_record_count         [num_potential_groups] = { 0 };


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

            thread_sum_quantity        [group_index] += line_quantity;
            thread_sum_base_price      [group_index] += line_price;
            thread_sum_charge          [group_index] += line_charge;
            thread_sum_discounted_price[group_index] += line_discounted_price;
            thread_sum_discount        [group_index] += line_discount;
            thread_record_count        [group_index] ++;
        }
    }

    // final aggregation

    // These manual casts are really unbecoming. We need a wrapper...
    #pragma unroll
    for (int group_index = 0; group_index < num_potential_groups; ++group_index) {
        atomicAdd( (unsigned long long* ) & sum_quantity        [group_index], thread_sum_quantity        [group_index]);
        atomicAdd( (unsigned long long* ) & sum_base_price      [group_index], thread_sum_base_price      [group_index]);
        atomicAdd( (unsigned long long* ) & sum_charge          [group_index], thread_sum_charge          [group_index]);
        atomicAdd( (unsigned long long* ) & sum_discounted_price[group_index], thread_sum_discounted_price[group_index]);
        atomicAdd( (unsigned long long* ) & sum_discount        [group_index], thread_sum_discount        [group_index]);
        atomicAdd(                        & record_count        [group_index], thread_record_count        [group_index]);
    }

}

} // namespace cuda
