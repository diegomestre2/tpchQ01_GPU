#pragma once

#include "preprocessor_shorthands.cuh"
#include "atomics.cuh"
#include "constants.hpp"
#include "data_types.h"
#include "bit_operations.hpp"

namespace cuda {

__global__
void in_registers_per_thread_ht_tpchQ01(
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

            #pragma unroll
            for(int i = 0; i < num_potential_groups; i++) {
                if(i == group_index) {
                    thread_sum_quantity        [i] += line_quantity;
                    thread_sum_base_price      [i] += line_price;
                    thread_sum_charge          [i] += line_charge;
                    thread_sum_discounted_price[i] += line_discounted_price;
                    thread_sum_discount        [i] += line_discount;
                    thread_record_count        [i] ++;
                }
            }
        }
    }

    // final aggregation

    #pragma unroll
    for (int group_index = 0; group_index < num_potential_groups; ++group_index) {
        atomicAdd( & sum_quantity        [group_index], thread_sum_quantity        [group_index]);
        atomicAdd( & sum_base_price      [group_index], thread_sum_base_price      [group_index]);
        atomicAdd( & sum_charge          [group_index], thread_sum_charge          [group_index]);
        atomicAdd( & sum_discounted_price[group_index], thread_sum_discounted_price[group_index]);
        atomicAdd( & sum_discount        [group_index], thread_sum_discount        [group_index]);
        atomicAdd( & record_count        [group_index], thread_record_count        [group_index]);
    }
}

 __global__
void in_registers_per_thread_ht_tpchQ01_compressed(
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

            #pragma unroll
            for(int i = 0; i < num_potential_groups; i++) {
                if(i == group_index) {
                    thread_sum_quantity        [i] += line_quantity;
                    thread_sum_base_price      [i] += line_price;
                    thread_sum_charge          [i] += line_charge;
                    thread_sum_discounted_price[i] += line_discounted_price;
                    thread_sum_discount        [i] += line_discount;
                    thread_record_count        [i] ++;
                }
            }
        }
    }

    // final aggregation

    #pragma unroll
    for (int group_index = 0; group_index < num_potential_groups; ++group_index) {
        atomicAdd( & sum_quantity        [group_index], thread_sum_quantity        [group_index]);
        atomicAdd( & sum_base_price      [group_index], thread_sum_base_price      [group_index]);
        atomicAdd( & sum_charge          [group_index], thread_sum_charge          [group_index]);
        atomicAdd( & sum_discounted_price[group_index], thread_sum_discounted_price[group_index]);
        atomicAdd( & sum_discount        [group_index], thread_sum_discount        [group_index]);
        atomicAdd( & record_count        [group_index], thread_record_count        [group_index]);
    }

}

 __global__
void in_registers_per_thread_ht_tpchQ01_filter_pushdown_compressed(
    sum_quantity_t*                      __restrict__ sum_quantity,
    sum_base_price_t*                    __restrict__ sum_base_price,
    sum_discounted_price_t*              __restrict__ sum_discounted_price,
    sum_charge_t*                        __restrict__ sum_charge,
    sum_discount_t*                      __restrict__ sum_discount,
    cardinality_t*                       __restrict__ record_count,
    const bit_container_t*               __restrict__ precomputed_filter,
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
        auto passes_filter = get_bit(precomputed_filter, i);
    	if (passes_filter) {
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

            #pragma unroll
            for(int i = 0; i < num_potential_groups; i++) {
                if(i == group_index) {
                    thread_sum_quantity        [i] += line_quantity;
                    thread_sum_base_price      [i] += line_price;
                    thread_sum_charge          [i] += line_charge;
                    thread_sum_discounted_price[i] += line_discounted_price;
                    thread_sum_discount        [i] += line_discount;
                    thread_record_count        [i] ++;
                }
            }
        }
    }

    // final aggregation

    #pragma unroll
    for (int group_index = 0; group_index < num_potential_groups; ++group_index) {
        atomicAdd( & sum_quantity        [group_index], thread_sum_quantity        [group_index]);
        atomicAdd( & sum_base_price      [group_index], thread_sum_base_price      [group_index]);
        atomicAdd( & sum_charge          [group_index], thread_sum_charge          [group_index]);
        atomicAdd( & sum_discounted_price[group_index], thread_sum_discounted_price[group_index]);
        atomicAdd( & sum_discount        [group_index], thread_sum_discount        [group_index]);
        atomicAdd( & record_count        [group_index], thread_record_count        [group_index]);
    }

}

} // namespace cuda
