/*
 * In this kernel variant, each hash table has as many
 * corresponding threads as it has entries (i.e. potential values),
 * with all such threads belonging to the same warp. A warp
 * may thus be responsible for multiple hash tables; and the
 * number of possible values (= hash table size) is limited
 * to the warp size. For larger tables it is still possible
 * to adopt a similar approach by having each thread be
 * responsible for multiple values, but this suffers
 * from an increasing instruction count per single update,
 * a problem also shared by the per-thread in-registers hash
 * table approach.
 */
#pragma once

#include "preprocessor_shorthands.cuh"
#include "atomics.cuh"
#include "constants.hpp"
#include "data_types.h"
#include "bit_operations.hpp"

namespace cuda {

__global__
void in_registers_ht_tpchQ01(
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
    enum {
        tables_per_warp       = warp_size / num_potential_groups,
        active_lanes_per_warp = num_potential_groups * tables_per_warp,
    };

    auto lane_index = threadIdx.x % warp_size;
    auto warp_index = threadIdx.x / warp_size;

    if (lane_index >= active_lanes_per_warp) { return; }

    auto intra_warp_table_index = lane_index / num_potential_groups;
    auto lane_group_index = lane_index % num_potential_groups;

    // Following are the aggregate "tables" - where each thread
    // holds just one aggregate per table, for a single group

    sum_quantity_t         single_group_sum_quantity         { 0 };
    sum_base_price_t       single_group_sum_base_price       { 0 };
    sum_discounted_price_t single_group_sum_discounted_price { 0 };
    sum_charge_t           single_group_sum_charge           { 0 };
    sum_discount_t         single_group_sum_discount         { 0 };
    cardinality_t          single_group_record_count         { 0 };

    // At each iteration of the main loop, a thread handles a single input tuple;
    // and the same tuple must be handled by all threads corresponding to a single
    // aggregates table. Also, there's no need to have that tuple considered
    // by any other grid thread. We this calculate the initial tuple index to consider
    // using "aggregate table indices".

    auto num_warps_per_block = blockDim.x / warp_size;
        // Note: The block size must be a multiple of warp_size, otherwise
        // this instruction will mess things up.
    auto num_tables_per_block = num_warps_per_block * tables_per_warp;
    cardinality_t intra_warp_tuple_index = intra_warp_table_index;
    cardinality_t intra_block_tuple_index = warp_index * tables_per_warp + intra_warp_tuple_index;
    cardinality_t initial_tuple_index =
        blockIdx.x * num_tables_per_block + intra_block_tuple_index;
    cardinality_t stride = (gridDim.x * num_tables_per_block ); // this is grid-stride, sort of

    for(cardinality_t i = initial_tuple_index; i < num_tuples; i += stride) {
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

            int line_group_index =
                (encode_return_flag(line_return_flag) << line_status_bits) + encode_line_status(line_status_);

            if (line_group_index == lane_group_index) {
                single_group_sum_quantity         += line_quantity;
                single_group_sum_base_price       += line_price;
                single_group_sum_charge           += line_charge;
                single_group_sum_discounted_price += line_discounted_price;
                single_group_sum_discount         += line_discount;
                single_group_record_count         ++;
            }
        }
    }

    // final aggregation

    atomicAdd( & sum_quantity        [lane_group_index], single_group_sum_quantity         );
    atomicAdd( & sum_base_price      [lane_group_index], single_group_sum_base_price       );
    atomicAdd( & sum_charge          [lane_group_index], single_group_sum_charge           );
    atomicAdd( & sum_discounted_price[lane_group_index], single_group_sum_discounted_price );
    atomicAdd( & sum_discount        [lane_group_index], single_group_sum_discount         );
    atomicAdd( & record_count        [lane_group_index], single_group_record_count         );
}

 __global__
void in_registers_ht_tpchQ01_compressed(
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
    enum {
        tables_per_warp       = warp_size / num_potential_groups,
        active_lanes_per_warp = num_potential_groups * tables_per_warp,
    };

    auto lane_index = threadIdx.x % warp_size;
    auto warp_index = threadIdx.x / warp_size;

    if (lane_index >= active_lanes_per_warp) { return; }

    auto intra_warp_table_index = lane_index / num_potential_groups;
    auto lane_group_index = lane_index % num_potential_groups;

    // Following are the aggregate "tables" - where each thread
    // holds just one aggregate per table, for a single group

    sum_quantity_t         single_group_sum_quantity         { 0 };
    sum_base_price_t       single_group_sum_base_price       { 0 };
    sum_discounted_price_t single_group_sum_discounted_price { 0 };
    sum_charge_t           single_group_sum_charge           { 0 };
    sum_discount_t         single_group_sum_discount         { 0 };
    cardinality_t          single_group_record_count         { 0 };

    // At each iteration of the main loop, a thread handles a single input tuple;
    // and the same tuple must be handled by all threads corresponding to a single
    // aggregates table. Also, there's no need to have that tuple considered
    // by any other grid thread. We this calculate the initial tuple index to consider
    // using "aggregate table indices".

    auto num_warps_per_block = blockDim.x / warp_size;
        // Note: The block size must be a multiple of warp_size, otherwise
        // this instruction will mess things up.
    auto num_tables_per_block = num_warps_per_block * tables_per_warp;
    cardinality_t intra_warp_tuple_index = intra_warp_table_index;
    cardinality_t intra_block_tuple_index = warp_index * tables_per_warp + intra_warp_tuple_index;
    cardinality_t initial_tuple_index =
        blockIdx.x * num_tables_per_block + intra_block_tuple_index;
    cardinality_t stride = (gridDim.x * num_tables_per_block ); // this is grid-stride, sort of


    for(cardinality_t i = initial_tuple_index; i < num_tuples; i += stride) {
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

            int line_group_index = (line_return_flag << line_status_bits) + line_status_;

            if (line_group_index == lane_group_index) {
                single_group_sum_quantity         += line_quantity;
                single_group_sum_base_price       += line_price;
                single_group_sum_charge           += line_charge;
                single_group_sum_discounted_price += line_discounted_price;
                single_group_sum_discount         += line_discount;
                single_group_record_count         ++;
            }
        }
    }

    // final aggregation

    atomicAdd( & sum_quantity        [lane_group_index], single_group_sum_quantity         );
    atomicAdd( & sum_base_price      [lane_group_index], single_group_sum_base_price       );
    atomicAdd( & sum_charge          [lane_group_index], single_group_sum_charge           );
    atomicAdd( & sum_discounted_price[lane_group_index], single_group_sum_discounted_price );
    atomicAdd( & sum_discount        [lane_group_index], single_group_sum_discount         );
    atomicAdd( & record_count        [lane_group_index], single_group_record_count         );
}

 __global__
void in_registers_ht_tpchQ01_filter_pushdown_compressed(
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
    enum {
        tables_per_warp       = warp_size / num_potential_groups,
        active_lanes_per_warp = num_potential_groups * tables_per_warp,
    };

    auto lane_index = threadIdx.x % warp_size;
    auto warp_index = threadIdx.x / warp_size;

    if (lane_index >= active_lanes_per_warp) { return; }

    auto intra_warp_table_index = lane_index / num_potential_groups;
    auto lane_group_index = lane_index % num_potential_groups;

    // Following are the aggregate "tables" - where each thread
    // holds just one aggregate per table, for a single group

    sum_quantity_t         single_group_sum_quantity         { 0 };
    sum_base_price_t       single_group_sum_base_price       { 0 };
    sum_discounted_price_t single_group_sum_discounted_price { 0 };
    sum_charge_t           single_group_sum_charge           { 0 };
    sum_discount_t         single_group_sum_discount         { 0 };
    cardinality_t          single_group_record_count         { 0 };

    // At each iteration of the main loop, a thread handles a single input tuple;
    // and the same tuple must be handled by all threads corresponding to a single
    // aggregates table. Also, there's no need to have that tuple considered
    // by any other grid thread. We this calculate the initial tuple index to consider
    // using "aggregate table indices".

    auto num_warps_per_block = blockDim.x / warp_size;
        // Note: The block size must be a multiple of warp_size, otherwise
        // this instruction will mess things up.
    auto num_tables_per_block = num_warps_per_block * tables_per_warp;
    cardinality_t intra_warp_tuple_index = intra_warp_table_index;
    cardinality_t intra_block_tuple_index = warp_index * tables_per_warp + intra_warp_tuple_index;
    cardinality_t initial_tuple_index =
        blockIdx.x * num_tables_per_block + intra_block_tuple_index;
    cardinality_t stride = (gridDim.x * num_tables_per_block ); // this is grid-stride, sort of

    for(cardinality_t i = initial_tuple_index; i < num_tuples; i += stride) {
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

            int line_group_index = (line_return_flag << line_status_bits) + line_status_;

            if (line_group_index == lane_group_index) {
                single_group_sum_quantity         += line_quantity;
                single_group_sum_base_price       += line_price;
                single_group_sum_charge           += line_charge;
                single_group_sum_discounted_price += line_discounted_price;
                single_group_sum_discount         += line_discount;
                single_group_record_count         ++;
            }
        }
    }

    // final aggregation

    atomicAdd( & sum_quantity        [lane_group_index], single_group_sum_quantity         );
    atomicAdd( & sum_base_price      [lane_group_index], single_group_sum_base_price       );
    atomicAdd( & sum_charge          [lane_group_index], single_group_sum_charge           );
    atomicAdd( & sum_discounted_price[lane_group_index], single_group_sum_discounted_price );
    atomicAdd( & sum_discount        [lane_group_index], single_group_sum_discount         );
    atomicAdd( & record_count        [lane_group_index], single_group_record_count         );
}

} // namespace cuda
