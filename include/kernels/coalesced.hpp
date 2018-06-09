#pragma once

#include "kernel.hpp"
#include "constants.hpp"
#include "data_types.h"
#include "atomic_operations.cuh"
#include "bit_operations.h"

#ifdef __CUDACC__
#define __fhd__  __forceinline__ __host__ __device__
#define __fd__   __forceinline__ __device__
#else
#define __fhd__ inline
#define __fd__  inline
#endif


namespace cuda {

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

/*
		__global__
    void thread_local_tpchQ01_coalesced(
        sum_quantity_t *sum_quantity,
        sum_base_price_t *sum_base_price,
        sum_discounted_price_t *sum_discounted_price,
        sum_charge_t *sum_charge,
        sum_discount_t *sum_discount,
        cardinality_t *record_count,
        ship_date_t *shipdate,
        discount_t *discount,
        extended_price_t *extendedprice,
        ship_date_t *shipdate,
        tax_t *tax,
        return_flag_t *returnflag,
        line_status_t *linestatus,
        quantity_t *quantity,
        cardinality_t cardinality) {

        constexpr size_t N = 18;

        uint64_t i =  (blockIdx.x * blockDim.x + threadIdx.x);
        uint64_t end = (uint64_t)cardinality;
        uint64_t stride = (blockDim.x * gridDim.x); //Grid-Stride

        for(; i < end; i+=stride) {
            if (shipdate[i] <= threshold_ship_date) {
                const int disc = discount[i];
                const int price = extendedprice[i];
                const int disc_1 = Decimal64::ToValue(1, 0) - disc;
                const int tax_1 = tax[i] + Decimal64::ToValue(1, 0);
                const int disc_price = Decimal64::Mul(disc_1, price);
                const int charge = Decimal64::Mul(disc_price, tax_1);
                const idx_t idx = magic_hash(returnflag[i], linestatus[i]);
                
                #pragma unroll
                for(int j = 0; j < N; ++j){
                    if(j == idx){
                        agg[j].sum_quantity   += quantity[i];
                        agg[j].sum_base_price += price;
                        agg[j].sum_charge     += charge;
                        agg[j].sum_disc_price += disc_price;
                        agg[j].sum_disc       += disc;
                        agg[j].count          += 1;
                    }
                }
            }
        }
        // final aggregation
        #pragma unroll
        for (i = 0; i < N; ++i) {
            if (!agg[i].count) {
                continue;
            }
            atomicAdd((unsigned long long*) &aggregations[i].sum_quantity, agg[i].sum_quantity);
            atomicAdd((unsigned long long*) &aggregations[i].sum_base_price,  agg[i].sum_base_price);
            atomicAdd((unsigned long long*) &aggregations[i].sum_charge,  agg[i].sum_charge);
            atomicAdd((unsigned long long*) &aggregations[i].sum_disc_price,  agg[i].sum_disc_price);
            atomicAdd((unsigned long long*) &aggregations[i].sum_disc,  agg[i].sum_disc);
            atomicAdd((unsigned long long*) &aggregations[i].count,  agg[i].count);
        }
    }
*/

 __global__
void thread_local_tpchQ01_small_datatypes_coalesced(
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
	cardinality_t                                     cardinality)
 {
	sum_quantity_t         thread_sum_quantity         [num_potential_groups] = { 0 };
	sum_base_price_t       thread_sum_base_price       [num_potential_groups] = { 0 };
	sum_discounted_price_t thread_sum_discounted_price [num_potential_groups] = { 0 };
	sum_charge_t           thread_sum_charge           [num_potential_groups] = { 0 };
	sum_discount_t         thread_sum_discount         [num_potential_groups] = { 0 };
	cardinality_t          thread_record_count         [num_potential_groups] = { 0 };

	cardinality_t i = (blockIdx.x * blockDim.x + threadIdx.x);
	cardinality_t stride = (blockDim.x * gridDim.x); //Grid-Stride
	for(; i < cardinality; i += stride) {
		if (shipdate[i] <= compressed_threshold_ship_date) {
			// TODO: Some of these calculations could work on uint32_t
			auto line_quantity         = quantity[i];
			auto line_discount         = discount[i];
			auto line_price            = extended_price[i];
			auto line_discount_factor  = Decimal64::ToValue(1, 0) - line_discount;
			auto line_discounted_price = Decimal64::Mul(line_discount_factor, line_price);
			auto line_tax_factor       = tax[i] + Decimal64::ToValue(1, 0);
			auto line_charge           = Decimal64::Mul(line_discounted_price, line_tax_factor);
			auto line_status_          = get_bit_resolution_element<log_line_status_bits, cardinality_t>(line_status, i);
			auto line_return_flag      = get_bit_resolution_element<log_return_flag_bits, cardinality_t>(return_flag, i);

			#pragma unroll
			for(int group_index = 0; group_index != num_potential_groups; ++group_index) {
				if(group_index == group_index) {
					thread_sum_quantity        [group_index] += line_quantity;
					thread_sum_base_price      [group_index] += line_price;
					thread_sum_charge          [group_index] += line_charge;
					thread_sum_discounted_price[group_index] += line_discounted_price;
					thread_sum_discount        [group_index] += line_discount;
					thread_record_count        [group_index] += 1;
				}
			}
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
