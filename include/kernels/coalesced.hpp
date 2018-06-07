#pragma once

#include "kernel.hpp"

namespace cuda {
    __global__
    void thread_local_tpchQ01_coalesced(
        ship_date_t *shipdate,
        DISCOUNT_TYPE *discount,
        EXTENDEDPRICE_TYPE *extendedprice,
        tax_t *tax,
        return_flag_t *returnflag,
        line_status_t *linestatus,
        quantity_t *quantity,
        AggrHashTable *aggregations,
        u64_t cardinality) {

        constexpr size_t N = 18;
        AggrHashTableLocal agg[N];
        memset(agg, 0, sizeof(AggrHashTableLocal) * N);

        u64_t i =  (blockIdx.x * blockDim.x + threadIdx.x);
        u64_t end = (u64_t)cardinality;
        u64_t stride = (blockDim.x * gridDim.x); //Grid-Stride

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
            atomicAdd(&aggregations[i].sum_quantity, (u64_t) agg[i].sum_quantity);
            atomicAdd(&aggregations[i].sum_base_price, (u64_t) agg[i].sum_base_price);
            atomicAdd(&aggregations[i].sum_charge, (u64_t) agg[i].sum_charge);
            atomicAdd(&aggregations[i].sum_disc_price, (u64_t) agg[i].sum_disc_price);
            atomicAdd(&aggregations[i].sum_disc, (u64_t) agg[i].sum_disc);
            atomicAdd(&aggregations[i].count, (u64_t) agg[i].count);
        }
    }

     __global__
    void compressed::thread_local_tpchQ01_datatypes_coalesced(
        compressed::ship_date_t *shipdate,
        compressed_discount_t *discount,
        compressed_extended_price_t *extendedprice,
        compressed::tax_t *tax,
        compressed::return_flag_t *returnflag,
        compressed::line_status_t *linestatus,
        compressed::quantity_t *quantity,
        AggrHashTable *aggregations,
        u64_t cardinality) {

        constexpr uint8_t RETURNFLAG_MASK[] = { 0x03, 0x0C, 0x30, 0xC0 };
        constexpr uint8_t LINESTATUS_MASK[] = { 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80 };

        constexpr size_t N = 8;
        AggrHashTable agg[N];
        memset(agg, 0, sizeof(AggrHashTable) * N);

        u64_t i = (blockIdx.x * blockDim.x + threadIdx.x);
        u64_t end = (u64_t)cardinality;
        u64_t stride = (blockDim.x * gridDim.x); //Grid-Stride
        for(; i < end; i+=stride) {
            if (shipdate[i] <= compressed_threshold_ship_date) {
                const int disc = discount[i];
                const int price = extendedprice[i];
                const int disc_1 = Decimal64::ToValue(1, 0) - disc;
                const int tax_1 = tax[i] + Decimal64::ToValue(1, 0);
                const int disc_price = Decimal64::Mul(disc_1, price);
                const int charge = Decimal64::Mul(disc_price, tax_1);
                const uint8_t rflag = (returnflag[i / 4] & RETURNFLAG_MASK[i % 4]) >> (2 * (i % 4));
                const uint8_t lstatus = (linestatus[i / 8] & LINESTATUS_MASK[i % 8]) >> (i % 8);
                const uint8_t idx = rflag + 4 * lstatus;
                
                #pragma unroll
                for(int j = 0; j != N; ++j){
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
            atomicAdd(&aggregations[i].sum_quantity, (u64_t) agg[i].sum_quantity * 100);
            atomicAdd(&aggregations[i].sum_base_price, (u64_t) agg[i].sum_base_price);
            atomicAdd(&aggregations[i].sum_charge, (u64_t) agg[i].sum_charge);
            atomicAdd(&aggregations[i].sum_disc_price, (u64_t) agg[i].sum_disc_price);
            atomicAdd(&aggregations[i].sum_disc, (u64_t) agg[i].sum_disc);
            atomicAdd(&aggregations[i].count, (u64_t) agg[i].count);
        }
    }
}
