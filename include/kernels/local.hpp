#pragma once

#include "kernel.hpp"
#include "data_types.h"

namespace cuda {
    __global__
    void thread_local_tpchQ01(
        sum_quantity_t *sum_quantity,
        sum_base_price_t *sum_base_price,
        sum_discounted_price_t *sum_discounted_price,
        sum_charge_t *sum_charge,
        sum_discount_t *sum_discount,
        record_count_t *record_count,
        ship_date_t *shipdate,
        DISCOUNT_TYPE *discount,
        EXTENDEDPRICE_TYPE *extendedprice,
        tax_t *tax,
        return_flag_t *returnflag,
        line_status_t *linestatus,
        quantity_t *quantity,
        u64_t cardinality) {

        constexpr size_t N = 18;
        AggrHashTable agg[N];
        memset(agg, 0, sizeof(AggrHashTable) * N);

        u64_t i = VALUES_PER_THREAD * (blockIdx.x * blockDim.x + threadIdx.x);
        u64_t end = min((u64_t)cardinality, i + VALUES_PER_THREAD);
        
        for(; i < end; ++i) {
            if (shipdate[i] <= 729999) {
                const int disc = discount[i];
                const int price = extendedprice[i];
                const int disc_1 = Decimal64::ToValue(1, 0) - disc;
                const int tax_1 = tax[i] + Decimal64::ToValue(1, 0);
                const int disc_price = Decimal64::Mul(disc_1, price);
                const int charge = Decimal64::Mul(disc_price, tax_1);
                const idx_t idx = magic_hash(returnflag[i], linestatus[i]);
                
                agg[idx].sum_quantity   += quantity[i];
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
            atomicAdd((u64_t*)&sum_quantity[i], (u64_t) agg[i].sum_quantity);
            atomicAdd((u64_t*)&sum_base_price[i], (u64_t) agg[i].sum_base_price);
            atomicAdd((u64_t*)&sum_charge[i], (u64_t) agg[i].sum_charge);
            atomicAdd((u64_t*)&sum_discounted_price[i], (u64_t) agg[i].sum_disc_price);
            atomicAdd((u64_t*)&sum_discount[i], (u64_t) agg[i].sum_disc);
            atomicAdd((u64_t*)&record_count[i], (u64_t) agg[i].count);
        }
	}


    __global__
    void thread_local_tpchQ01_snall_datatypes(
        sum_quantity_t *sum_quantity,
        sum_base_price_t *sum_base_price,
        sum_discounted_price_t *sum_discounted_price,
        sum_charge_t *sum_charge,
        sum_discount_t *sum_discount,
        record_count_t *record_count,
        compressed::ship_date_t *shipdate,
        compressed::discount_t *discount,
        compressed::extended_price_t *extendedprice,
        compressed::tax_t *tax,
        compressed::return_flag_t *returnflag,
        compressed::line_status_t *linestatus,
        compressed::quantity_t *quantity,
        u64_t cardinality) {

        constexpr uint8_t RETURNFLAG_MASK[] = { 0x03, 0x0C, 0x30, 0xC0 };
        constexpr uint8_t LINESTATUS_MASK[] = { 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80 };

        constexpr size_t N = 8;
        AggrHashTable agg[N];
        memset(agg, 0, sizeof(AggrHashTable) * N);

        u64_t i = VALUES_PER_THREAD * (blockIdx.x * blockDim.x + threadIdx.x);
        u64_t end = min((u64_t)cardinality, i + VALUES_PER_THREAD);
        for(; i < end; ++i) {
            if (shipdate[i] <= (729999 - ship_date_frame_of_reference)) {
                const int disc = discount[i];
                const int price = extendedprice[i];
                const int disc_1 = Decimal64::ToValue(1, 0) - disc;
                const int tax_1 = tax[i] + Decimal64::ToValue(1, 0);
                const int disc_price = Decimal64::Mul(disc_1, price);
                const int charge = Decimal64::Mul(disc_price, tax_1);
                const uint8_t rflag = (returnflag[i / 4] & RETURNFLAG_MASK[i % 4]) >> (2 * (i % 4));
                const uint8_t lstatus = (linestatus[i / 8] & LINESTATUS_MASK[i % 8]) >> (i % 8);
                const uint8_t idx = rflag + 4 * lstatus;
                
                agg[idx].sum_quantity   += quantity[i];
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
            atomicAdd((u64_t*)&sum_quantity[i], (u64_t) agg[i].sum_quantity);
            atomicAdd((u64_t*)&sum_base_price[i], (u64_t) agg[i].sum_base_price);
            atomicAdd((u64_t*)&sum_charge[i], (u64_t) agg[i].sum_charge);
            atomicAdd((u64_t*)&sum_discounted_price[i], (u64_t) agg[i].sum_disc_price);
            atomicAdd((u64_t*)&sum_discount[i], (u64_t) agg[i].sum_disc);
            atomicAdd((u64_t*)&record_count[i], (u64_t) agg[i].count);
        }
    }
}