#pragma once

#include "kernel.hpp"

namespace cuda {
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
        u64_t cardinality,
        int values_per_thread) {

        u64_t i = values_per_thread * (blockIdx.x * blockDim.x + threadIdx.x);
        u64_t end = min((u64_t)cardinality, i + values_per_thread);
        for(; i < end; ++i) {
            if (shipdate[i] <= 729999) {
                const int disc = discount[i];
                const int price = extendedprice[i];
                const int disc_1 = Decimal64::ToValue(1, 0) - disc;
                const int tax_1 = tax[i] + Decimal64::ToValue(1, 0);
                const int disc_price = Decimal64::Mul(disc_1, price);
                const int charge = Decimal64::Mul(disc_price, tax_1);
                const idx_t idx = magic_hash(returnflag[i], linestatus[i]);
                
                atomicAdd(&aggregations[idx].sum_quantity, (u64_t) quantity[i]);
                atomicAdd(&aggregations[idx].sum_base_price, (u64_t) price);
                atomicAdd(&aggregations[idx].sum_charge, (u64_t) charge);
                atomicAdd(&aggregations[idx].sum_disc_price, (u64_t) disc_price);
                atomicAdd(&aggregations[idx].sum_disc, (u64_t) disc);
                atomicAdd(&aggregations[idx].count, (u64_t) 1);
            }
        }
    }

    __global__
    void global_ht_tpchQ01_small_datatypes(
        SHIPDATE_TYPE_SMALL *shipdate,
        DISCOUNT_TYPE_SMALL *discount,
        EXTENDEDPRICE_TYPE_SMALL *extendedprice,
        TAX_TYPE_SMALL *tax,
        RETURNFLAG_TYPE_SMALL *returnflag,
        LINESTATUS_TYPE_SMALL *linestatus,
        QUANTITY_TYPE_SMALL *quantity,
        AggrHashTable *aggregations,
        u64_t cardinality,
        int values_per_thread) {

        constexpr uint8_t RETURNFLAG_MASK[] = { 0x03, 0x0C, 0x30, 0xC0 };
        constexpr uint8_t LINESTATUS_MASK[] = { 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80 };

        u64_t i = values_per_thread * (blockIdx.x * blockDim.x + threadIdx.x);
        u64_t end = min((u64_t)cardinality, i + values_per_thread);
        for(; i < end; ++i) {
            if (shipdate[i] <= (729999 - SHIPDATE_MIN)) {
                const int disc = discount[i];
                const int price = extendedprice[i];
                const int disc_1 = Decimal64::ToValue(1, 0) - disc;
                const int tax_1 = tax[i] + Decimal64::ToValue(1, 0);
                const int disc_price = Decimal64::Mul(disc_1, price);
                const int charge = Decimal64::Mul(disc_price, tax_1);
                const uint8_t rflag = (returnflag[i / 4] & RETURNFLAG_MASK[i % 4]) >> (2 * (i % 4));
                const uint8_t lstatus = (linestatus[i / 8] & LINESTATUS_MASK[i % 8]) >> (i % 8);
                const uint8_t idx = rflag + 4 * lstatus;

                atomicAdd(&aggregations[idx].sum_quantity, (u64_t) (quantity[i] * 100));
                atomicAdd(&aggregations[idx].sum_base_price, (u64_t) price);
                atomicAdd(&aggregations[idx].sum_charge, (u64_t) charge);
                atomicAdd(&aggregations[idx].sum_disc_price, (u64_t) disc_price);
                atomicAdd(&aggregations[idx].sum_disc, (u64_t) disc);
                atomicAdd(&aggregations[idx].count, (u64_t) 1);
            }
        }
    }
}
