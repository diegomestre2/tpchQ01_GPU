#pragma once

#include "kernel.hpp"

namespace cuda {
    __global__
    void global_ht_tpchQ01(
        ship_date_t *shipdate,
        DISCOUNT_TYPE *discount,
        EXTENDEDPRICE_TYPE *extendedprice,
        tax_t *tax,
        return_flag_t *returnflag,
        line_status_t *linestatus,
        quantity_t *quantity,
        AggrHashTable *aggregations,
        u64_t cardinality) {

        u64_t i = VALUES_PER_THREAD * (blockIdx.x * blockDim.x + threadIdx.x);
        u64_t end = min((u64_t)cardinality, i + VALUES_PER_THREAD);
        for(; i < end; ++i) {
            if (shipdate[i] <= threshold_ship_date) {
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
    void compressed::global_ht_tpchQ01_datatypes(
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

        u64_t i = VALUES_PER_THREAD * (blockIdx.x * blockDim.x + threadIdx.x);
        u64_t end = min((u64_t)cardinality, i + VALUES_PER_THREAD);
        for(; i < end; ++i) {
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
