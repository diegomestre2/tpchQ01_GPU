#pragma once

#include "kernel.hpp"
#include "data_types.h"
namespace cuda {
    __global__
    void global_ht_tpchQ01(
        sum_quantity_t *sum_quantity,
        sum_base_price_t *sum_base_price,
        sum_discounted_price_t *sum_discounted_price,
        sum_charge_t *sum_charge,
        sum_discount_t *sum_discount,
        cardinality_t *record_count,
        ship_date_t *shipdate,
        discount_t *discount,
        extended_price_t *extendedprice,
        tax_t *tax,
        return_flag_t *returnflag,
        line_status_t *linestatus,
        quantity_t *quantity,
        cardinality_t cardinality) {

        uint64_t i = values_per_thread * (blockIdx.x * blockDim.x + threadIdx.x);
        uint64_t end = min((uint64_t)cardinality, i + values_per_thread);
        for(; i < end; ++i) {
            if (shipdate[i] <= threshold_ship_date) {
                const int disc = discount[i];
                const int price = extendedprice[i];
                const int disc_1 = Decimal64::ToValue(1, 0) - disc;
                const int tax_1 = tax[i] + Decimal64::ToValue(1, 0);
                const int disc_price = Decimal64::Mul(disc_1, price);
                const int charge = Decimal64::Mul(disc_price, tax_1);
                const idx_t idx = magic_hash(returnflag[i], linestatus[i]);
                atomicAdd((unsigned long long*) &aggregations[idx].sum_quantity, quantity[i]);
                atomicAdd((unsigned long long*) &aggregations[idx].sum_base_price, price);
                atomicAdd((unsigned long long*) &aggregations[idx].sum_charge, charge);
                atomicAdd((unsigned long long*) &aggregations[idx].sum_disc_price, disc_price);
                atomicAdd((unsigned long long*) &aggregations[idx].sum_disc, disc);
                atomicAdd((unsigned long long*) &aggregations[idx].count, 1);
            }
        }
    }

    __global__
    void global_ht_tpchQ01_small_datatypes(
        sum_quantity_t *sum_quantity,
        sum_base_price_t *sum_base_price,
        sum_discounted_price_t *sum_discounted_price,
        sum_charge_t *sum_charge,
        sum_discount_t *sum_discount,
        cardinality_t *record_count,
        compressed::ship_date_t *shipdate,
        compressed::discount_t *discount,
        compressed::extended_price_t *extendedprice,
        compressed::tax_t *tax,
        compressed::return_flag_t *returnflag,
        compressed::line_status_t *linestatus,
        compressed::quantity_t *quantity,
        cardinality_t cardinality) {

        constexpr uint8_t RETURNFLAG_MASK[] = { 0x03, 0x0C, 0x30, 0xC0 };
        constexpr uint8_t LINESTATUS_MASK[] = { 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80 };

        uint64_t i = values_per_thread * (blockIdx.x * blockDim.x + threadIdx.x);
        uint64_t end = min((uint64_t)cardinality, i + values_per_thread);
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

                atomicAdd((unsigned long long*) &aggregations[idx].sum_quantity,  (quantity[i] * 100));
                atomicAdd((unsigned long long*) &aggregations[idx].sum_base_price,  price);
                atomicAdd((unsigned long long*) &aggregations[idx].sum_charge,  charge);
                atomicAdd((unsigned long long*) &aggregations[idx].sum_disc_price,  disc_price);
                atomicAdd((unsigned long long*) &aggregations[idx].sum_disc,  disc);
                atomicAdd((unsigned long long*) &aggregations[idx].count,  1);
            }
        }
    }
}
