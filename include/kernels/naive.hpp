#pragma once

#include "kernel.hpp"

namespace cuda {
    __global__
    void naive_tpchQ01(
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
        record_count_t cardinality){

        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < cardinality && shipdate[i] <= threshold_ship_date){//todate_(2, 9, 1998)) {
            const auto disc = discount[i];
            const auto price = extendedprice[i];
            const auto disc_1 = Decimal64::ToValue(1, 0) - disc;
            const auto tax_1 = tax[i] + Decimal64::ToValue(1, 0);
            const auto disc_price = Decimal64::Mul(disc_1, price);
            const auto charge = Decimal64::Mul(disc_price, tax_1);
            const idx_t idx = returnflag[i] << 8 | linestatus[i];

            atomicAdd((uint64_t*)&sum_quantity[idx], (uint64_t) quantity[i]);
            atomicAdd((uint64_t*)&sum_base_price[idx], (uint64_t) price);
            atomicAdd((uint64_t*)&sum_charge[idx], (uint64_t) charge);
            atomicAdd((uint64_t*)&sum_discounted_price[idx], (uint64_t)disc_price);
            atomicAdd((uint64_t*)&sum_discount[idx], (uint64_t) disc);
            atomicAdd((uint64_t*)&record_count[idx], (uint64_t) 1);
            
        }
    }
}
