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
        cardinality_t *record_count,
        ship_date_t *shipdate,
        discount_t *discount,
        extended_price_t *extendedprice,
        tax_t *tax,
        return_flag_t *returnflag,
        line_status_t *linestatus,
        quantity_t *quantity,
        cardinality_t cardinality){

        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < cardinality && shipdate[i] <= threshold_ship_date){//todate_(2, 9, 1998)) {
            const auto disc = discount[i];
            const auto price = extendedprice[i];
            const auto disc_1 = Decimal64::ToValue(1, 0) - disc;
            const auto tax_1 = tax[i] + Decimal64::ToValue(1, 0);
            const auto disc_price = Decimal64::Mul(disc_1, price);
            const auto charge = Decimal64::Mul(disc_price, tax_1);
            const idx_t idx = returnflag[i] << 8 | linestatus[i];
            atomicAdd((unsigned long long*)&(aggregations[idx].sum_quantity), quantity[i]);
            atomicAdd((unsigned long long*)&(aggregations[idx].sum_base_price), price);
            auto old = atomicAdd((unsigned long long*)&(aggregations[idx].sum_charge), charge);
            if (old + charge < charge) {
                atomicAdd((unsigned long long*)&(aggregations[idx].sum_charge) + 1, 1);
            }

            auto old_2 = atomicAdd((unsigned long long*)&(aggregations[idx].sum_disc_price), disc_price);
            if (old_2 + disc_price < disc_price) {
                atomicAdd((unsigned long long*)&(aggregations[idx].sum_disc_price) + 1, 1);
            }
            atomicAdd((unsigned long long*)&(aggregations[idx].sum_disc), disc);
            atomicAdd((unsigned long long*)&(aggregations[idx].count), 1);
            
        }
    }
}
