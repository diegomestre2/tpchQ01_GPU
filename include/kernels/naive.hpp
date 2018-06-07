#pragma once

#include "kernel.hpp"

namespace cuda {
    __global__
    void naive_tpchQ01(ship_date_t *shipdate, DISCOUNT_TYPE *discount, EXTENDEDPRICE_TYPE *extendedprice, tax_t *tax, 
        return_flag_t *returnflag, line_status_t *linestatus, quantity_t *quantity, AggrHashTable *aggregations, size_t cardinality){

        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < cardinality && shipdate[i] <= threshold_ship_date){//todate_(2, 9, 1998)) {
            const auto disc = discount[i];
            const auto price = extendedprice[i];
            const auto disc_1 = Decimal64::ToValue(1, 0) - disc;
            const auto tax_1 = tax[i] + Decimal64::ToValue(1, 0);
            const auto disc_price = Decimal64::Mul(disc_1, price);
            const auto charge = Decimal64::Mul(disc_price, tax_1);
            const idx_t idx = returnflag[i] << 8 | linestatus[i];
            atomicAdd((u64_t*)&(aggregations[idx].sum_quantity), (u64_t) quantity[i]);
            atomicAdd((u64_t*)&(aggregations[idx].sum_base_price), (u64_t)price);
            auto old = atomicAdd((u64_t*)&(aggregations[idx].sum_charge), charge);
            if (old + charge < charge) {
                atomicAdd((u64_t*)&(aggregations[idx].sum_charge) + 1, 1);
            }

            auto old_2 = atomicAdd((u64_t*)&(aggregations[idx].sum_disc_price), disc_price);
            if (old_2 + disc_price < disc_price) {
                atomicAdd((u64_t*)&(aggregations[idx].sum_disc_price) + 1, 1);
            }
            atomicAdd((u64_t*)&(aggregations[idx].sum_disc), (u64_t)disc);
            atomicAdd((u64_t*)&(aggregations[idx].count), (u64_t)1);
            
        }
    }
}
