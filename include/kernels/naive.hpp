#pragma once

#include "kernel.hpp"

namespace cuda {
    __global__
    void naive_tpchQ01(SHIPDATE_TYPE *shipdate, DISCOUNT_TYPE *discount, EXTENDEDPRICE_TYPE *extendedprice, TAX_TYPE *tax, 
        RETURNFLAG_TYPE *returnflag, LINESTATUS_TYPE *linestatus, QUANTITY_TYPE *quantity, GPUAggrHashTable *aggregations, size_t cardinality){

        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < cardinality && shipdate[i] <= 729999){//todate_(2, 9, 1998)) {
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
