#pragma once

#include "kernel.hpp"

namespace cuda {
    __global__
    void thread_local_coalesced_tpchQ01(
        SHIPDATE_TYPE *shipdate,
        DISCOUNT_TYPE *discount,
        EXTENDEDPRICE_TYPE *extendedprice,
        TAX_TYPE *tax,
        RETURNFLAG_TYPE *returnflag,
        LINESTATUS_TYPE *linestatus,
        QUANTITY_TYPE *quantity,
        AggrHashTable *aggregations,
        u64_t cardinality) {

        constexpr size_t N = 4;
        AggrHashTableLocal agg[N];
        memset(agg, 0, sizeof(AggrHashTableLocal) * N);

        u64_t i =  (blockIdx.x * blockDim.x + threadIdx.x);
        u64_t end = (u64_t)cardinality;
        u64_t stride = (blockDim.x * gridDim.x); //Grid-Stride
        for(; i < end; i+=stride) {
            if (shipdate[i] <= 729999 - SHIPDATE_MIN) {
                const int disc = discount[i];
                const int price = extendedprice[i];
                const int disc_1 = Decimal64::ToValue(1, 0) - disc;
                const int tax_1 = tax[i] + Decimal64::ToValue(1, 0);
                const int disc_price = Decimal64::Mul(disc_1, price);
                const int charge = Decimal64::Mul(disc_price, tax_1);
                
                if(returnflag[i] == 'A' && linestatus[i] == 'F'){
                    agg[0].sum_quantity   += quantity[i];
                    agg[0].sum_base_price += price;
                    agg[0].sum_charge     += charge;
                    agg[0].sum_disc_price += disc_price;
                    agg[0].sum_disc       += disc;
                    agg[0].count          += 1;
                }else if(returnflag[i] == 'N' && linestatus[i] == 'F'){
                    agg[1].sum_quantity   += quantity[i];
                    agg[1].sum_base_price += price;
                    agg[1].sum_charge     += charge;
                    agg[1].sum_disc_price += disc_price;
                    agg[1].sum_disc       += disc;
                    agg[1].count          += 1;

                }else if(returnflag[i] == 'N' && linestatus[i] == 'O'){
                    agg[2].sum_quantity   += quantity[i];
                    agg[2].sum_base_price += price;
                    agg[2].sum_charge     += charge;
                    agg[2].sum_disc_price += disc_price;
                    agg[2].sum_disc       += disc;
                    agg[2].count          += 1;

                }else if(returnflag[i] == 'R' && linestatus[i] == 'F'){
                    agg[3].sum_quantity   += quantity[i];
                    agg[3].sum_base_price += price;
                    agg[3].sum_charge     += charge;
                    agg[3].sum_disc_price += disc_price;
                    agg[3].sum_disc       += disc;
                    agg[3].count          += 1;
                }
            }
        }
        // final aggregation
        #pragma unroll
        for (i = 0; i < N; ++i) {
            atomicAdd(&aggregations[i].sum_quantity, (u64_t) agg[i].sum_quantity * 100);
            atomicAdd(&aggregations[i].sum_base_price, (u64_t) agg[i].sum_base_price);
            if (atomicAdd(&aggregations[i].sum_charge, (u64_t) agg[i].sum_charge) < agg[i].sum_charge) {
                atomicAdd(&aggregations[i].sum_charge_hi, 1);
            }
            if (atomicAdd(&aggregations[i].sum_disc_price, (u64_t) agg[i].sum_disc_price) < agg[i].sum_disc_price) {
                atomicAdd(&aggregations[i].sum_disc_price_hi, 1);
            }
            atomicAdd(&aggregations[i].sum_disc, (u64_t) agg[i].sum_disc);
            atomicAdd(&aggregations[i].count, (u64_t) agg[i].count);
        }
    }
}
