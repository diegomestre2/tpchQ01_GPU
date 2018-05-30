#ifndef H_KERNEL_NAIVE
#define H_KERNEL_NAIVE

#include "common.hpp"

struct KernelNaive : BaseKernel {
	using BaseKernel::BaseKernel;

	__attribute__((noinline)) void operator()() noexcept {
		kernel_prologue();

		for (size_t i=0; i<cardinality; i++) {
			if (shipdate[i] <= cmp.dte_val) {
				const auto disc = discount[i];
				const auto price = extendedprice[i];
				const auto disc_1 = Decimal64::ToValue(1, 0) - disc;
				const auto tax_1 = tax[i] + Decimal64::ToValue(1, 0);
				const auto disc_price = Decimal64::Mul(disc_1, price);
				const auto charge = Decimal64::Mul(disc_price, tax_1);
				const idx_t idx = returnflag[i] << 8 | linestatus[i];
				aggrs0[idx].sum_quantity += quantity[i];
				aggrs0[idx].sum_base_price += price;
				aggrs0[idx].sum_disc_price = int128_add64(aggrs0[idx].sum_disc_price, disc_price);
				aggrs0[idx].sum_charge = int128_add64(aggrs0[idx].sum_charge, charge);
				aggrs0[idx].sum_disc += disc;
				aggrs0[idx].count++;
			}
		}
	}
};



#endif