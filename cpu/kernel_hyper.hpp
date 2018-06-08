#ifndef H_KERNEL_HYPER
#define H_KERNEL_HYPER

#include "common.hpp"

template<bool use_flag>
struct KernelHyPer : BaseKernel {
	using BaseKernel::BaseKernel;

	__attribute__((noinline)) void operator()() noexcept {
		kernel_prologue();

		int64_t disc_1;
		int64_t tax_1;
		int64_t disc_price;
		int64_t charge;

		if (!use_flag) {
			for (size_t i=0; i<cardinality; i++) {
				if (shipdate[i] <= cmp.dte_val) {
					const auto disc = discount[i];
					const auto price = extendedprice[i];
					

					if (UNLIKELY(__builtin_ssubll_overflow(Decimal64::ToValue(1, 0), disc, (long long int*)&disc_1))) {
						handle_overflow();
					}
					
					if (UNLIKELY(__builtin_saddll_overflow(Decimal64::ToValue(1, 0), tax[i], (long long int*)&tax_1))) {
						handle_overflow();
					}

					
					if (UNLIKELY(__builtin_smulll_overflow(disc_1, price, (long long int*)&disc_price))) {
						handle_overflow();
					}

					if (UNLIKELY(__builtin_smulll_overflow(disc_price, tax_1, (long long int*)&charge))) {
						handle_overflow();
					}
					const idx_t idx = returnflag[i] << 8 | linestatus[i];
					if (UNLIKELY(__builtin_saddll_overflow(aggrs0[idx].sum_quantity, quantity[i], (long long int*)&aggrs0[idx].sum_quantity))) {
						handle_overflow();
					}
					if (UNLIKELY(__builtin_saddll_overflow(aggrs0[idx].sum_base_price, price, (long long int*)&aggrs0[idx].sum_base_price))) {
						handle_overflow();	
					}

					if (UNLIKELY(__builtin_add_overflow(aggrs0[idx].sum_disc_price, disc_price, (int128_t*)&aggrs0[idx].sum_disc_price))) {
						handle_overflow();	
					}

					if (UNLIKELY(__builtin_add_overflow(aggrs0[idx].sum_charge, charge, (int128_t*)&aggrs0[idx].sum_charge))) {
						handle_overflow();	
					}	

					if (UNLIKELY(__builtin_saddll_overflow(aggrs0[idx].sum_disc, disc, (long long int*)&aggrs0[idx].sum_disc))) {
						handle_overflow();	
					}
					if (UNLIKELY(__builtin_uaddll_overflow(aggrs0[idx].count, 1, (long long unsigned int*)&aggrs0[idx].count))) {
						handle_overflow();	
					}
				}
			}
		} else {
			bool flag = false;
			for (size_t i=0; i<cardinality; i++) {
				if (shipdate[i] <= cmp.dte_val) {
					const auto disc = discount[i];
					const auto price = extendedprice[i];

					flag |= __builtin_ssubll_overflow(Decimal64::ToValue(1, 0), disc, (long long int*)&disc_1);
					flag |= __builtin_saddll_overflow(Decimal64::ToValue(1, 0), tax[i], (long long int*)&tax_1);
					flag |= __builtin_smulll_overflow(disc_1, price, (long long int*)&disc_price);
					flag |= __builtin_smulll_overflow(disc_price, tax_1, (long long int*)&charge);
					const idx_t idx = returnflag[i] << 8 | linestatus[i];
					flag |= __builtin_saddll_overflow(aggrs0[idx].sum_quantity, quantity[i], (long long int*)&aggrs0[idx].sum_quantity);
					flag |= __builtin_saddll_overflow(aggrs0[idx].sum_base_price, price, (long long int*)&aggrs0[idx].sum_base_price);

					flag |= __builtin_add_overflow(aggrs0[idx].sum_disc_price, disc_price, (int128_t*)&aggrs0[idx].sum_disc_price);
					flag |= __builtin_add_overflow(aggrs0[idx].sum_charge, charge, (int128_t*)&aggrs0[idx].sum_charge);

					flag |= __builtin_saddll_overflow(aggrs0[idx].sum_disc, disc, (long long int*)&aggrs0[idx].sum_disc);
					flag |= __builtin_uaddll_overflow(aggrs0[idx].count, 1, (long long unsigned int*)&aggrs0[idx].count);
				}
			}

			if (flag) {
				handle_overflow();
			}
		}

		kernel_epilogue();
	}
};

#endif
