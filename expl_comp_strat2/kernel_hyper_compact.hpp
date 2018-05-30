#ifndef H_KERNEL_HYPER_COMPACT
#define H_KERNEL_HYPER_COMPACT

#include "common.hpp"

template<bool use_flag>
struct KernelHyPerCompact : BaseKernel {
	KernelHyPerCompact(const lineitem& li) : BaseKernel(li) {
		kernel_compact_init();
	}

	kernel_compact_declare

	__attribute__((noinline)) void operator()() noexcept {
		kernel_prologue();

		const int8_t one = Decimal64::ToValue(1, 0);
		const int16_t date = cmp.dte_val;

		int8_t disc_1;
		int8_t tax_1;
		int32_t disc_price;
		int64_t charge;

		if (!use_flag) {
			for (size_t i=0; i<cardinality; i++) {
				if (l_shipdate[i] <= date) {
					const auto disc = l_discount[i];
					const auto price = l_extendedprice[i];

					if (UNLIKELY(__builtin_sub_overflow(one, disc, &disc_1))) {
						handle_overflow();
					}
					
					if (UNLIKELY(__builtin_add_overflow(one, l_tax[i], &tax_1))) {
						handle_overflow();
					}

					if (UNLIKELY(__builtin_mul_overflow(disc_1, price, &disc_price))) {
						handle_overflow();
					}
					if (UNLIKELY(__builtin_mul_overflow(disc_price, tax_1, &charge))) {
						handle_overflow();
					}
					const int16_t idx = (int16_t)(returnflag[i] << 8) | linestatus[i];
					if (UNLIKELY(__builtin_add_overflow(aggrs0[idx].sum_quantity, l_quantity[i], &aggrs0[idx].sum_quantity))) {
						handle_overflow();
					}
					if (UNLIKELY(__builtin_add_overflow(aggrs0[idx].sum_base_price, price, &aggrs0[idx].sum_base_price))) {
						handle_overflow();	
					}

					if (UNLIKELY(__builtin_add_overflow(aggrs0[idx].sum_disc_price, disc_price, &aggrs0[idx].sum_disc_price))) {
						handle_overflow();	
					}

					if (UNLIKELY(__builtin_add_overflow(aggrs0[idx].sum_charge, charge, &aggrs0[idx].sum_charge))) {
						handle_overflow();	
					}

					if (UNLIKELY(__builtin_add_overflow(aggrs0[idx].sum_disc, disc, &aggrs0[idx].sum_disc))) {
						handle_overflow();	
					}
					if (UNLIKELY(__builtin_add_overflow(aggrs0[idx].count, 1, &aggrs0[idx].count))) {
						handle_overflow();	
					}
				}
			}
		} else {
			bool flag = false;
			for (size_t i=0; i<cardinality; i++) {
				if (l_shipdate[i] <= date) {
					const auto disc = l_discount[i];
					const auto price = l_extendedprice[i];

					flag |= __builtin_sub_overflow(one, disc, &disc_1);
					flag |= __builtin_add_overflow(one, l_tax[i], &tax_1);

					flag |= __builtin_mul_overflow(disc_1, price, &disc_price);
					flag |= __builtin_mul_overflow(disc_price, tax_1, &charge);

					const int16_t idx = (int16_t)(returnflag[i] << 8) | linestatus[i];
					flag |= __builtin_add_overflow(aggrs0[idx].sum_quantity, l_quantity[i], &aggrs0[idx].sum_quantity);
					flag |= __builtin_add_overflow(aggrs0[idx].sum_base_price, price, &aggrs0[idx].sum_base_price);

					flag |= __builtin_add_overflow(aggrs0[idx].sum_disc_price, disc_price, &aggrs0[idx].sum_disc_price);
					flag |= __builtin_add_overflow(aggrs0[idx].sum_charge, charge, &aggrs0[idx].sum_charge);

					flag |= __builtin_add_overflow(aggrs0[idx].sum_disc, disc, &aggrs0[idx].sum_disc);
					flag |= __builtin_add_overflow(aggrs0[idx].count, 1, &aggrs0[idx].count);
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
