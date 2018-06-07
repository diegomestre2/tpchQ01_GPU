#ifndef H_KERNEL_NAIVE_COMPACT
#define H_KERNEL_NAIVE_COMPACT

#include "common.hpp"

struct KernelNaiveCompact : BaseKernel {
	kernel_compact_declare

	KernelNaiveCompact(const lineitem& li) : BaseKernel(li) {
		kernel_compact_init();
	}


	typedef struct { uint64_t lo; uint64_t hi; } emu128;

	template<typename T>
	void sum_128(__int128_t* dest, T val) {
		emu128* d = (emu128*)dest;
		emu128 tmp = *d;

		tmp.lo += val;

		if (UNLIKELY(tmp.lo < val)) {
			tmp.hi++;
			*d = tmp;
		} else {
			int64_t* d64 = (int64_t*)dest;
			*d64 = tmp.lo;
		}
	}



	__attribute__((noinline)) void operator()() noexcept {
		kernel_prologue();

		return task(0, cardinality);
	}

	NOINL void task(size_t offset, size_t morsel_num) {
		//kernel_small_dt_prologue();
		kernel_prologue();

		auto o_sd = l_shipdate + offset;
		auto o_disc = l_discount + offset;
		auto o_eprice = l_extendedprice + offset;
		auto o_quantity = l_quantity + offset;
		tax += offset;
		returnflag += offset;
		linestatus += offset;

		const int8_t one = Decimal64::ToValue(1, 0);
		const int16_t date = cmp.dte_val;

		assert(offset < cardinality);

		for (size_t i=0; i<morsel_num; i++) {
			if (o_sd[i] <= date) {
				const auto disc = o_disc[i];
				const auto price = o_eprice[i];
				const int8_t disc_1 = one - disc;
				const int8_t tax_1 = tax[i] + one;
				const int32_t disc_price = disc_1 * price;
				const int64_t charge = (int64_t)disc_price * tax_1; // seems weird but apparently this triggers a multiplication with 64-bit result
				const int16_t idx = (int16_t)(returnflag[i] << 8) | linestatus[i];
				aggrs0[idx].sum_quantity += (int64_t)o_quantity[i];
				aggrs0[idx].sum_base_price += (int64_t)price;
				aggrs0[idx].sum_disc_price = int128_add64(aggrs0[idx].sum_disc_price, disc_price);
				aggrs0[idx].sum_charge = int128_add64(aggrs0[idx].sum_charge, charge);	
				aggrs0[idx].sum_disc += disc;
				aggrs0[idx].count++;
			}
		}
	}
};



#endif