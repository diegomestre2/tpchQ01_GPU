#ifndef H_KERNEL_X100_OLD
#define H_KERNEL_X100_OLD

#include "common.hpp"

enum ProjectFlavour {
	kSinglePrims
};

template<AggrFlavour aggr_flavour, bool nsm, ProjectFlavour project_flavour, bool magic = false>
struct KernelOldX100 : BaseKernel {
	static constexpr size_t kVectorsize = MAX_VSIZE;

	idx_t* pos;
	idx_t* lim;
	idx_t* grp;

	uint16_t** grppos;
	uint16_t* selbuf;

	decltype(li.l_shipdate.val)* RESTRICT v_shipdate;
	decltype(li.l_returnflag.val)* RESTRICT v_returnflag;
	decltype(li.l_linestatus.val)* RESTRICT v_linestatus;
	decltype(li.l_discount.val)* RESTRICT v_discount;
	decltype(li.l_tax.val)* RESTRICT v_tax;
	decltype(li.l_extendedprice.val)* RESTRICT v_extendedprice;
	decltype(li.l_quantity.val)* RESTRICT v_quantity;
	decltype(Decimal64::dec_val)* RESTRICT v_disc_1;
	decltype(Decimal64::dec_val)* RESTRICT v_tax_1;
	idx_t* RESTRICT v_idx;
	decltype(Decimal64::dec_val)* RESTRICT v_disc_price;
	decltype(Decimal64::dec_val)* RESTRICT v_charge;
	sel_t* RESTRICT v_sel;

	#define scan(name) v_##name = name;
	#define scan_epilogue(name) v_##name += chunk_size;

	KernelOldX100(const lineitem& li) : BaseKernel(li) {
		grppos = new_array<uint16_t*>(kGrpPosSize);
		selbuf = new_array<uint16_t>(kSelBufSize);
		pos = new_array<idx_t>(kVectorsize);
		lim = new_array<idx_t>(kVectorsize);
		grp = new_array<idx_t>(kVectorsize);

		v_shipdate = new_array<decltype(li.l_shipdate.val)>(kVectorsize);

		v_linestatus = new_array<decltype(li.l_linestatus.val)>(kVectorsize);
		v_returnflag = new_array<decltype(li.l_returnflag.val)>(kVectorsize);

		v_discount = new_array<decltype(li.l_discount.val)>(kVectorsize);

		v_tax = new_array<decltype(li.l_tax.val)>(kVectorsize);
		v_extendedprice = new_array<decltype(li.l_extendedprice.val)>(kVectorsize);

		v_quantity = new_array<decltype(li.l_quantity.val)>(kVectorsize);

		v_disc_1 = new_array<decltype(Decimal64::dec_val)>(kVectorsize);
		v_tax_1 = new_array<decltype(Decimal64::dec_val)>(kVectorsize);

		v_idx = new_array<idx_t>(kVectorsize);
		v_disc_price = new_array<decltype(Decimal64::dec_val)>(kVectorsize);
		v_charge = new_array<decltype(Decimal64::dec_val)>(kVectorsize);

		v_sel = new_array<sel_t>(kVectorsize);
	}

	void NOINL operator()() {
		kernel_prologue();

		/* Ommitted vector allocation on stack, 
		 * because C++ compilers will produce invalid results together with magic_preaggr (d270d85b8dcef5f295b1c10d4b2336c9be858541)
		 * Moving allocations to class fixed these issues which will be triggered with O1, O2 and O3 */

		scan(shipdate);
		scan(returnflag);
		scan(linestatus);
		scan(discount);
		scan(tax);
		scan(extendedprice);
		scan(quantity);

		const auto dec_one = Decimal64::ToValue(1, 0);

		
		size_t done=0;
		while (done < cardinality) {
			sel_t* sel = v_sel;
			const size_t chunk_size = min(kVectorsize, cardinality - done);

			size_t n = chunk_size;

			// select
			const size_t num = Primitives::select_int32_t(sel, nullptr, n, false, v_shipdate, cmp.dte_val);
			if (num > kVectorsize / 2) {
				if (num != n)
					n = sel[num-1]+1;
				sel = nullptr;
			} else {
				n = num;
			}

			if (!num) {
				scan_epilogue(returnflag);
				scan_epilogue(linestatus);
				scan_epilogue(shipdate);
				scan_epilogue(discount);
				scan_epilogue(tax);
				scan_epilogue(extendedprice);
				scan_epilogue(quantity);
				done += chunk_size;
				continue;
			}

			Primitives::old_map_gid(v_idx, sel, n, v_returnflag, v_linestatus);
			Primitives::old_map_disc_1(v_disc_1, sel, n, dec_one, v_discount);
			Primitives::old_map_tax_1(v_tax_1, sel, n, dec_one, v_tax);
			Primitives::old_map_mul(v_disc_price, sel, n, v_disc_1, v_extendedprice);
			Primitives::old_map_mul(v_charge, sel, n, v_disc_price, v_tax_1);

			const auto prof_ag_start = rdtsc();

			sel_t* aggr_sel = num == chunk_size ? nullptr : /* full vector, nothing filtered */ v_sel;

			switch (aggr_flavour) {
			case kMagic: {
				const auto prof_magic_start = rdtsc();
				const size_t num_groups = Primitives::partial_shuffle_scalar(v_idx, aggr_sel, num, pos, lim, grp,
					grppos, grppos + kGrpPosSize/2, selbuf, selbuf + kSelBufSize/2);
				sum_magic_time += rdtsc() - prof_magic_start;

				/* pre-aggregate */
				auto print_dec = [] (auto s, auto x) { printf("%s%ld.%ld", s, Decimal64::GetInt(x), Decimal64::GetFrac(x)); };

				int64_t ag_sum_quantity, ag_sum_base_price;
				int128_t ag_sum_disc_price; /* TODO: actually 128 bits long, but we can make it shorter due to partial aggr*/
				int128_t ag_sum_charge; /* TODO: actually 128 bits long, but we can make it shorter due to partial aggr*/
				int64_t ag_sum_disc, ag_count;

				Primitives::old_ordaggr_all_in_one(aggrs0, pos, lim, grp, num_groups, v_quantity, v_extendedprice, v_disc_price, v_charge, v_disc_1);
				break; // kMagic
			}

			case k1Step:
				Primitives::for_each(aggr_sel, num, [&] (auto i) {
					auto g = v_idx[i];
					if (nsm) {
						aggrs0[g].sum_quantity += v_quantity[i];
						aggrs0[g].sum_base_price += v_extendedprice[i];
						aggrs0[g].sum_disc_price = int128_add64(aggrs0[g].sum_disc_price, v_disc_price[i]);
						aggrs0[g].sum_charge = int128_add64(aggrs0[g].sum_charge, v_charge[i]);
						aggrs0[g].sum_disc += v_disc_1[i];
						aggrs0[g].count ++;
					} else {
						aggr_dsm0_sum_quantity[g] += v_quantity[i];
						aggr_dsm0_sum_base_price[g] += v_extendedprice[i];
						aggr_dsm0_sum_disc_price[g] = int128_add64(aggr_dsm0_sum_disc_price[g], v_disc_price[i]);
						aggr_dsm0_sum_charge[g] = int128_add64(aggr_dsm0_sum_charge[g], v_charge[i]);
						aggr_dsm0_sum_disc[g] += v_disc_1[i];
						aggr_dsm0_count[g] ++;
					}
				});
				break; // k1Step

			case kMultiplePrims:
				Primitives::for_each(aggr_sel, num, [&] (auto i) {
					auto g = v_idx[i];
					if (nsm) {
						aggrs0[g].sum_quantity += v_quantity[i];
					} else {
						aggr_dsm0_sum_quantity[g] += v_quantity[i];
					}
				});
				Primitives::for_each(aggr_sel, num, [&] (auto i) {
					auto g = v_idx[i];
					if (nsm) {
						aggrs0[g].sum_base_price += v_extendedprice[i];
					} else {
						aggr_dsm0_sum_base_price[g] += v_extendedprice[i];
					}
				});
				Primitives::for_each(aggr_sel, num, [&] (auto i) {
					auto g = v_idx[i];
					if (nsm) {
						aggrs0[g].sum_disc_price = int128_add64(aggrs0[g].sum_disc_price, v_disc_price[i]);
					} else {
						aggr_dsm0_sum_disc_price[g] = int128_add64(aggr_dsm0_sum_disc_price[g], v_disc_price[i]);
					}
				});
				Primitives::for_each(aggr_sel, num, [&] (auto i) {
					auto g = v_idx[i];
					if (nsm) {
						aggrs0[g].sum_charge = int128_add64(aggrs0[g].sum_charge, v_charge[i]);
					} else {
						aggr_dsm0_sum_charge[g] = int128_add64(aggr_dsm0_sum_charge[g], v_charge[i]);
					}
				});
				Primitives::for_each(aggr_sel, num, [&] (auto i) {
					auto g = v_idx[i];
					if (nsm) {
						aggrs0[g].sum_disc += v_disc_1[i];
					} else {
						aggr_dsm0_sum_disc[g] += v_disc_1[i];
					}
				});
				Primitives::for_each(aggr_sel, num, [&] (auto i) {
					auto g = v_idx[i];
					if (nsm) {
						aggrs0[g].count ++;
					} else {
						aggr_dsm0_count[g]++;
					}
				});
				break;
			}

			sum_aggr_time += rdtsc() - prof_ag_start;

			scan_epilogue(returnflag);
			scan_epilogue(linestatus);
			scan_epilogue(shipdate);
			scan_epilogue(discount);
			scan_epilogue(tax);
			scan_epilogue(extendedprice);
			scan_epilogue(quantity);
			done += chunk_size;
		};
	}

	#undef scan
	#undef scan_epilogue

};
#endif