#ifndef H_KERNEL_X100
#define H_KERNEL_X100

#include "common.hpp"

enum AggrFlavour {
	k1Step, kMultiplePrims, kMagic, kMagicFused, kNoAggr
};

enum Avx512Flavour {
	kNoAvx512, kCompare, kPopulationCount
};

template<AggrFlavour aggr_flavour, bool nsm, Avx512Flavour avx512 = kNoAvx512>
struct KernelX100 : BaseKernel {
	static constexpr size_t kVectorsize = MAX_VSIZE;

	idx_t* RESTRICT pos;
	idx_t* RESTRICT lim;
	idx_t* RESTRICT grp;

	uint16_t** grppos;
	uint16_t* selbuf;

	int16_t* RESTRICT v_shipdate;
	int8_t* RESTRICT v_returnflag;
	int8_t* RESTRICT v_linestatus;
	int8_t* RESTRICT v_discount;
	int8_t* RESTRICT v_tax;
	int32_t* RESTRICT v_extendedprice;
	int16_t* RESTRICT v_quantity;

	int8_t* RESTRICT v_disc_1;
	int8_t* RESTRICT v_tax_1;

	idx_t* RESTRICT v_idx; // TODO: make int16_t
	int32_t* RESTRICT v_disc_price;
	int64_t*  RESTRICT v_charge;
	sel_t* RESTRICT v_sel;

	kernel_compact_declare

	#define scan(name) v_##name = (l_##name);
	#define scan_epilogue(name) v_##name += chunk_size;

	KernelX100(const lineitem& li) : BaseKernel(li) {
		grppos = new_array<uint16_t*>(kGrpPosSize);
		selbuf = new_array<uint16_t>(kSelBufSize);
		pos = new_array<idx_t>(kVectorsize);
		lim = new_array<idx_t>(kVectorsize);
		grp = new_array<idx_t>(kVectorsize);

		kernel_compact_init();

		v_shipdate = new_array<int16_t>(kVectorsize);

		v_returnflag = new_array<int8_t>(kVectorsize);
		v_linestatus = new_array<int8_t>(kVectorsize);

		v_shipdate = new_array<int16_t>(kVectorsize);
		v_discount = new_array<int8_t>(kVectorsize);

		v_tax = new_array<int8_t>(kVectorsize);
		v_extendedprice = new_array<int32_t>(kVectorsize);

		v_quantity = new_array<int16_t>(kVectorsize);



		v_disc_1 = new_array<int8_t>(kVectorsize);
		v_tax_1 = new_array<int8_t>(kVectorsize);

		v_idx = new_array<idx_t>(kVectorsize);
		v_disc_price = new_array<int32_t>(kVectorsize);
		v_charge = new_array<int64_t>(kVectorsize);

		v_sel = new_array<sel_t>(kVectorsize);
	}

    struct ExprProf {
#ifdef PROFILE
        size_t tuples = 0;
        size_t time = 0;
#endif
    };
    
    ExprProf prof_select, prof_map_gid, prof_map_disc_1,
    	prof_map_tax_1, prof_map_disc_price, prof_map_charge,
    	prof_aggr_quantity, prof_aggr_base_price, prof_aggr_disc_price,
		prof_aggr_charge, prof_aggr_disc, prof_aggr_count;

#ifdef PROFILE
	int64_t prof_num_full_aggr = 0;
	int64_t prof_num_strides = 0;
#endif

    template<typename T>
    auto ProfileLambda(ExprProf& prof, size_t tuples, T&& fun) {
#ifdef PROFILE
        prof.tuples += tuples;
        auto begin = rdtsc();
        auto result = fun();        
        prof.time += rdtsc() - begin;
        return std::move(result);
#else
        return fun();
#endif
    }
    
	void Profile(size_t total_tuples) override {
#ifdef PROFILE
#define p(name) do { \
		ExprProf& d = prof_##name; \
		if (d.tuples > 0) { printf("%s %f\n", #name, (float)d.time / (float)d.tuples); } \
	} while(false)

		p(select);
		p(map_gid);
		p(map_disc_1);
		p(map_tax_1);
        p(map_disc_price);
        p(map_charge);

        p(aggr_quantity);
		p(aggr_base_price);
		p(aggr_disc_price);
		p(aggr_charge);
		p(aggr_disc);
		p(aggr_count);

		printf("aggr on full vector (no tuples filtered) %d/%d\n", prof_num_full_aggr, prof_num_strides);
#endif
	}

	NOINL void operator()() {
		kernel_prologue();

		/* Ommitted vector allocation on stack, 
		 * because C++ compilers will produce invalid results together with magic_preaggr (d270d85b8dcef5f295b1c10d4b2336c9be858541)
		 * Moving allocations to class fixed these issues which will be triggered with O1, O2 and O3 */
		
		const int16_t date = cmp.dte_val;
		const int8_t int8_t_one_discount = (int8_t)Decimal64::ToValue(1, 0);
		const int8_t int8_t_one_tax = (int8_t)Decimal64::ToValue(1, 0);

		scan(shipdate);
		scan(returnflag);
		scan(linestatus);
		scan(discount);
		scan(tax);
		scan(extendedprice);
		scan(quantity);


		size_t done=0;
		while (done < cardinality) {
			sel_t* sel = v_sel;
			const size_t chunk_size = min(kVectorsize, cardinality - done);

			size_t n = chunk_size;

			const size_t num = ProfileLambda(prof_select, n,
				[&] () { 
					if (avx512 == kNoAvx512) {
						return Primitives::select_int16_t(sel, nullptr, n, false, v_shipdate, date);
					} else {
						return Primitives::select_int16_t_avx512(sel, nullptr, n, false, v_shipdate, date);
					}
				});

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

			if (num > kVectorsize / 2) {
				if (num != n)
					n = sel[num-1]+1;
				sel = nullptr;
			} else {
				n = num;
			}

#ifdef PROFILE
			const auto prof_sc_start = rdtsc();
#endif

			ProfileLambda(prof_map_gid, n, [&] () {
				if (avx512 == kNoAvx512) {
					/* Faster version of "Primitives::map_gid(v_idx, sel, n, v_returnflag, v_linestatus);"
					 * But current version cannot print the groupids properly.
					 * Anyway this primitive behaves terrible on KNL as well as the hand-optimized ones (6 cycles/tuple)
					 */
					return Primitives::map_gid2_dom_restrict(v_idx, sel, n, v_returnflag, li.l_returnflag.minmax.min, li.l_returnflag.minmax.max,
						v_linestatus, li.l_linestatus.minmax.min, li.l_linestatus.minmax.max);
				} else {
					return Primitives::map_gid(v_idx, sel, n, v_returnflag, v_linestatus);
				}
			});

			ProfileLambda(prof_map_disc_1, n, [&] () {
				return Primitives::map_disc_1(v_disc_1, sel, n, int8_t_one_discount, v_discount);
			});

			ProfileLambda(prof_map_tax_1, n, [&] () {
				return Primitives::map_tax_1(v_tax_1, sel, n, v_tax, int8_t_one_tax);
			});

			ProfileLambda(prof_map_disc_price, n, [&] () {
				return Primitives::map_disc_price(v_disc_price, sel, n, v_disc_1, v_extendedprice);			
			});

			ProfileLambda(prof_map_charge, n, [&] () {
				return Primitives::map_charge(v_charge, v_disc_price, v_tax_1, sel, n);
			});

#ifdef PROFILE
			const auto prof_ag_start = rdtsc();
#endif

			sel_t* aggr_sel = num == chunk_size ? nullptr /* full vector, nothing filtered */ : v_sel;
#ifdef PROFILE
			prof_num_full_aggr += num == chunk_size;
			prof_num_strides++;
#endif
			switch (aggr_flavour) {
			case kNoAggr:
				break;

			case kMagicFused:
			case kMagic: {
				auto gp0 = grppos;
				auto gp1 = grppos + kGrpPosSize/2;
				auto sb0 = selbuf;
				auto sb1 = selbuf + kSelBufSize/2;

#ifdef PROFILE
				const auto prof_magic_start = rdtsc();
#endif

				size_t num_groups;
				switch (avx512) {
				case kNoAvx512:
					num_groups = Primitives::partial_shuffle_scalar(v_idx, aggr_sel, num, pos, lim, grp, gp0, gp1, sb0, sb1);
					break;
				case kCompare:
					num_groups = Primitives::partial_shuffle_avx512_cmp(v_idx, v_sel, num, pos, lim, grp, gp0, gp1, sb0, sb1);
					break;
				case kPopulationCount:
					num_groups = Primitives::partial_shuffle_avx512(v_idx, v_sel, num, pos, lim, grp, gp0, gp1, sb0, sb1);
					break;
				}

#ifdef PROFILE
				sum_magic_time += rdtsc() - prof_magic_start;
#endif
				/* pre-aggregate */
				if (aggr_flavour == kMagicFused) {
					Primitives::ordaggr_all_in_one(aggrs0, pos, lim, grp, num_groups, v_quantity, v_extendedprice, v_disc_price, v_charge, v_disc_1);
				} else {
					#define aggregate(prof, ag, vec) do { \
							ProfileLambda(prof_aggr_##prof, n, [&] () { \
								return avx512 == kNoAvx512 ? \
									Primitives::ordaggr_##ag(aggrs0, pos, lim, grp, num_groups, vec) : \
									Primitives::par_ordaggr_##ag(aggrs0, pos, lim, grp, num_groups, vec); \
							}); \
						} while (false)

					aggregate(quantity, quantity, v_quantity);
					aggregate(base_price, extended_price, v_extendedprice);
					aggregate(disc_price, disc_price, v_disc_price);
					aggregate(charge, charge, v_charge);
					aggregate(disc, disc, v_disc_1);

					ProfileLambda(prof_aggr_count, n, [&] () {
						return Primitives::ordaggr_count(aggrs0, pos, lim, grp, num_groups);
					});

					#undef aggregate
				}

				break; // kMagic
			}

			case k1Step:
				Primitives::for_each(aggr_sel, num, [&] (auto i) {
					const auto g = v_idx[i];
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
				ProfileLambda(prof_aggr_quantity, n, [&] () {
					return Primitives::for_each(aggr_sel, num, [&] (auto i) {
						auto g = v_idx[i];
						if (nsm) {
							aggrs0[g].sum_quantity += v_quantity[i];
						} else {
							aggr_dsm0_sum_quantity[g] += v_quantity[i];
						}
					});		
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

#ifdef PROFILE
			sum_aggr_time += rdtsc() - prof_ag_start;
#endif
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