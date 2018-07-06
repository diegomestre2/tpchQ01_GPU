#ifndef H_AGGR_BENCH
#define H_AGGR_BENCH

#include <sstream>
#include <vector>
#include <time.h>
#include "common.hpp"
#include "vectorized.hpp"
#include "tpch_kit.hpp"

#include <tuple>
#include <random>

constexpr int64_t kSeed = 42;

#define NUM_REPS 25
constexpr size_t CARD = 100000000; // 100 million

// #define DEBUG

enum AggrType {
	kDirect,
	kDirectFused,
	kInRegister,
	kInRegisterFused,
};

struct StaticColumn : Column<int64_t> {
	StaticColumn() : Column(CARD) {}
};

typedef StaticColumn col_t;

struct Data {
	static constexpr size_t NUM_AGGRS = 10;

	col_t aggr_cols[NUM_AGGRS];
	Column<idx_t> aggr_grp;

	Data() : aggr_grp(CARD) {
		// hope this happens after the initialization (unspecified in cpp)
		for (size_t k=0; k<NUM_AGGRS; k++) {
			col_t* col = &aggr_cols[k];

			for (size_t i=0; i<CARD; i++) {
				col->Push(i + k);
			}
		}

		for (size_t i=0; i<CARD; i++) {
			aggr_grp.Push(1);
		}
	}

};

static Data data;

template<AggrType aggr_type, size_t NUM_AGGRS, size_t NUM_GROUPS>struct AggrBench : IKernel {

	static constexpr size_t kVectorsize = 1024;
	static constexpr size_t kSpread = 1024;

	static constexpr size_t GetBits(size_t f) {
		size_t i=0;
		for (; (((size_t) 1) << i) < f; i++);
		return i;
	}

	static constexpr size_t kVectorbits = GetBits(kVectorsize);

	struct AggregateItem {
		int64_t aggrs[NUM_AGGRS];
	};

	const size_t aggr_table_size = kSpread * NUM_GROUPS;
	AggregateItem* aggr_table;

	idx_t* RESTRICT pos;
	idx_t* RESTRICT lim;
	idx_t* RESTRICT grp;

	uint16_t** grppos;
	uint16_t* selbuf;

	std::default_random_engine generator;
	std::uniform_int_distribution<idx_t> distribution;

	size_t prof_prepare = 0;
	size_t prof_aggr = 0;

	AggrBench() : distribution(1,NUM_GROUPS) {
		generator.seed(kSeed);

		grppos = new_array<uint16_t*>(kGrpPosSize);
		selbuf = new_array<uint16_t>(kSelBufSize);
		pos = new_array<idx_t>(kVectorsize);
		lim = new_array<idx_t>(kVectorsize);
		grp = new_array<idx_t>(kVectorsize);

		aggr_table = new_array<AggregateItem>(aggr_table_size);

		auto gids = data.aggr_grp.get();
		for (size_t i=0; i<CARD; i++) {
			gids[i] = kSpread * (distribution(generator) - 1);
		}
	}

	void NOINL aggr_direct(sel_t* RESTRICT aggr_sel, int num, int64_t ag, int64_t* RESTRICT col, idx_t* RESTRICT gid) {
		Primitives::for_each(aggr_sel, num, [&] (auto i) {
#ifdef DEBUG
			assert(gid[i] >= 0);
			assert(gid[i] <= kSpread * NUM_GROUPS);
#endif			
			aggr_table[gid[i]].aggrs[ag] += col[i];
		});
	}

	void NOINL ordaggr(idx_t* RESTRICT pos, idx_t* RESTRICT lim, idx_t* RESTRICT grp, int num_groups, int num, int64_t ag, int64_t* RESTRICT col) {
		int64_t partial;
#if 0
		for (size_t i=0; i<num; i++) {
			assert(pos[i] >= 0);
			assert(pos[i] < num);
		}
#endif
		Primitives::ordaggr(pos, lim, grp, num_groups,
			[&] (auto g) { partial = 0; },
			[&] (auto g, auto i) { partial += col[i]; },
			[&] (auto group_idx) {
				auto gid = grp[group_idx];

#ifdef DEBUG
				assert(gid >= 0);
				assert(gid <= kSpread * NUM_GROUPS);
#endif			
				aggr_table[gid].aggrs[ag] += partial;
			});		
	}

	void NOINL ordfused(idx_t* RESTRICT pos, idx_t* RESTRICT lim, idx_t* RESTRICT grp, int num_groups, int num, int64_t* RESTRICT col0, int64_t* RESTRICT col1) {
		int64_t p0, p1;
		Primitives::ordaggr(pos, lim, grp, num_groups,
			[&] (auto g) { p0 = 0; p1 = 0; },
			[&] (auto g, auto i) { p0 += col0[i]; p1 += col1[i]; },
			[&] (auto group_idx) {
				auto gid = grp[group_idx];
				aggr_table[gid].aggrs[0] += p0;
				aggr_table[gid].aggrs[1] += p1;
			});		
	}

	void NOINL ordfused(idx_t* RESTRICT pos, idx_t* RESTRICT lim, idx_t* RESTRICT grp, int num_groups, int num, int64_t* RESTRICT col0, int64_t* RESTRICT col1, int64_t* RESTRICT col2) {
		int64_t p0, p1, p2;
		Primitives::ordaggr(pos, lim, grp, num_groups,
			[&] (auto g) { p0 = 0; p1 = 0; p2 = 0; },
			[&] (auto g, auto i) { p0 += col0[i]; p1 += col1[i]; p2 += col2[i]; },
			[&] (auto group_idx) {
				auto gid = grp[group_idx];
				aggr_table[gid].aggrs[0] += p0;
				aggr_table[gid].aggrs[1] += p1;
				aggr_table[gid].aggrs[2] += p2;
			});		
	}

	void NOINL ordfused(idx_t* RESTRICT pos, idx_t* RESTRICT lim, idx_t* RESTRICT grp, int num_groups, int num, int64_t* RESTRICT col0, int64_t* RESTRICT col1, int64_t* RESTRICT col2, int64_t* RESTRICT col3) {
		int64_t p0, p1, p2, p3;
		Primitives::ordaggr(pos, lim, grp, num_groups,
			[&] (auto g) { p0 = 0; p1 = 0; p2 = 0; p3 = 0; },
			[&] (auto g, auto i) { p0 += col0[i]; p1 += col1[i]; p2 += col2[i]; p3 += col3[i]; },
			[&] (auto group_idx) {
				auto gid = grp[group_idx];
				aggr_table[gid].aggrs[0] += p0;
				aggr_table[gid].aggrs[1] += p1;
				aggr_table[gid].aggrs[2] += p2;
				aggr_table[gid].aggrs[3] += p3;
			});		
	}

	void NOINL ordfused(idx_t* RESTRICT pos, idx_t* RESTRICT lim, idx_t* RESTRICT grp, int num_groups, int num, int64_t* RESTRICT col0, int64_t* RESTRICT col1, int64_t* RESTRICT col2, int64_t* RESTRICT col3, int64_t* RESTRICT col4) {
		int64_t p0, p1, p2, p3, p4;
		Primitives::ordaggr(pos, lim, grp, num_groups,
			[&] (auto g) { p0 = 0; p1 = 0; p2 = 0; p3 = 0; p4 = 0; },
			[&] (auto g, auto i) { p0 += col0[i]; p1 += col1[i]; p2 += col2[i]; p3 += col3[i]; p4 += col4[i]; },
			[&] (auto group_idx) {
				auto gid = grp[group_idx];
				aggr_table[gid].aggrs[0] += p0;
				aggr_table[gid].aggrs[1] += p1;
				aggr_table[gid].aggrs[2] += p2;
				aggr_table[gid].aggrs[3] += p3;
				aggr_table[gid].aggrs[4] += p4;
			});		
	}

	void NOINL ordfused(idx_t* RESTRICT pos, idx_t* RESTRICT lim, idx_t* RESTRICT grp, int num_groups, int num,
			int64_t* RESTRICT col0, int64_t* RESTRICT col1, int64_t* RESTRICT col2, int64_t* RESTRICT col3,
			int64_t* RESTRICT col4, int64_t* RESTRICT col5) {
		int64_t p0, p1, p2, p3, p4, p5;
		Primitives::ordaggr(pos, lim, grp, num_groups,
			[&] (auto g) { p0 = 0; p1 = 0; p2 = 0; p3 = 0; p4 = 0; p5 = 0; },
			[&] (auto g, auto i) { p0 += col0[i]; p1 += col1[i]; p2 += col2[i]; p3 += col3[i]; p4 += col4[i]; p5 += col5[i]; },
			[&] (auto group_idx) {
				auto gid = grp[group_idx];
				aggr_table[gid].aggrs[0] += p0;
				aggr_table[gid].aggrs[1] += p1;
				aggr_table[gid].aggrs[2] += p2;
				aggr_table[gid].aggrs[3] += p3;
				aggr_table[gid].aggrs[4] += p4;
				aggr_table[gid].aggrs[5] += p5;
			});		
	}

	void NOINL ordfused(idx_t* RESTRICT pos, idx_t* RESTRICT lim, idx_t* RESTRICT grp, int num_groups, int num,
			int64_t* RESTRICT col0, int64_t* RESTRICT col1, int64_t* RESTRICT col2, int64_t* RESTRICT col3,
			int64_t* RESTRICT col4, int64_t* RESTRICT col5, int64_t* RESTRICT col6) {
		int64_t p0, p1, p2, p3, p4, p5, p6;
		Primitives::ordaggr(pos, lim, grp, num_groups,
			[&] (auto g) { p0 = 0; p1 = 0; p2 = 0; p3 = 0; p4 = 0; p5 = 0; p6 = 0; },
			[&] (auto g, auto i) { p0 += col0[i]; p1 += col1[i]; p2 += col2[i]; p3 += col3[i]; p4 += col4[i]; p5 += col5[i]; p6 += col6[i]; },
			[&] (auto group_idx) {
				auto gid = grp[group_idx];
				aggr_table[gid].aggrs[0] += p0;
				aggr_table[gid].aggrs[1] += p1;
				aggr_table[gid].aggrs[2] += p2;
				aggr_table[gid].aggrs[3] += p3;
				aggr_table[gid].aggrs[4] += p4;
				aggr_table[gid].aggrs[5] += p5;
				aggr_table[gid].aggrs[6] += p6;
			});		
	}

	void NOINL ordfused(idx_t* RESTRICT pos, idx_t* RESTRICT lim, idx_t* RESTRICT grp, int num_groups, int num,
			int64_t* RESTRICT col0, int64_t* RESTRICT col1, int64_t* RESTRICT col2, int64_t* RESTRICT col3,
			int64_t* RESTRICT col4, int64_t* RESTRICT col5, int64_t* RESTRICT col6, int64_t* RESTRICT col7) {
		int64_t p0, p1, p2, p3, p4, p5, p6, p7;
		Primitives::ordaggr(pos, lim, grp, num_groups,
			[&] (auto g) { p0 = 0; p1 = 0; p2 = 0; p3 = 0; p4 = 0; p5 = 0; p6 = 0; p7 = 0; },
			[&] (auto g, auto i) { p0 += col0[i]; p1 += col1[i]; p2 += col2[i]; p3 += col3[i]; p4 += col4[i]; p5 += col5[i]; p6 += col6[i]; p7 += col7[i]; },
			[&] (auto group_idx) {
				auto gid = grp[group_idx];
				aggr_table[gid].aggrs[0] += p0;
				aggr_table[gid].aggrs[1] += p1;
				aggr_table[gid].aggrs[2] += p2;
				aggr_table[gid].aggrs[3] += p3;
				aggr_table[gid].aggrs[4] += p4;
				aggr_table[gid].aggrs[5] += p5;
				aggr_table[gid].aggrs[6] += p6;
				aggr_table[gid].aggrs[7] += p7;
			});		
	}

	void NOINL ordfused(idx_t* RESTRICT pos, idx_t* RESTRICT lim, idx_t* RESTRICT grp, int num_groups, int num,
			int64_t* RESTRICT col0, int64_t* RESTRICT col1, int64_t* RESTRICT col2, int64_t* RESTRICT col3,
			int64_t* RESTRICT col4, int64_t* RESTRICT col5, int64_t* RESTRICT col6, int64_t* RESTRICT col7,
			int64_t* RESTRICT col8) {
		int64_t p0, p1, p2, p3, p4, p5, p6, p7, p8;
		Primitives::ordaggr(pos, lim, grp, num_groups,
			[&] (auto g) { p0 = 0; p1 = 0; p2 = 0; p3 = 0; p4 = 0; p5 = 0; p6 = 0; p7 = 0; p8 = 0; },
			[&] (auto g, auto i) { p0 += col0[i]; p1 += col1[i]; p2 += col2[i]; p3 += col3[i]; p4 += col4[i]; p5 += col5[i]; p6 += col6[i]; p7 += col7[i]; p8 += col8[i]; },
			[&] (auto group_idx) {
				auto gid = grp[group_idx];
				aggr_table[gid].aggrs[0] += p0;
				aggr_table[gid].aggrs[1] += p1;
				aggr_table[gid].aggrs[2] += p2;
				aggr_table[gid].aggrs[3] += p3;
				aggr_table[gid].aggrs[4] += p4;
				aggr_table[gid].aggrs[5] += p5;
				aggr_table[gid].aggrs[6] += p6;
				aggr_table[gid].aggrs[7] += p7;
				aggr_table[gid].aggrs[8] += p8;
			});		
	}

	void NOINL ordfused(idx_t* RESTRICT pos, idx_t* RESTRICT lim, idx_t* RESTRICT grp, int num_groups, int num,
			int64_t* RESTRICT col0, int64_t* RESTRICT col1, int64_t* RESTRICT col2, int64_t* RESTRICT col3,
			int64_t* RESTRICT col4, int64_t* RESTRICT col5, int64_t* RESTRICT col6, int64_t* RESTRICT col7,
			int64_t* RESTRICT col8, int64_t* RESTRICT col9) {
		int64_t p0, p1, p2, p3, p4, p5, p6, p7, p8, p9;
		Primitives::ordaggr(pos, lim, grp, num_groups,
			[&] (auto g) { p0 = 0; p1 = 0; p2 = 0; p3 = 0; p4 = 0; p5 = 0; p6 = 0; p7 = 0; p8 = 0; p9 = 0; },
			[&] (auto g, auto i) { p0 += col0[i]; p1 += col1[i]; p2 += col2[i]; p3 += col3[i]; p4 += col4[i]; p5 += col5[i]; p6 += col6[i]; p7 += col7[i]; p8 += col8[i]; p9 += col9[i]; },
			[&] (auto group_idx) {
				auto gid = grp[group_idx];
				aggr_table[gid].aggrs[0] += p0;
				aggr_table[gid].aggrs[1] += p1;
				aggr_table[gid].aggrs[2] += p2;
				aggr_table[gid].aggrs[3] += p3;
				aggr_table[gid].aggrs[4] += p4;
				aggr_table[gid].aggrs[5] += p5;
				aggr_table[gid].aggrs[6] += p6;
				aggr_table[gid].aggrs[7] += p7;
				aggr_table[gid].aggrs[8] += p8;
				aggr_table[gid].aggrs[9] += p9;
			});		
	}


	void prepare() {
		memset(aggr_table, 0, sizeof(AggregateItem) * aggr_table_size);

		prof_prepare = 0;
		prof_aggr = 0;
	}

	void NOINL operator()() {
		int64_t* cols[NUM_AGGRS];

		auto gp0 = grppos;
		auto gp1 = grppos + kGrpPosSize/2;
		auto sb0 = selbuf;
		auto sb1 = selbuf + kSelBufSize/2;

		// set initial pointers
		for (size_t k=0; k<NUM_AGGRS; k++) {
			cols[k] = data.aggr_cols[k].get();
		}
		auto gid = data.aggr_grp.get();


		size_t done=0;
		while (done < CARD) {
			size_t num_groups;
			const size_t chunk_size = min(kVectorsize, CARD - done);

			size_t n = chunk_size;


			size_t num = n;

			size_t prof_start;

			switch (aggr_type) {
			case kDirect:
				prof_start = get_cycles();
				for (size_t ag = 0; ag < NUM_AGGRS; ag++) {
					aggr_direct(nullptr, num, ag, cols[ag], gid);
				}
				prof_aggr += get_cycles() - prof_start;
				break;

			case kDirectFused:
				prof_start = get_cycles();
				Primitives::for_each(nullptr, num, [&] (auto i) {
					for (size_t ag = 0; ag < NUM_AGGRS; ag++) {
						auto col = cols[ag];
						aggr_table[gid[i]].aggrs[ag] += col[i];
					}
				});
				prof_aggr += get_cycles() - prof_start;
				break;				

			case kInRegister:
				prof_start = get_cycles();
				num_groups = Primitives::partial_shuffle_scalar(gid, nullptr, num, pos, lim, grp, gp0, gp1, sb0, sb1);
				{
					auto tmp = get_cycles();
					prof_prepare += tmp - prof_start;
					prof_start = tmp;
				}
				for (size_t ag = 0; ag < NUM_AGGRS; ag++) {
					ordaggr(pos, lim, grp, num_groups, num, ag, cols[ag]);
				}
				prof_aggr += get_cycles() - prof_start;
				break;

			case kInRegisterFused:
				prof_start = get_cycles();
				num_groups = Primitives::partial_shuffle_scalar(gid, nullptr, num, pos, lim, grp, gp0, gp1, sb0, sb1);
				{
					auto tmp = get_cycles();
					prof_prepare += tmp - prof_start;
					prof_start = tmp;
				}
				switch (NUM_AGGRS) {
				case 1: ordaggr(pos, lim, grp, num_groups, num, 0, cols[0]); break;
				case 2: ordfused(pos, lim, grp, num_groups, num, cols[0], cols[1]); break;
				case 3: ordfused(pos, lim, grp, num_groups, num, cols[0], cols[1], cols[2]); break;
				case 4: ordfused(pos, lim, grp, num_groups, num, cols[0], cols[1], cols[2], cols[3]); break;
				case 5: ordfused(pos, lim, grp, num_groups, num, cols[0], cols[1], cols[2], cols[3], cols[4]); break;
				case 6: ordfused(pos, lim, grp, num_groups, num, cols[0], cols[1], cols[2], cols[3], cols[4], cols[5]); break;
				case 7: ordfused(pos, lim, grp, num_groups, num, cols[0], cols[1], cols[2], cols[3], cols[4], cols[5], cols[6]); break;
				case 8: ordfused(pos, lim, grp, num_groups, num, cols[0], cols[1], cols[2], cols[3], cols[4], cols[5], cols[6], cols[7]); break;
				case 9: ordfused(pos, lim, grp, num_groups, num, cols[0], cols[1], cols[2], cols[3], cols[4], cols[5], cols[6], cols[7], cols[8]); break;
				case 10: ordfused(pos, lim, grp, num_groups, num, cols[0], cols[1], cols[2], cols[3], cols[4], cols[5], cols[6], cols[7], cols[8], cols[9]); break;
				}
				prof_aggr += get_cycles() - prof_start;
				break;

			default:
				assert(false);
				break;
			}

			// next range
			for (size_t k=0; k<NUM_AGGRS; k++) {
				cols[k] += chunk_size;
			}
			gid += chunk_size;

			done += chunk_size;
		}
	}

	void NOINL print() {
		for (size_t i=0; i<aggr_table_size; i++) {
			size_t k;
			int64_t sum = 0;
			for (k=0; k<NUM_AGGRS; k++) {
				sum += aggr_table[i].aggrs[k];	
			}
			if (sum) {
				printf("# %lld: ", i);
				for (k=0; k<NUM_AGGRS; k++) {
					printf("%s%lld", k ? ", " : "", aggr_table[i].aggrs[k]);
				}
				printf("\n");
			}
		}
	}

	void NOINL validate() {
		auto sum_first_n = [] (int64_t n) { return (n*(n+1)) / 2; };

		size_t k;

		int64_t sum_ag[NUM_AGGRS];
		memset(sum_ag, 0, sizeof(sum_ag));

		for (size_t i=0; i<aggr_table_size; i++) {
			for (k=0; k<NUM_AGGRS; k++) {
				sum_ag[k] += aggr_table[i].aggrs[k];	
			}
		}

		for (k=0; k<NUM_AGGRS; k++) {
			auto s = sum_first_n(CARD-1);
			auto a = CARD * k;
			auto chk = s + a;
			assert(sum_ag[k] == chk);
		}
	}
};

#define run(type, aggrs, groups) do { \
		printf("%s %lld %lld ", #type, aggrs, groups); \
		int64_t total_cycles = 0, total_millis = 0, total_aggr = 0, total_prepare = 0; \
		AggrBench<type, aggrs, groups> ag; \
		for (size_t rep=0; rep < NUM_REPS; rep++) { \
			timespec ts_start, ts_end; \
			ag.prepare(); \
			clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts_start); \
			const auto start = get_cycles(); \
			ag(); \
			const auto time = get_cycles() - start; \
			clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts_end); \
			double million = 1000000.0; \
			uint64_t millisec = (ts_end.tv_sec - ts_start.tv_sec) * 1000.0 + ((ts_end.tv_nsec - ts_start.tv_nsec) / million); \
			if (rep != 0) { /* Throw cold run away */ \
				total_cycles += time; total_millis += millisec; \
				total_prepare += ag.prof_prepare; total_aggr += ag.prof_aggr; \
			} \
		} \
		const size_t hot_reps = (NUM_REPS-1); \
		printf("%lld %lld %lld %lld\n", total_millis / hot_reps, total_cycles / hot_reps, total_prepare / hot_reps, total_aggr / hot_reps); \
		/* ag.print(); */ \
		/* ag.validate(); */ \
	} while (false)	

#define run_till_10_aggrs(type, g) run(type, 1, g); run(type, 2, g);  run(type, 3, g); run(type, 4, g);  run(type, 5, g); run(type, 6, g); run(type, 7, g);  run(type, 8, g); run(type, 9, g);  run(type, 10, g);

#define run_till_2048_groups(t) run_till_10_aggrs(t, 1); run_till_10_aggrs(t, 2); run_till_10_aggrs(t, 4);  run_till_10_aggrs(t, 8); run_till_10_aggrs(t, 16); run_till_10_aggrs(t, 32); run_till_10_aggrs(t, 64);


#endif
