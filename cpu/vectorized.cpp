#include "vectorized.hpp"

#ifdef __AVX512F__
#define HAND_OPT_CODE false
#else
#define HAND_OPT_CODE true
#endif

int
Primitives::partial_shuffle_scalar(idx_t* RESTRICT gids, sel_t* RESTRICT aggr_sel, int num, idx_t* RESTRICT sel,
	idx_t* RESTRICT lim, idx_t* RESTRICT grp, uint16_t** RESTRICT grppos0, uint16_t** RESTRICT grppos1, 
	uint16_t* RESTRICT selbuf0, uint16_t* RESTRICT selbuf1)
{
	#define declare(n) \
		uint16_t* RESTRICT buf_ins##n=selbuf##n; \
		const uint16_t* RESTRICT buf_end##n = selbuf##n + GROUP_BUF_SIZE; \
		uint16_t buf##n[MAX_ACTIVE_GROUPS]; \
		uint16_t* RESTRICT max_gid##n = buf##n;

	declare(0);
	declare(1);

	/* Reduce TLB thrashing by removing positional access on vector 'gids' */
	gids = Primitives::desel(sel, aggr_sel, gids, num);

	#define new_group(n, dstpos) \
		do { \
			if (UNLIKELY(buf_ins##n >= buf_end##n)) { \
				printf("gid %d -> %d, %d\n", gid, gid & 0xFF, gid >> 8); \
				assert(false && "bail out"); /* Never happens in Q1 */ \
			} \
			*dstpos = buf_ins##n; \
			buf_ins##n += MAX_VSIZE; \
			*max_gid##n = gid; \
			max_gid##n++; \
		} while (0)


	#define insert(n, idx, org) \
		do { \
			const auto gid = gids[org]; \
			auto dstpos = &grppos##n[gid]; \
			if (UNLIKELY(!(*dstpos))) { \
				new_group(n, dstpos); \
			} \
			**dstpos = idx; \
			(*dstpos)++; \
		} while (0)

#if 0
	/* One iterator */
	if (aggr_sel) {
		for (int i=0; i<num; i++) {
			auto k = aggr_sel[i];
			insert(0, k, i);
		}
	} else {
		for (int i=0; i<num; i++) {
			insert(0, i, i);
		}
	}
#else
	/* Two iterators. Improves CPU utilization but risks TLB thrashing */
	Primitives::for_each_2(aggr_sel, num,
		[&] (auto i0, auto i1, auto org0, auto org1) { insert(0, i0, org0); insert(1, i1, org1); },
		[&] (auto i0, auto org0) { insert(0, i0, org0); });
#endif

	int64_t num_groups = 0, num_tuples = 0;

	#define copy(nr, gid) do { \
			const auto pos = grppos##nr[gid]; \
			const auto num = (pos - selbuf##nr) % MAX_VSIZE; /* Optimized into bit-wise operations */ \
			const uint16_t* RESTRICT start = pos - num; \
			for (size_t i=0; i<num; i++) { \
				sel[i] = start[i]; \
			} \
			grppos##nr[gid] = NULL; \
			sel += num; \
			num_tuples += num; \
		} while (0)


	{ /* Build selection vector */
		for (auto curr=buf0; curr < max_gid0; curr++) {
			const auto gid = *curr;

			copy(0, gid);
			if (grppos1[gid]) { copy(1, gid); }

			lim[num_groups] = num_tuples;
			grp[num_groups] = gid;
			num_groups++;
		}

		for (auto curr=buf1; curr < max_gid1; curr++) {
			const auto gid = *curr;
			if (!grppos1[gid]) {
				continue;
			}

			copy(1, gid);
			lim[num_groups] = num_tuples;
			grp[num_groups] = gid;
			num_groups++;
		}
	}

	// assert(i <= 4);
	return num_groups;

	#undef insert
	#undef assert_pos
	#undef copy
	#undef declare
	#undef new_group
}

static int32_t* group_sel_curr[MAX_ACTIVE_GROUPS]; /* selection vector of each group */
static int32_t* group_sel_start[MAX_ACTIVE_GROUPS];
static int32_t group_sel_buf[GROUP_BUF_SIZE];
static int32_t group_gids[16];

int Primitives::partial_shuffle_avx512_cmp(idx_t* RESTRICT group_ids, sel_t* RESTRICT aggr_sel, int num, idx_t* RESTRICT out_sel,
	idx_t* RESTRICT lim, idx_t* RESTRICT grp,  uint16_t** RESTRICT grppos0, uint16_t** RESTRICT grppos1,
	uint16_t* RESTRICT selbuf0, uint16_t* RESTRICT selbuf1)
{
#ifdef __AVX512F__
	// assert(aggr_sel);

#define new_group(gid) do { \
		group_sel_curr[gid] = &group_sel_buf[group_sel_buf_idx]; \
		group_sel_start[gid] = group_sel_curr[gid]; \
		group_sel_buf_idx += MAX_VSIZE; \
		group_gids[group_gids_idx++] = gid; \
		if (group_sel_buf_idx > GROUP_BUF_SIZE) { \
			assert(false && "bail out"); return -1; /* Too much groups */ \
		} \
	} while (false)


	int32_t group_buf[16];
	int i = 0;
	int group_idx;
	int64_t group_sel_buf_idx = 0;
	int64_t group_gids_idx = 0;

	const auto zero = _mm512_setzero_epi32();

	while (i + 16 < num) {
		// cast our selection vector down to have a more efficient SIMD ops
		const auto sel1 = _mm512_cvtepi64_epi32(_mm512_load_epi64(aggr_sel + i));
		const auto sel2 = _mm512_cvtepi64_epi32(_mm512_load_epi64(aggr_sel + i + 8));

		auto sel = _mm512_inserti64x4(_mm512_inserti64x4(zero, sel1, 0), sel2, 1);

		// gather and cast down
		const auto gids = _mm512_i32gather_epi32(sel, (int*)(group_ids), 8);

		auto cflct = _mm512_conflict_epi32(gids);
		auto uniq_mask = _mm512_cmpeq_epi32_mask(cflct, zero);
		_mm512_mask_compressstoreu_epi32(&group_buf[0], uniq_mask, gids);

		auto num_groups = _mm_popcnt_u32(uniq_mask);

		for (group_idx=0; group_idx<num_groups; group_idx++) {
			const auto gid = group_buf[group_idx];

			if (UNLIKELY(!group_sel_start[gid])) { /* This one actually almost never changes */
				new_group(gid);
			}

			auto matches = _mm512_cmpeq_epi32_mask(gids, _mm512_set1_epi32(gid));
			_mm512_mask_compressstoreu_epi32(group_sel_curr[gid], matches, sel);
			/* Advance selection vector of this group */
			group_sel_curr[gid] += _mm_popcnt_u32(matches);
		}

		i += 16;
	}

	// remaining items
	while (i < num) {
		const auto idx = aggr_sel[i];
		const auto gid = group_ids[idx];
		if (!group_sel_start[gid]) {
			new_group(gid);
		}
		*group_sel_curr[gid] = idx;
		group_sel_curr[gid]++;
		i++;
	}

	// go through gids and concat selection vectors

	for (i=0; i<group_gids_idx; i++) {
		auto gid = group_gids[i];

		int32_t* sel = group_sel_start[gid];
		int32_t* end = group_sel_curr[gid];

		while (sel + 8 < end) {
			auto tmp = _mm256_loadu_si256((__m256i*)sel);
			_mm512_storeu_si512((__m512i*)out_sel, _mm512_cvtepi32_epi64(tmp));
			sel += 8;
			out_sel += 8;
		}

		while (sel != end) {
			*out_sel = *sel;
			out_sel++;
			sel++;
		}
	}

	int64_t num_groups = 0;
	int64_t num_tuples = 0;
	for (i=0; i<group_gids_idx; i++) {
		auto gid = group_gids[i];

		int32_t* sel = group_sel_start[gid];
		int32_t* end = group_sel_curr[gid];

		size_t n = end - sel;

		num_tuples += n;
		lim[num_groups] = num_tuples;
		grp[num_groups] = gid;

		num_groups++;

		// reset
		group_sel_curr[gid] = nullptr;
		group_sel_start[gid] = nullptr;
	}

	return num_groups;
#else
	return partial_shuffle_scalar(group_ids, aggr_sel, num, out_sel, lim, grp, grppos0, grppos1, selbuf0, selbuf1);
#endif
}

#undef new_group

/** Quick heler for accessing register internals */
union m512_epi32 {
	__m512i v;
	int32_t a[16];
};

int Primitives::partial_shuffle_avx512(idx_t* RESTRICT groups, sel_t* RESTRICT aggr_sel, int num, idx_t* RESTRICT out_sel,
	idx_t* RESTRICT lim, idx_t* RESTRICT grp, uint16_t** RESTRICT grppos0, uint16_t** RESTRICT grppos1,
	uint16_t* RESTRICT selbuf0, uint16_t* RESTRICT selbuf1)
{
#ifdef __AVX512F__
	assert(aggr_sel);

	constexpr int kCommonOffset = sizeof(int64_t);

	int i = 0;
	const auto zero = _mm512_setzero_epi32();
	const auto min_offsets = _mm512_set1_epi32(kCommonOffset);
	const auto one = _mm512_set1_epi32(1);

	__m512i popcnt_table = popcount_up_to_16_epi32_table;

	/** Large array indexed by GID which provides the current index into 'group_sel' */
	static int32_t group_base[65536]; // TODO: maybe even int16_t as we limit ourselves tyo 64 groups * 1024 items
	/** Buffer for all the selecion lists */
	static int32_t group_sel[kCommonOffset + GROUP_BUF_SIZE];
	/** Pointer into 'group_sel'
	 * Can be used for fast iteration over groups or counting the total amount of tuples in a group */
	static int32_t group_sel_start[MAX_ACTIVE_GROUPS];

	int total_groups = 0;

#define sequential(gid, sel_idx) do { \
		auto idx = group_base[gid]; \
		if (idx < kCommonOffset) { \
			/* Allocate new index in group_sel */ \
			group_base[gid] = kCommonOffset + total_groups * MAX_VSIZE; \
			group_sel_start[total_groups] = group_base[gid]; \
			/* printf("new group gid=%d group_base[gid]=%d idx=%d sel_start[idx]=%d\n", gid, group_base[gid], total_groups, group_sel_start[total_groups]); */ \
			total_groups++; \
			if (UNLIKELY(total_groups > MAX_ACTIVE_GROUPS)) { \
				assert(false && "bail out"); \
			} \
		} \
		group_sel[group_base[gid]] = sel_idx; \
		group_base[gid]++; \
	} while (false)

	while (i + 16 < num) {
		// cast our selection vector down to have a more efficient SIMD ops
		const auto sel1 = _mm512_cvtepi64_epi32(_mm512_load_epi64(aggr_sel + i));
		const auto sel2 = _mm512_cvtepi64_epi32(_mm512_load_epi64(aggr_sel + i + 8));

		auto sel = _mm512_inserti64x4(_mm512_inserti64x4(zero, sel1, 0), sel2, 1);
		
		// gather and cast down
		const auto gids = _mm512_i32gather_epi32(sel, (int*)(groups), 8);
		
		auto cflct = _mm512_conflict_epi32(gids);
		
		// population count gives us the indices inside each group
		auto pop = popcount_up_to_16_bitw_epi32(cflct);
		
		// generate index for each group
		auto idx = _mm512_i32gather_epi32(gids, group_base, 4);
		
		/* We always add kCommonOffset to the offsets, so 0 means uninitialized */
		auto new_groups_mask = _mm512_cmplt_epi32_mask(idx, min_offsets);

		// find new groups <=> idx <= 0
		int num_new_groups = _mm_popcnt_u32(new_groups_mask);

		// get absolute indices
		idx = _mm512_add_epi32(pop, idx);
		// increment by one to prepare for the next round
		auto new_idx = _mm512_add_epi32(idx, one);

		if (num_new_groups == 0) {
			_mm512_i32scatter_epi32(group_sel, idx, sel, 4);

			// increment by scattering back / last write wins (according to intel manual)
			_mm512_i32scatter_epi32(group_base, gids, new_idx, 4);
		} else {
			m512_epi32 gid_vec;
			m512_epi32 sel_vec;

			gid_vec.v = gids;
			sel_vec.v = sel;

			/* Get rid of existing groups by flushing them into buffers, before we start to sequntially
			 * resolve the problematic cases. */
			auto existing_groups_mask = _mm512_knot(new_groups_mask);
			_mm512_mask_i32scatter_epi32(group_sel, existing_groups_mask, idx, sel, 4);

			// increment by scattering back / last write wins (according to intel manual)
			_mm512_mask_i32scatter_epi32(group_base, existing_groups_mask, gids, new_idx, 4);

			/* Two cases: 1. A completely new group or 2. a conflicting new group.
			 * This should be a cold path anyway, so we just resolve this by trying again
			 * in sequential fashion */
			// assert(num_new_groups > 0);
			int group_counter = 0;
			
			for (int j=0; j<16; j++) {
				if ((1 << j) & new_groups_mask) {
					const auto k = sel_vec.a[j];
					const auto gid = gid_vec.a[j];
					sequential(gid, k);
					group_counter++;
				}
			}
			// assert(group_counter == num_new_groups);
		}

		i += 16;
	}

	/* Cold path, sequentially process remainder */
	while (i < num) {
		const auto k = aggr_sel[i];
		const auto gid = groups[k];
		sequential(gid, k);
		i++;
	}

	int num_groups = 0;
	int acc_num = 0;
	/* Build selection vector from lists */
	for (int i=0; i<MAX_ACTIVE_GROUPS; i++) {
		int32_t first = group_sel_start[i];
		if (!first) {
			break;
		}
		
		const int32_t* sel16 = &group_sel[first];
		const auto gid = groups[*sel16];

		// assert(first < group_base[gid]);

		const int32_t* sel16_end = &group_sel[group_base[gid]];
		const size_t sel_num = sel16_end - sel16;

		while (sel16 + 8 < sel16_end) {
			auto tmp = _mm256_loadu_si256((__m256i*)sel16);
			_mm512_storeu_si512((__m512i*)out_sel, _mm512_cvtepi32_epi64(tmp));
			sel16 += 8;
			out_sel += 8;
		}

		
		while (sel16 != sel16_end) {
			*out_sel = *sel16;
			sel16++;
			out_sel++;
		}

		acc_num += sel_num;
		lim[num_groups] = acc_num;
		grp[num_groups] = gid;
		num_groups++;

		/* Clean up */
		group_sel_start[i] = 0;
		group_base[gid] = 0;
	}

	assert(num_groups == total_groups);

	#undef sequential

	return total_groups;
#else
	return partial_shuffle_scalar(groups, aggr_sel, num, out_sel, lim, grp, grppos0, grppos1, selbuf0, selbuf1);
#endif
}

inline static void
cast_int32_t_int64_t(const __m128i& inp, __m128i& r1, __m128i& r2)
{
	r1 = _mm_cvtepi32_epi64(inp);
	r2 = _mm_cvtepi32_epi64(_mm_shuffle_epi32(inp, 14));
}


template<bool GET_LO>
inline static void
cast_int8_t_int64_t(const __m128i& inp, __m128i& r1, __m128i& r2, __m128i& r3, __m128i& r4)
{
	auto split = [] (const auto& a, auto& lo, auto& hi) {
		lo = _mm_cvtepi8_epi64(a);
		hi = _mm_cvtepi8_epi64(_mm_srli_epi32(a, 16));
	};

	if (GET_LO) {
		split(inp, r1, r2);
		split(_mm_shuffle_epi32(inp, 1), r3, r4);
	} else {
		split(_mm_shuffle_epi32(inp, 2), r1, r2);
		split(_mm_shuffle_epi32(inp, 3), r3, r4);
	}
}

uint64_t
Primitives::map_charge(int64_t*RESTRICT res, int32_t*RESTRICT col1, int8_t*RESTRICT col2, sel_t*RESTRICT sel, int n)
{
	if (HAND_OPT_CODE && !sel) {
		n = (n-1) >> 4;

		for(int i=0; i<=n; i++) {
			const auto c1 = (__m128i*)col1 + i*4;
			const auto c2 = (__m128i*)col2 + i;
			auto out = (__m128i*) res + i * 8;

			const __m128i l_16 = _mm_loadu_si128(c2);

			/* Multiplies and stores the result */
			auto mul_store = [&] (const auto& a, const auto& b, const int off) {
				_mm_storeu_si128(out + off, _mm_mul_epi32(a, b));
			};

			auto load_cast_int32_t = [&] (const int off, auto& lo, auto& hi) {
				cast_int32_t_int64_t(_mm_loadu_si128(c1 + off), lo, hi);
			};

			__m128i l1, l2, l3, l4;
			__m128i r1, r2, r3, r4;

			/* Process first 8 multiplcations */
			cast_int8_t_int64_t<true>(l_16, r1, r2, r3, r4);
			load_cast_int32_t(0, l1, l2);
			load_cast_int32_t(1, l3, l4);

			mul_store(l1, r1, 0);
			mul_store(l2, r2, 1);
			mul_store(l3, r3, 2);
			mul_store(l4, r4, 3);

			/* Process next 8 multiplcations */
			cast_int8_t_int64_t<false>(l_16, r1, r2, r3, r4);
			load_cast_int32_t(2, l1, l2);
			load_cast_int32_t(3, l3, l4);

          	mul_store(l1, r1, 4);
			mul_store(l2, r2, 5);
			mul_store(l3, r3, 6);
			mul_store(l4, r4, 7);
		}
		return n;
	}

	return map(res, sel, n, [&] (auto i) {
		return ((int64_t) col1[i])*col2[i];
	});
}

int Primitives::select_int32_t(sel_t* RESTRICT out, sel_t* RESTRICT sel, int n, bool data_dep, int* RESTRICT a, int b) {
	return select(out, sel, n, data_dep, [&] (size_t i) { return a[i] <= b; });
}

int Primitives::select_int16_t(sel_t* RESTRICT out, sel_t* RESTRICT sel, int n, bool data_dep, int16_t* RESTRICT a, int16_t b) {
    return select(out, sel, n, data_dep, [&] (size_t i) { return a[i] <= b; });
}

int Primitives::select_int16_t_avx512(sel_t* RESTRICT out, sel_t* RESTRICT sel, int n, bool data_dep, int16_t* RESTRICT a, int16_t b) {
	return select_int16_t(out, sel, n, data_dep, a, b);
}

int Primitives::map_gid2_dom_restrict(idx_t* RESTRICT out, sel_t* RESTRICT sel, int n, int8_t* RESTRICT a, int8_t min_a, int8_t max_a, int8_t* RESTRICT b, int8_t min_b, int8_t max_b) {
	uint8_t d = max_b - min_b;
	if (min_a) {
		if (min_b) {
			return map(out, sel, n, [&] (size_t i) {
		        return ((uint8_t) (a[i] - (uint8_t) min_a)) * d + ((uint8_t) (b[i] - (uint8_t) min_b));
		    });
		} else {
			return map(out, sel, n, [&] (size_t i) {
		        return ((uint8_t) (a[i] - (uint8_t) min_a)) * d + (uint8_t)b[i];
		    });
		}
	} else {
		if (min_b) {
			return map(out, sel, n, [&] (size_t i) {
				return a[i] * d + (uint8_t) (b[i] - (uint8_t) min_b);
		    });
		} else {
			return map(out, sel, n, [&] (size_t i) {
				return (a[i]) * d + (b[i]);
		    });
		}
	}
}

int Primitives::map_gid(idx_t* RESTRICT out, sel_t* RESTRICT sel, int n, int8_t* RESTRICT a, int8_t* RESTRICT b) {
    return map(out, sel, n, [&] (size_t i) {
        uint16_t idx =  a[i] << 8 | b[i];
        return idx;
    });    
}

int Primitives::map_disc_1(int8_t* RESTRICT out, sel_t* RESTRICT sel, int n, int8_t RESTRICT b, int8_t* RESTRICT a) {
    return map(out, sel, n, [&] (size_t i) {
        return b - (int8_t)a[i];
    });
}

int Primitives::map_tax_1(int8_t* RESTRICT out, sel_t* RESTRICT sel, int n, int8_t* RESTRICT a, int8_t RESTRICT b) {
    return map(out, sel, n, [&] (size_t i) {
        return b + a[i];
    });
}


int Primitives::map_disc_price(int32_t* RESTRICT out, sel_t* RESTRICT sel, int n, int8_t* RESTRICT a, int32_t* RESTRICT b) {
    return map(out, sel, n, [&] (size_t i) {
        int32_t x = a[i];
        int32_t y = b[i];
        int32_t z = x * y;
        return z;
    });        
}

int Primitives::ordaggr_quantity(AggrHashTable* RESTRICT aggr0, idx_t* RESTRICT pos, idx_t* RESTRICT lim, idx_t* RESTRICT grp, idx_t num_groups, int16_t* RESTRICT quantity) {
    int64_t ag_sum_quantity;
    return Primitives::ordaggr(pos, lim, grp, num_groups,
        [&] (auto g) { /* Init aggregate */
            ag_sum_quantity = 0;
        },
        [&] (auto g, auto i) { /* Update aggregate */
            ag_sum_quantity += quantity[i]; 
        },
        [&] (auto group_idx) { /* Finalize aggregate*/
            auto g = grp[group_idx];
            aggr0[g].sum_quantity += ag_sum_quantity;
        });
};

int Primitives::ordaggr_extended_price(AggrHashTable* RESTRICT aggr0, idx_t* RESTRICT pos, idx_t* RESTRICT lim, idx_t* RESTRICT grp, idx_t num_groups, int32_t* RESTRICT price)
{
	int64_t ag_sum_base_price;

	return Primitives::ordaggr(pos, lim, grp, num_groups,
		[&] (auto g) { /* Init aggregate */
			ag_sum_base_price = 0;
		},
		[&] (auto g, auto i) { /* Update aggregate */
			ag_sum_base_price += price[i];
		},
		[&] (auto group_idx) { /* Finalize aggregate*/
			auto g = grp[group_idx];
			aggr0[g].sum_base_price += ag_sum_base_price;
		});	
}

int Primitives::ordaggr_disc_price(AggrHashTable* RESTRICT aggr0, idx_t* RESTRICT pos, idx_t* RESTRICT lim, idx_t* RESTRICT grp, idx_t num_groups, int32_t* RESTRICT v_disc_price)
{
	int64_t ag_sum_disc_price;

	return Primitives::ordaggr(pos, lim, grp, num_groups,
		[&] (auto g) { /* Init aggregate */
			ag_sum_disc_price = 0;
		},
		[&] (auto g, auto i) { /* Update aggregate */
			ag_sum_disc_price += v_disc_price[i];
		},
		[&] (auto group_idx) { /* Finalize aggregate*/
			auto g = grp[group_idx];
			aggr0[g].sum_disc_price = int128_add64(aggr0[g].sum_disc_price, ag_sum_disc_price);
		});	
}

int Primitives::ordaggr_charge(AggrHashTable* RESTRICT aggr0, idx_t* RESTRICT pos, idx_t* RESTRICT lim, idx_t* RESTRICT grp, idx_t num_groups, int64_t* RESTRICT v_charge)
{
	int64_t ag_sum_charge;

	return Primitives::ordaggr(pos, lim, grp, num_groups,
		[&] (auto g) { /* Init aggregate */
			ag_sum_charge = 0;
		},
		[&] (auto g, auto i) { /* Update aggregate */
			ag_sum_charge += v_charge[i];
		},
		[&] (auto group_idx) { /* Finalize aggregate*/
			auto g = grp[group_idx];
			aggr0[g].sum_charge = int128_add64(aggr0[g].sum_charge, ag_sum_charge);
		});	
}

int Primitives::ordaggr_disc(AggrHashTable* RESTRICT aggr0, idx_t* RESTRICT pos, idx_t* RESTRICT lim, idx_t* RESTRICT grp, idx_t num_groups, int8_t* RESTRICT v_disc_1)
{
	int64_t ag_sum_disc;

	return Primitives::ordaggr(pos, lim, grp, num_groups,
		[&] (auto g) { /* Init aggregate */
			ag_sum_disc = 0;
		},
		[&] (auto g, auto i) { /* Update aggregate */
			ag_sum_disc += v_disc_1[i];
		},
		[&] (auto group_idx) { /* Finalize aggregate*/
			auto g = grp[group_idx];
			aggr0[g].sum_disc += ag_sum_disc;
		});	
}

int Primitives::ordaggr_count(AggrHashTable* RESTRICT aggr0, idx_t* RESTRICT pos, idx_t* RESTRICT lim, idx_t* RESTRICT grp, idx_t num_groups)
{
	size_t i = 0;
	for (int g=0; g<num_groups; g++) {
		aggr0[grp[g]].count += lim[g] - i;
		i = lim[g];
	}
}


template<typename T, typename AccT, int DOP, bool FixAlign, typename VecInit,
	typename SelLoad, typename ValGather, typename ValLoad,
	typename VecAdd, typename VecReduce, typename FinalizeFun>
int ordaggr_sum(idx_t* RESTRICT pos, idx_t* RESTRICT lim, idx_t* RESTRICT grp, idx_t num_groups, T* RESTRICT v_charge,
	VecInit&& init_vec, SelLoad&& sel_load, ValGather&& val_gather, ValLoad&& val_load,
	VecAdd&& add, VecReduce&& reduce, FinalizeFun&& finalize)
{
	AccT ag_sum_charge;
	int64_t k = 0;

	const size_t ALIGNMENT = 64;

	if (pos) {
		for (idx_t g=0; g<num_groups; g++) {
			ag_sum_charge = 0;
			// prologue thanks to alignment when we switch to next group
			if (FixAlign && (size_t)(pos + k) % ALIGNMENT) {
				while ((size_t)(pos + k) % ALIGNMENT && k < lim[g]) {
		            ag_sum_charge += v_charge[pos[k]];
		            k++;
		        }
			}

			if (k + DOP < lim[g]) {
				auto sum = init_vec();
				do {
		        	auto v_sel = sel_load(pos + k);
		        	sum = add(sum, val_gather(v_sel, v_charge));
		        	k += DOP;
		        } while (k + DOP < lim[g]);
		        ag_sum_charge += reduce(sum);
			}

			while (k < lim[g]) {
	            ag_sum_charge += v_charge[pos[k]];
	            k++;
	        }

	        finalize(g, ag_sum_charge);
	    }
	} else {
		for (idx_t g=0; g<num_groups; g++) {
			ag_sum_charge = 0;
			if (FixAlign && (size_t)(v_charge + k) % ALIGNMENT) {
				while ((size_t)(v_charge + k) % ALIGNMENT && k < lim[g]) {
		            ag_sum_charge += v_charge[k];
		            k++;
		        }
			}

			if (k + DOP < lim[g]) {
				auto sum = init_vec();
				do {
		        	sum = add(sum, val_load(v_charge + k));
		        	k += DOP;
		        } while (k + DOP < lim[g]);
		        ag_sum_charge += reduce(sum);
			}

			while (k < lim[g]) {
	            ag_sum_charge += v_charge[k];
	            k++;
	        }

	        finalize(g, ag_sum_charge);
	    }
	}

	return 0;
}


template<typename T, typename ValGather, typename ValLoad, typename FinalizeFun>
int ordaggr_sum_32(idx_t* RESTRICT pos, idx_t* RESTRICT lim, idx_t* RESTRICT grp, idx_t num_groups, T* RESTRICT data, ValGather&& val_gather, ValLoad&& val_load, FinalizeFun&& finalize)
{
	return ordaggr_sum<T, int64_t, 16, true>(pos, lim, grp, num_groups, data,
		[] () { return _mm512_set1_epi32(0); },
		[] (auto sel) {
			/* load part of selection vector and convert to 32 bit indices in order to fit more */
			auto lo = _mm512_cvtepi64_epi32(_mm512_load_epi64(sel));
			auto hi = _mm512_cvtepi64_epi32(_mm512_load_epi64(sel + 8));

			auto r = _mm512_set1_epi32(0);

			r = _mm512_inserti64x4(r, lo, 0);
			r = _mm512_inserti64x4(r, hi, 1);
			return r;
		},
		[&] (auto sel, auto val) { return val_gather(sel, val); },
		[&] (auto val) { return val_load(val); },
		[] (auto old, auto next) { return _mm512_add_epi32(old, next); },
		[] (auto acc) { return m512_hsum_epi32(acc); },
		[&] (auto a, auto b) { finalize(a, b); }
	); 
}

template<typename T, typename ValGather, typename ValLoad, typename FinalizeFun>
int ordaggr_sum_64(idx_t* RESTRICT pos, idx_t* RESTRICT lim, idx_t* RESTRICT grp, idx_t num_groups, T* RESTRICT data, ValGather&& val_gather, ValLoad&& val_load, FinalizeFun&& finalize)
{
	return ordaggr_sum<T, int64_t, 8, true>(pos, lim, grp, num_groups, data,
		[] () { return _mm512_set1_epi64(0); },
		[] (auto sel) { return _mm512_load_epi64(sel); },
		[&] (auto sel, auto val) { return val_gather(sel, val); },
		[&] (auto val) { return val_load(val); },
		[] (auto old, auto next) { return _mm512_add_epi64(old, next); },
		[] (auto acc) { return m512_hsum_epi64(acc); },
		[&] (auto a, auto b) { finalize(a, b); }
	); 
}

template<typename FinalizeFun>
int ordaggr_int64_t_sum(idx_t* RESTRICT pos, idx_t* RESTRICT lim, idx_t* RESTRICT grp, idx_t num_groups, int64_t* RESTRICT data, FinalizeFun&& finalize)
{
	return ordaggr_sum_64<int64_t>(pos, lim, grp, num_groups, data,
		[] (auto sel, auto val) { return  _mm512_i64gather_epi64(sel, (const long long int*)val, 8); },
		[] (auto val) { return _mm512_loadu_si512(val); },
		[&] (auto a, auto b) { finalize(a, b); });
}

template<typename FinalizeFun>
int ordaggr_int32_t_sum(idx_t* RESTRICT pos, idx_t* RESTRICT lim, idx_t* RESTRICT grp, idx_t num_groups, int32_t* RESTRICT data, FinalizeFun&& finalize)
{
	return ordaggr_sum_32<int32_t>(pos, lim, grp, num_groups, data,
		[] (auto sel, auto val) { return  _mm512_i32gather_epi32(sel, (const int*)val, 4); },
		[] (auto val) { return _mm512_load_epi32(val); },
		[&] (auto a, auto b) { finalize(a, b); }
	);
}

template<typename FinalizeFun>
int ordaggr_int32_t_sum64(idx_t* RESTRICT pos, idx_t* RESTRICT lim, idx_t* RESTRICT grp, idx_t num_groups, int32_t* RESTRICT data, FinalizeFun&& finalize)
{
	return ordaggr_sum_64<int32_t>(pos, lim, grp, num_groups, data,
		[] (auto sel, auto val) { return  _mm512_and_epi64(_mm512_set1_epi64(0xFFFFFFFF), _mm512_i64gather_epi64(sel, (const long long int*)val, 4)); },
		[] (auto val) { return _mm512_cvtepi32_epi64(_mm256_load_si256((__m256i*)val)); },
		[&] (auto a, auto b) { finalize(a, b); }
	);
}

template<typename FinalizeFun>
int ordaggr_int16_t_sum(idx_t* RESTRICT pos, idx_t* RESTRICT lim, idx_t* RESTRICT grp, idx_t num_groups, int16_t* RESTRICT data, FinalizeFun&& finalize)
{
	return ordaggr_sum_32<int16_t>(pos, lim, grp, num_groups, data,
		[] (auto sel, auto val) { return  _mm512_and_epi32(_mm512_set1_epi32(0xFFFF), _mm512_i32gather_epi32(sel, (int*)val, 2)); }, // gather 2-bytes and mask the use-less bits
		[] (auto val) { return _mm512_cvtepi16_epi32(_mm256_load_si256((__m256i*)val)); }, // load subsequent 2-bytes values and cast them up to 32 bit
		[&] (auto a, auto b) { finalize(a, b); }
	);
}

template<typename FinalizeFun>
int ordaggr_int8_t_sum(idx_t* RESTRICT pos, idx_t* RESTRICT lim, idx_t* RESTRICT grp, idx_t num_groups, int8_t* RESTRICT data, FinalizeFun&& finalize)
{
	return ordaggr_sum_32<int8_t>(pos, lim, grp, num_groups, data,
		[] (auto sel, auto val) { return  _mm512_and_epi32(_mm512_set1_epi32(0xFF), _mm512_i32gather_epi32(sel, (int*)val, 1)); }, // gather 1-bytes and mask the use-less bits
		[] (auto val) { return _mm512_cvtepi8_epi32(_mm_load_si128((__m128i*)val)); }, // load subsequent 1-bytes values and cast them up to 32 bit
		[&] (auto a, auto b) { finalize(a, b); }
	);
}


int Primitives::par_ordaggr_quantity(AggrHashTable* RESTRICT aggr0, idx_t* RESTRICT pos, idx_t* RESTRICT lim, idx_t* RESTRICT grp, idx_t num_groups, int16_t* RESTRICT quantity) {
#ifdef __AVX512F__
	return ordaggr_int16_t_sum(pos, lim, grp, num_groups, quantity,
		[&] (auto group_idx, int64_t val) {
			auto g = grp[group_idx];
			aggrs0[g].sum_quantity += val;
		}
	);
#else
	return ordaggr_quantity(aggr0, pos, lim, grp, num_groups, quantity);
#endif
};

int Primitives::par_ordaggr_extended_price(AggrHashTable* RESTRICT aggr0, idx_t* RESTRICT pos, idx_t* RESTRICT lim, idx_t* RESTRICT grp, idx_t num_groups, int32_t* RESTRICT price)
{
#ifdef __AVX512F__
	return ordaggr_int32_t_sum(pos, lim, grp, num_groups, price,
		[&] (auto group_idx, int64_t val) {
			auto g = grp[group_idx];
			aggrs0[g].sum_base_price += val;
		}
	);
#else
	return ordaggr_extended_price(aggr0, pos, lim, grp, num_groups, price);
#endif
}

int Primitives::par_ordaggr_disc_price(AggrHashTable* RESTRICT aggr0, idx_t* RESTRICT pos, idx_t* RESTRICT lim, idx_t* RESTRICT grp, idx_t num_groups, int32_t* RESTRICT v_disc_price)
{
#ifdef __AVX512F__
	return ordaggr_int32_t_sum64(pos, lim, grp, num_groups, v_disc_price,
		[&] (auto group_idx, int64_t val) {
			auto g = grp[group_idx];
			aggrs0[g].sum_disc_price = int128_add64(aggrs0[g].sum_disc_price, val);
		}
	);
#else
	return ordaggr_disc_price(aggr0, pos, lim, grp, num_groups, v_disc_price);
#endif
}

int Primitives::par_ordaggr_charge(AggrHashTable* RESTRICT aggr0, idx_t* RESTRICT pos, idx_t* RESTRICT lim, idx_t* RESTRICT grp, idx_t num_groups, int64_t* RESTRICT v_charge)
{
#ifdef __AVX512F__
	return ordaggr_int64_t_sum(pos, lim, grp, num_groups, v_charge,
		[&] (auto group_idx, int64_t val) {
			auto g = grp[group_idx];
			aggrs0[g].sum_charge = int128_add64(aggrs0[g].sum_charge, val);
		}
	);
#else
	return ordaggr_charge(aggr0, pos, lim, grp, num_groups, v_charge);
#endif
}

int Primitives::par_ordaggr_disc(AggrHashTable* RESTRICT aggr0, idx_t* RESTRICT pos, idx_t* RESTRICT lim, idx_t* RESTRICT grp, idx_t num_groups, int8_t* RESTRICT v_disc_1)
{
#ifdef __AVX512F__
	return ordaggr_int8_t_sum(pos, lim, grp, num_groups, v_disc_1,
		[&] (auto group_idx, int64_t val) {
			auto g = grp[group_idx];
			aggrs0[g].sum_disc += val;
		}
	);
#else
	return ordaggr_disc(aggr0, pos, lim, grp, num_groups, v_disc_1);
#endif
}

int Primitives::ordaggr_all_in_one(AggrHashTable* RESTRICT aggr0, idx_t* RESTRICT pos, idx_t* RESTRICT lim, idx_t* RESTRICT grp, idx_t num_groups, int16_t* RESTRICT quantity,
	int32_t* RESTRICT extendedprice, int32_t* RESTRICT disc_price, int64_t* RESTRICT charge, int8_t* RESTRICT disc_1)
{
	int64_t ag_sum_quantity;
	int64_t ag_sum_base_price;
	int64_t ag_sum_disc_price;
	int64_t ag_sum_charge;
	int64_t ag_sum_disc;

	Primitives::ordaggr_fst(pos, lim, grp, num_groups,
		[&] (auto g) { /* Init aggregate */
			ag_sum_quantity = 0;
			ag_sum_base_price = 0;
			ag_sum_disc_price = 0;
			ag_sum_charge = 0;
			ag_sum_disc = 0;
		},
		[&] (auto g, auto i) { /* Update aggregate */
			ag_sum_quantity += quantity[i];	
			ag_sum_base_price += extendedprice[i];
			ag_sum_disc_price += disc_price[i];
			ag_sum_charge += charge[i];
			ag_sum_disc += disc_1[i];
		},
		[&] (auto group_idx, auto first) { /* Finalize aggregate*/
			auto g = grp[group_idx]; /* Figure out the real group */

			aggr0[g].sum_quantity += ag_sum_quantity;
			aggr0[g].sum_base_price += ag_sum_base_price;
			aggr0[g].sum_disc_price = int128_add64(aggr0[g].sum_disc_price, ag_sum_disc_price);
			aggr0[g].sum_charge = int128_add64(aggr0[g].sum_charge, ag_sum_charge);
			aggr0[g].sum_disc += ag_sum_disc;
			aggr0[g].count += lim[group_idx] - first;
		});	
}

int Primitives::old_ordaggr_all_in_one(AggrHashTable* RESTRICT aggr0, idx_t* RESTRICT pos, idx_t* RESTRICT lim, idx_t* RESTRICT grp, idx_t num_groups, int64_t* RESTRICT quantity,
	int64_t* RESTRICT extendedprice, int64_t* RESTRICT disc_price, int64_t* RESTRICT charge, int64_t* RESTRICT disc_1)
{
	int64_t ag_sum_quantity;
	int64_t ag_sum_base_price;
	int128_t ag_sum_disc_price; /* TODO: actually 128 bits long, but we can make it shorter due to partial aggr, see ordaggr_all_in_one() */
	int128_t ag_sum_charge; /* TODO: actually 128 bits long, but we can make it shorter due to partial aggr, see ordaggr_all_in_one() */
	int64_t ag_sum_disc;

	Primitives::ordaggr_fst(pos, lim, grp, num_groups,
		[&] (auto g) { /* Init aggregate */
			ag_sum_quantity = 0;
			ag_sum_base_price = 0;
			ag_sum_disc_price = 0;
			ag_sum_charge = 0;
			ag_sum_disc = 0;
		},
		[&] (auto g, auto i) { /* Update aggregate */
			ag_sum_quantity += quantity[i];	
			ag_sum_base_price += extendedprice[i];
			ag_sum_disc_price += disc_price[i];
			ag_sum_charge += charge[i];
			ag_sum_disc += disc_1[i];
		},
		[&] (auto group_idx, auto first) { /* Finalize aggregate*/
			auto g = grp[group_idx]; /* Figure out the real group */

			aggr0[g].sum_quantity += ag_sum_quantity;
			aggr0[g].sum_base_price += ag_sum_base_price;
			aggr0[g].sum_disc_price = int128_add64(aggr0[g].sum_disc_price, ag_sum_disc_price);
			aggr0[g].sum_charge = int128_add64(aggr0[g].sum_charge, ag_sum_charge);
			aggr0[g].sum_disc += ag_sum_disc;
			aggr0[g].count += lim[group_idx] - first;
		});	
}

int Primitives::old_map_gid(idx_t* RESTRICT out, sel_t* RESTRICT sel, int n, char* RESTRICT v_returnflag, char* RESTRICT v_linestatus)
{
	return Primitives::map(out, sel, n, [&] (size_t i) {
			return v_returnflag[i] << 8 | v_linestatus[i];
		});	
}

int Primitives::old_map_disc_1(int64_t* RESTRICT v_disc_1, sel_t* RESTRICT sel, int n, int64_t dec_one, int64_t* RESTRICT v_discount)
{
	return Primitives::map(v_disc_1, sel, n, [&] (size_t i) {
			return dec_one - v_discount[i];
		});
}

int Primitives::old_map_tax_1(int64_t* RESTRICT v_disc_1, sel_t* RESTRICT sel, int n, int64_t dec_one, int64_t* RESTRICT v_tax)
{
	return Primitives::map(v_disc_1, sel, n, [&] (size_t i) {
			return v_tax[i] + dec_one;
		});
}

int Primitives::old_map_mul(int64_t* RESTRICT v_disc_price, sel_t* RESTRICT sel, int n, int64_t* RESTRICT v_disc_1, int64_t* RESTRICT v_extendedprice)
{
	return Primitives::map(v_disc_price, sel, n, [&] (size_t i) {
			return Decimal64::Mul(v_disc_1[i], v_extendedprice[i]);
		});
}