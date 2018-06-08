#ifndef H_KERNEL_AVX512
#define H_KERNEL_AVX512

#include "common.hpp"

#ifdef __AVX512F__
#include <x86intrin.h>
#include <immintrin.h>
#endif

template<bool mask_aggr_memops = false, bool mask_scan = false, bool int128_aggr = false>
struct AVX512 : public BaseKernel {
	using BaseKernel::BaseKernel;	

	__attribute__((noinline)) void operator()()
	{
#ifdef __AVX512F__
		kernel_prologue();

		const size_t num_tuples = 16;
		const __m512i v_cmp = _mm512_set1_epi32(cmp.dte_val);
		const __m512i v_dec1 = _mm512_set1_epi64(Decimal64::ToValue(1, 0));
		const __m512i v_card = _mm512_set1_epi64(cardinality);
		const auto zero = _mm512_setzero_epi32();
		const auto seq0_epi32 = _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);

		// __m512i v_offsets = _mm512_set_epi64(0, 1, 2, 3, 4, 5, 6, 7);
		__m512i v_offsets = _mm512_set_epi64(7, 6, 5, 4, 3, 2, 1, 0);;

		size_t offset = 0;
		while (true) {
			if (offset >= cardinality) {
				break;
			} 

			// load 16 values
			__m512i v_shipdate = _mm512_load_epi32(shipdate + offset);

			/* Select */
			__mmask8 mask_lo, mask_hi;
			__mmask16 mask;

			{
				mask = _mm512_cmple_epi32_mask(v_shipdate, v_cmp);
				int ilo = mask;
				int ihi = mask >> 8;

				mask_lo = _mm512_kand(_mm512_cmplt_epi64_mask(v_offsets, v_card), ilo);
				mask_hi = _mm512_kand(_mm512_cmplt_epi64_mask(_mm512_add_epi64(v_offsets, _mm512_set1_epi64(num_tuples / 2)), v_card), ihi);
				mask = _mm512_kmov(_mm512_kor(mask_lo, (mask_hi << 8)));
			}

			/* Other columns */

			// load 16 values
			__m128i v_linestatus = _mm_load_si128((__m128i*)(linestatus + offset));
			__m128i v_returnflag = _mm_load_si128((__m128i*)(returnflag + offset));

			auto load_epi64 = [zero] (const auto& msk, const auto& col, size_t off) {
				return mask_scan ? _mm512_mask_load_epi64(zero, msk, col + off) : _mm512_load_epi64(col + off);
			};

			// load 8 values
			__m512i v_disc = load_epi64(mask_lo, discount, offset);
			__m512i v_tax = load_epi64(mask_lo, tax, offset);
			__m512i v_price = load_epi64(mask_lo, extendedprice, offset);
			__m512i v_quant = load_epi64(mask_lo, quantity, offset);

			__m512i v_disc_1 = _mm512_sub_epi64(v_dec1, v_disc);
			__m512i v_tax_1 = _mm512_add_epi64(v_dec1, v_tax);

			__m512i v_disc_price = mul_64_64_64_avx512(v_disc_1, v_price);
			__m512i v_charge = mul_64_64_64_avx512(v_disc_price, v_tax_1);

			// and another batch of 8
			auto hi = num_tuples / 2;
			__m512i v2_disc = load_epi64(mask_hi, discount, offset+hi);
			__m512i v2_tax = load_epi64(mask_hi, tax, offset+hi);
			__m512i v2_price = load_epi64(mask_hi, extendedprice, offset+hi);
			__m512i v2_quant = load_epi64(mask_hi, quantity, offset+hi);

			__m512i v2_disc_1 = _mm512_sub_epi64(v_dec1, v2_disc);
			__m512i v2_tax_1 = _mm512_add_epi64(v_dec1, v2_tax);

			__m512i v2_disc_price = mul_64_64_64_avx512(v2_disc_1, v2_price);
			__m512i v2_charge = mul_64_64_64_avx512(v2_disc_price, v2_tax_1);


			/* Aggr */

			// cast char columns to 16 bits
			__m256i v_linestatus16 = _mm256_cvtepi8_epi16(v_linestatus);
			__m256i v_returnflag16 = _mm256_cvtepi8_epi16(v_returnflag);

			__m256i v_gids16 = _mm256_or_si256(_mm256_slli_epi16(v_returnflag16, 8), v_linestatus16);

			// split into lower and higher half
			
			// TODO: improve with AVX 512 VL + F
			__m128i v_gids_lo = _mm256_extractf128_si256(v_gids16, 0);
			__m128i v_gids_hi = _mm256_extractf128_si256(v_gids16, 1);

			// cast to 32 bit, because we need 32-bit later and it still fits
			__m256i v_gids32_lo = _mm256_cvtepi16_epi32(v_gids_lo);
			__m256i v_gids32_hi = _mm256_cvtepi16_epi32(v_gids_hi);

			auto make_index = [] (const __m256i& idx) {
				// (multiply by DOP i.e. 8) and add non-conflicting offsets
				return _mm256_add_epi32(
					// _mm256_set_epi32(0, 1, 2, 3, 4, 5, 6, 7),
					_mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0),
					_mm256_slli_epi32(idx, 3));
			};

			__m256i v_idx_lo = make_index(v_gids32_lo);
			__m256i v_idx_hi = make_index(v_gids32_hi);

			auto gather512_64 = [&zero] (const auto& mask, const auto& idx, const auto& t) {
				return mask_aggr_memops ? _mm512_mask_i32gather_epi64(zero, mask, idx, t, 8) : _mm512_i32gather_epi64(idx, t, 8);					
			};

			auto scatter512_64 = [&zero] (const auto& mask, const auto& idx, auto& t, const auto& new_ag) {
				if (mask_aggr_memops) {
					_mm512_mask_i32scatter_epi64(t, mask, idx, new_ag, 8);
				} else {
					_mm512_i32scatter_epi64(t, idx, new_ag, 8);	
				}
			};

			// update aggregates
			auto apply_aggr64 = [gather512_64, scatter512_64] (const __m256i& idx, int64_t* table, const __mmask8& mask, auto&& update) {
				auto t = (long long int*)(table);

				auto old_ag = gather512_64(mask, idx, t);
				scatter512_64(mask, idx, t, update(old_ag));
			};

			auto sum_aggr64 = [apply_aggr64] (const __m256i& idx, int64_t* table, const __mmask8& mask, const __m512i& data) {
				apply_aggr64(idx, table, mask, [data, mask] (auto& old) {
					return _mm512_mask_add_epi64(old, mask, old, data);	
				});
			};

			auto cnt_aggr64 = [apply_aggr64] (const __m256i& idx, int64_t* table, const __mmask8& mask) {
				apply_aggr64(idx, table, mask, [mask] (auto& old) -> __m512i {
					return _mm512_mask_add_epi64(old, mask, old, _mm512_maskz_set1_epi64(mask, 1));	
				});
			};

			auto sum_aggr128_64 = [gather512_64, scatter512_64] (const __m256i& idx, int64_t* table_lo, int64_t* table_hi, const __mmask8& mask, const auto& data) {
				auto t_lo = (long long int*)(table_lo);
				auto lo = gather512_64(mask, idx, t_lo);

				if (int128_aggr) {
					auto t_hi = (long long int*)(table_hi);
					auto hi = gather512_64(mask, idx, t_hi);
					
					add64_to_int128(hi, lo, mask, data);

					scatter512_64(mask, idx, t_lo, lo);
					scatter512_64(mask, idx, t_hi, hi);
				} else { /* 64 bit aggregate */
					lo = _mm512_mask_add_epi64(lo, mask, lo, data);

					scatter512_64(mask, idx, t_lo, lo);
				}
			};


			auto aggregate = [cnt_aggr64, sum_aggr64, sum_aggr128_64] (const auto& idx, const auto& msk, const auto& quant, const auto& price, const auto& disc,
					const auto& disc_price, const auto& charge) {
				cnt_aggr64(idx, (int64_t*)&aggr_avx0_count, msk);
				sum_aggr64(idx, (int64_t*)&aggr_avx0_sum_quantity, msk, quant);
				sum_aggr64(idx, (int64_t*)&aggr_avx0_sum_base_price, msk, price);
				sum_aggr64(idx, (int64_t*)&aggr_avx0_sum_disc, msk, disc);

				sum_aggr128_64(idx, (int64_t*)&aggr_avx0_sum_disc_price_lo, (int64_t*)&aggr_avx0_sum_disc_price_hi, msk, disc_price);
				sum_aggr128_64(idx, (int64_t*)&aggr_avx0_sum_charge_lo, (int64_t*)&aggr_avx0_sum_charge_hi, msk, charge);
			};
#ifdef PROFILE
			auto prof_ag_start = rdtsc();
#endif
			aggregate(v_idx_lo, mask_lo, v_quant, v_price, v_disc, v_disc_price, v_charge);
			aggregate(v_idx_hi, mask_hi, v2_quant, v2_price, v2_disc, v2_disc_price, v2_charge);

#ifdef PROFILE
			sum_aggr_time += rdtsc() - prof_ag_start;
#endif
			offset += num_tuples;
			v_offsets = _mm512_add_epi64(v_offsets, _mm512_set1_epi64(num_tuples));
		};
#endif
	}
};

#endif