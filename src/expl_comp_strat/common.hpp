#ifndef H_COMMON
#define H_COMMON

#include <cstdint>

#define PROFILE
#define PRINT_RESULTS
//#define PRINT_MINMAX

#include <sstream>
#include <vector>
#include "tpch_kit.hpp"
#include <limits>

#include <x86intrin.h>

#ifdef DEBUG
#define DBG_ASSERT(x) assert(x)
#else
#define DBG_ASSERT(x)
#endif


// #define RESTRICT
#define RESTRICT __restrict__ 
typedef int64_t sel_t;

typedef __int128 int128_t;

typedef sel_t idx_t;

#define ALIGN  __attribute__ ((aligned (64)))

#define NOINL __attribute__ ((noinline))
#define LIKELY(x)       __builtin_expect((!!(x)),1)
#define UNLIKELY(x)     __builtin_expect((!!(x)),0)


#define MAX_ACTIVE_GROUPS 64
static constexpr size_t MAX_GROUPS = std::numeric_limits<unsigned short>::max();
static constexpr size_t AVX_GROUPS = MAX_GROUPS * 8;

static constexpr size_t MAX_VSIZE = 1024;
static constexpr size_t GROUP_BUF_SIZE = MAX_ACTIVE_GROUPS * MAX_VSIZE;


static constexpr size_t kGroupDomain = std::numeric_limits<uint16_t>::max();
static constexpr size_t kGrpPosSize = kGroupDomain * 2;
static constexpr size_t kSelBufSize = 2 * GROUP_BUF_SIZE;


#ifdef __AVX512F__
#include <x86intrin.h>
#include <immintrin.h>

void print64_512 (const char* prefix, __m512i a);
void print32_512(const char* prefix, __m512i a);

inline static int64_t
mul_64_64_64_scalar(int64_t a, int64_t b)
{
    uint64_t a_lo = a & 0xFFFFFFFF;
    uint64_t a_hi = (a >> 32);
    uint64_t b_lo = b & 0xFFFFFFFF;
    uint64_t b_hi = (b >> 32);
    
    uint64_t x = a_lo * b_lo;
    uint64_t y = (a_lo * b_hi) << 32;
    uint64_t z = (a_hi * b_lo) << 32;

    return x+y+z;
}

inline static  __m512i
mul_64_64_64_avx512(__m512i a, __m512i b)
{
#ifdef __AVX512DQ__
	return _mm512_mullo_epi64(a, b);
#else
    // 64-bit input -> get low 32-bits and multiply into 64-bit result
    // i.e. a_lo * b_lo
    auto x = _mm512_mul_epu32(a, b);
    
    // shift hi parts into low parts
    auto a_hi = _mm512_srai_epi64(a, 32);
    auto b_hi = _mm512_srai_epi64(b, 32);

    auto y = _mm512_slli_epi64(_mm512_mul_epu32(a, b_hi), 32); // a_lo * b_hi
    auto z = _mm512_slli_epi64(_mm512_mul_epu32(a_hi, b), 32); // a_hi * b_lo
    
    return _mm512_add_epi64(_mm512_add_epi64(x, y), z);
#endif
}


inline static void
add64_to_int128(__m512i& hi, __m512i& lo, __mmask8 mask, __m512i data)
{
	/* resembles: lo += b; hi += lo < b */

	const __m512i old_lo = lo;
	const __m512i old_hi = hi;

	auto new_lo = _mm512_mask_add_epi64(old_lo, mask, old_lo, data);

	mask &= _mm512_cmplt_epi64_mask(new_lo, data);
	auto new_hi = _mm512_mask_add_epi64(old_hi, mask, old_hi, _mm512_set1_epi64(1));
	// auto new_hi = old_hi;

	hi = new_hi;
	lo = new_lo;
}
#endif

#define int128_add64(a, b) (a) + (b)


/* Reinvent std::min, because the standard function requires its arguments to be references
 * which will result into linking errors when using constants or constexprs */
template<typename T>
inline constexpr T min(T a, T b)
{
	return a < b ? a : b;
}

static constexpr size_t GetBits(size_t f) {
	size_t i=0;
	for (; (((size_t) 1) << i) < f; i++);
	return i;
}

inline static uint64_t
get_cycles()
{
    unsigned int lo,hi;
    __asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
    return ((uint64_t)hi << 32) | lo;
}

inline static uint64_t
rdtsc()
{
#ifdef PROFILE
    return get_cycles();
#else
    return 0;
#endif
}

using Decimal32 = Decimal<9, 2>;
using Decimal64 = Decimal<15, 2>;
using Decimal128 = Decimal<15, 2, int128_t>;


struct AggrHashTable {
	int64_t sum_quantity;
	int64_t count;
	int64_t sum_base_price;
	int64_t sum_disc;
	int128_t sum_disc_price;
	int128_t sum_charge;
};


struct GPUAggrHashTable {
	uint64_t sum_quantity;
	uint64_t count;
	uint64_t sum_base_price;
	uint64_t sum_disc;
	uint64_t sum_disc_price;
	uint64_t sum_charge;
	uint64_t sum_disc_price_hi;
	uint64_t sum_charge_hi;
};

struct GPUAggrHashTableLocal {
	uint64_t sum_quantity;
	uint64_t count;
	uint64_t sum_base_price;
	uint64_t sum_disc;
	uint64_t sum_disc_price;
	uint64_t sum_charge;
};

#define kernel_compact_declare \
    int16_t* RESTRICT l_shipdate; \
	int8_t* RESTRICT l_returnflag; \
	int8_t* RESTRICT l_linestatus; \
	int8_t* RESTRICT l_discount; \
	int8_t* RESTRICT l_tax; \
	int32_t* RESTRICT l_extendedprice; \
	int16_t* RESTRICT l_quantity;

#define kernel_compact_init_magic(col) l_##col[i] = li.l_##col.get()[i];

#define kernel_compact_init() do { \
		size_t cardinality = li.l_extendedprice.cardinality; \
	    l_shipdate = new_array<int16_t>(cardinality);  \
	    l_returnflag = new_array<int8_t>(cardinality);  \
	    l_linestatus = new_array<int8_t>(cardinality); \
	    l_discount = new_array<int8_t>(cardinality); \
	    l_tax = new_array<int8_t>(cardinality); \
	    l_extendedprice = new_array<int32_t>(cardinality); \
	    l_quantity = new_array<int16_t>(cardinality); \
	    for (size_t i=0; i<cardinality; i++) { \
	        kernel_compact_init_magic(shipdate); \
	        l_returnflag[i] = li.l_returnflag.get()[i]; /* Too lazy for this column */ \
	        l_linestatus[i] = li.l_linestatus.get()[i]; /* Too lazy for this column */ \
	        kernel_compact_init_magic(discount); \
	        kernel_compact_init_magic(tax); \
	        kernel_compact_init_magic(extendedprice); \
	        kernel_compact_init_magic(quantity); \
	    } \
    } while (false)

#define kernel_compact_prologue_magic(col) auto col = li.l_##col.get();

#define kernel_compact_prologue() \
    int8_t* returnflag = (int8_t*)li.l_returnflag.get(); \
	int8_t* linestatus = (int8_t*)li.l_linestatus.get(); \
	const size_t cardinality = li.l_extendedprice.cardinality; \
	kernel_compact_prologue_magic(shipdate); \
	kernel_compact_prologue_magic(discount); \
	kernel_compact_prologue_magic(tax); \
	kernel_compact_prologue_magic(extendedprice); \
	kernel_compact_prologue_magic(quantity);


struct IKernel {
private:
	bool m_clean;
	std::vector<void*> m_alloced;

public:
	IKernel(bool clean = true) : m_clean(clean) {}
	template<typename T>
	inline T* new_array(size_t num)
	{
		T* r = static_cast<T*>(aligned_alloc(4*1024, sizeof(T) * num));
		assert(r);
		assert((size_t) r % 64 == 0);
		memset(r, 0, sizeof(T) * num);
		m_alloced.push_back(r);
		return r;
	}

	~IKernel();
};

struct BaseKernel : IKernel {
	const Date cmp;
	const lineitem& li;

	int64_t sum_aggr_time;
	int64_t sum_magic_time;

	BaseKernel(const lineitem& li)
	 : cmp(Date("1998-12-01", -1, -90) /* 1998-12-01 minus 90 days is */), li(li), sum_aggr_time(0), sum_magic_time(0) {
	}


public:
	virtual void Profile(size_t total_tuples);
};


extern "C" __attribute__((noinline)) void handle_overflow();

inline static int64_t
m128_hsum_epi64(__m128i a)
{
	return _mm_extract_epi64(a, 0) + _mm_extract_epi64(a, 1);
}

inline static int64_t
m256_hsum_epi64(__m256i a)
{
	auto lo = _mm256_extracti128_si256(a, 0);
	auto hi = _mm256_extracti128_si256(a, 1);
	return m128_hsum_epi64(_mm_add_epi64(lo, hi));
}

#ifdef __AVX512F__
inline static int64_t
m512_hsum_epi64(__m512i a)
{
	auto lo = _mm512_extracti64x4_epi64(a, 0);
	auto hi = _mm512_extracti64x4_epi64(a, 1);
	return m256_hsum_epi64(_mm256_add_epi64(lo, hi));
}
#endif

inline static int64_t
m128_hsum_epi32(__m128i a)
{
	return _mm_extract_epi32(a, 0) + _mm_extract_epi32(a, 1) + _mm_extract_epi32(a, 2) + _mm_extract_epi32(a, 3);
}

inline static int64_t
m256_hsum_epi32(__m256i a)
{
	auto lo = _mm256_extracti128_si256(a, 0);
	auto hi = _mm256_extracti128_si256(a, 1);
	return m128_hsum_epi32(_mm_add_epi32(lo, hi));
}

#ifdef __AVX512F__
inline static int64_t
m512_hsum_epi32(__m512i a)
{
	auto lo = _mm512_extracti64x4_epi64(a, 0);
	auto hi = _mm512_extracti64x4_epi64(a, 1);
	return m256_hsum_epi32(_mm256_add_epi32(lo, hi));
}
#endif

static inline constexpr int32_t
population_count(int32_t v)
{
	int32_t c = 0;

	for (; v; v >>= 1) {
		c += v & 1;
	}

	return c;
}

void
print_arr(long long int* a, size_t* indices = nullptr);

#ifdef __AVX512F__

void
print_reg(__m512i r);

void
print_reg16(__m512i r);

/** Generate m512i using a lambda function */
template<typename GEN>
static inline __m512i
_m512_generate_epi32(GEN&& gen)
{
	return _mm512_set_epi32(gen(15), gen(14), gen(13), gen(12), gen(11), gen(10), gen(9), gen(8),
		gen(7), gen(6), gen(5), gen(4), gen(3), gen(2), gen(1), gen(0));
}

// #define POPCOUNT_DEBUG

/* Calculates the number of bits set in each 32-bit element (but only the low 16-bits) in the AVX 512 register
 * using a in-register lookup table. This is pretty much the naive way and requires 4 permutes. */
static const auto popcount_up_to_16_epi32_table = _m512_generate_epi32([] (int idx) { return population_count(idx); });

// 5.4 cycles
inline static __m512i
popcount_up_to_16_epi32(__m512i a, __m512i pop_table)
{
	const auto lo4bits = _mm512_set1_epi32(0x0000000F);

	/* get the first 4 bits and use them as index into a in-register table */
	auto idx = _mm512_and_epi32(a, lo4bits);
	auto pop = _mm512_permutexvar_epi32(idx, pop_table);

	for (int i=0; i<3; i++) {
		/* get the next 4 bits */
		idx = _mm512_srli_epi32(a, (i+1)*4);
		idx = _mm512_and_epi32(idx, lo4bits);
		pop = _mm512_add_epi32(pop, _mm512_permutexvar_epi32(idx, pop_table));
	}

#ifdef POPCOUNT_DEBUG
	/* check results */
	int k = 0;
	_m512_map_epi32(a, pop, [&] (int a, int b) {
		int c = population_count(a);
		if (b != c) {
			printf("k=%d: a=%d pop(a)=%d b=%d\n", k, a, c, b);
			assert(false);
		}
		k++;
	});
#endif

	return pop;
}

/* Inspired by http://aggregate.ee.engr.uky.edu/MAGIC/#Population%20Count%20(Ones%20Count) takes around 5.6 cycles on KNL*/
inline static __m512i
popcount_up_to_16_bitw_epi32(__m512i a)
{
	const auto t1 = _mm512_and_epi32(_mm512_srli_epi32(a, 1), _mm512_set1_epi32(0x55555555));	
	a = _mm512_sub_epi32(a, t1);

	const auto t2 = _mm512_and_epi32(a, _mm512_set1_epi32(0x33333333));
	const auto t3 = _mm512_and_epi32(_mm512_srli_epi32(a, 2), _mm512_set1_epi32(0x33333333));	
	a = _mm512_add_epi32(t2, t3);

	a = _mm512_and_epi32(_mm512_add_epi32(a, _mm512_srli_epi32(a, 4)), _mm512_set1_epi32(0x0F0F0F0F));

	a = _mm512_add_epi32(a, _mm512_srli_epi32(a, 8));
	// a = _mm512_add_epi32(a, _mm512_srli_epi32(a, 16));

	auto pop = _mm512_and_epi32(a, _mm512_set1_epi32(0x0000003F));
#ifdef POPCOUNT_DEBUG
	/* check results */
	int k = 0;
	_m512_map_epi32(a, pop, [&] (int a, int b) {
		int c = population_count(a);
		if (b != c) {
			printf("k=%d: a=%d pop(a)=%d b=%d\n", k, a, c, b);
			assert(false);
		}
		k++;
	});
#endif

	return pop;

}

static const uint8_t pop_table_8[] = {0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8};

// 5.8 cycles on KNL
inline static __m512i
popcount_up_to_16_gather_epi32(__m512i a)
{
	const auto lo8bits = _mm512_set1_epi32(0x000000FF);

	auto lo = _mm512_and_epi32(a, lo8bits);
	auto hi = _mm512_srli_epi32(a, 8);

	/* Look those bytes up */
	auto lut = (const int*)(&pop_table_8[0]);
	auto pop_lo = _mm512_i32gather_epi32(lo, lut, 1);
	auto pop_hi = _mm512_i32gather_epi32(hi, lut, 1);

	/* Low granularity requires masking as we get whole words.
	 * We can do that later because we will never overflow */
	return _mm512_and_epi32(lo8bits, _mm512_add_epi32(pop_lo, pop_hi));
}

#endif

#endif
