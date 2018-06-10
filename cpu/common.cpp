#include "common.hpp"
#include <cassert>

#define clear(x) memset(x, 0, sizeof(x[0]) * MAX_GROUPS)
#define clear_avx(x) memset(x, 0, sizeof(x[0]) * AVX_GROUPS)

extern "C" __attribute__((noinline)) void
handle_overflow()
{
	assert(false && "overflow");
}

#ifdef __AVX512F__

void
print64_512 (const char* prefix, __m512i a)
{
	union {
		__m512i avx;
		int64_t b[8];
	} r;

	printf("%-10s ", prefix);

	r.avx = a;
	for (size_t i=0; i<8; i++) {
		printf("%s%-10lld", i ? ", " : "", r.b[i]);
	}
	printf("\n");
}

void
print32_512(const char* prefix, __m512i a)
{
	union {
		__m512i avx;
		int32_t b[16];
	} r;

	printf("%-10s ", prefix);

	r.avx = a;
	for (size_t i=0; i<16; i++) {
		printf("%s%-10ld", i ? ", " : "", r.b[i]);
	}

	printf("\n");
}

#endif /* __AVX512F__ */


void
BaseKernel::Profile(size_t total_tuples)
{

}

BaseKernel::BaseKernel(const lineitem& li)
 : cmp(Date("1998-12-01", -1, -90) /* 1998-12-01 minus 90 days is */),
	li(li), sum_aggr_time(0), sum_magic_time(0)
{
	aggrs0 = new_array<AggrHashTable>(MAX_GROUPS);

	aggr_dsm0_sum_quantity = new_array<int64_t>(MAX_GROUPS);
	aggr_dsm0_count = new_array<int64_t>(MAX_GROUPS);
	aggr_dsm0_sum_base_price = new_array<int64_t>(MAX_GROUPS);
	aggr_dsm0_sum_disc_price = new_array<int128_t>(MAX_GROUPS);
	aggr_dsm0_sum_charge = new_array<int128_t>(MAX_GROUPS);
	aggr_dsm0_sum_disc = new_array<int64_t>(MAX_GROUPS);

	aggr_avx0_count = new_array<int64_t>(AVX_GROUPS);
	aggr_avx0_sum_quantity = new_array<int64_t>(AVX_GROUPS);
	aggr_avx0_sum_base_price = new_array<int64_t>(AVX_GROUPS);
	aggr_avx0_sum_disc_price_lo = new_array<int64_t>(AVX_GROUPS);
	aggr_avx0_sum_disc_price_hi = new_array<int64_t>(AVX_GROUPS);
	aggr_avx0_sum_charge_lo = new_array<int64_t>(AVX_GROUPS);
	aggr_avx0_sum_charge_hi = new_array<int64_t>(AVX_GROUPS);
	aggr_avx0_sum_disc = new_array<int64_t>(AVX_GROUPS);
}

void
BaseKernel::Clear()
{
	clear(aggrs0);

	clear(aggr_dsm0_sum_quantity);
	clear(aggr_dsm0_count);
	clear(aggr_dsm0_sum_base_price);
	clear(aggr_dsm0_sum_disc_price);
	clear(aggr_dsm0_sum_charge);
	clear(aggr_dsm0_sum_disc);

	clear_avx(aggr_avx0_count);
	clear_avx(aggr_avx0_sum_quantity);
	clear_avx(aggr_avx0_sum_base_price);
	clear_avx(aggr_avx0_sum_disc_price_lo);
	clear_avx(aggr_avx0_sum_charge_lo);
	clear_avx(aggr_avx0_sum_disc_price_hi);
	clear_avx(aggr_avx0_sum_charge_hi);
	clear_avx(aggr_avx0_sum_disc);
}

IKernel::~IKernel()
{
	if (m_clean && false) {
		for (auto p : m_alloced) {
			free(p);
		}	
	}
}

ComprData::ComprData(const lineitem& li) : BaseKernel(li)
{
	size_t cardinality = li.l_extendedprice.cardinality;
    l_shipdate = new_array<int16_t>(cardinality); 
    l_returnflag = new_array<int8_t>(cardinality);
    l_linestatus = new_array<int8_t>(cardinality);
    l_discount = new_array<int8_t>(cardinality);
    l_tax = new_array<int8_t>(cardinality);
    l_extendedprice = new_array<int32_t>(cardinality);
    l_quantity = new_array<int16_t>(cardinality);
    for (size_t i=0; i<cardinality; i++) {
#ifndef GPU
        kernel_compact_init_magic(shipdate);
#else
        l_shipdate[i] = li.l_shipdate.get()[i] - 727563;
        assert(l_shipdate[i] == li.l_shipdate.get()[i] - 727563);
#endif
        l_returnflag[i] = li.l_returnflag.get()[i]; /* Too lazy for this column */
        l_linestatus[i] = li.l_linestatus.get()[i]; /* Too lazy for this column */
        kernel_compact_init_magic(discount);
        kernel_compact_init_magic(tax);
        kernel_compact_init_magic(extendedprice);
        kernel_compact_init_magic(quantity);
    }
}

#include <numa.h>
#include <mutex>

std::mutex numa_data_mutex;
ComprData** numa_data;

size_t
ComprData::GetNumaNodes()
{
	assert(numa_available() != -1);

	return numa_num_configured_nodes();
}

ComprData*
ComprData::Get(const lineitem& li, size_t numa_node)
{
	size_t numa_nodes = GetNumaNodes();

	assert(numa_node < numa_nodes);

	std::unique_lock<std::mutex> lock(numa_data_mutex);

	if (!numa_data) {
		numa_data = new ComprData*[numa_nodes];
		for (size_t i=0; i<numa_nodes; i++) {
			numa_data[i] = nullptr;
		}
	}

	if (!numa_data[numa_node]) {
		numa_data[numa_node] = new ComprData(li);
	}

	return numa_data[numa_node];
}

void
print_arr(long long int* a, size_t* indices)
{
	if (indices) {
		for (size_t i=0; i<8; i++) {
			if (i != 0) {
				printf(", ");
			}
			printf("%ld", a[indices[i]]);
			
		}
	} else {
		for (size_t i=0; i<8; i++) {
			if (i != 0) {
				printf(", ");
			}
			printf("%ld", a[i]);
			
		}
	}
	printf("\n");
}

#ifdef __AVX512F__
void
print_reg(__m512i r)
{
	union {
		__m512i a;
		long long int b[8];
	} test;

	test.a = r;

	print_arr((long long int*)&test.b);	
}

void
print_reg16(__m512i r)
{
	union {
		__m512i a;
		int b[16];
	} test;

	test.a = r;

	for (size_t i=0; i<16; i++) {
		if (i != 0) {
			printf(", ");
		}
		printf("%d", test.b[i]);
		
	}
	printf("\n");
}
#endif