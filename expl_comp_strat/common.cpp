#include "common.hpp"
#include <cassert>

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

IKernel::~IKernel()
{
	if (m_clean) {
		for (auto p : m_alloced) {
			free(p);
		}	
	}
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