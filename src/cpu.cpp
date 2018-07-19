#include "cpu.h"
#include "cpu/common.hpp"
#include "cpu/vectorized.hpp"
#include "cpu/tpch_kit.hpp"
#include "cpu/kernel_x100.hpp"

uint32_t* precomp_filter = nullptr;
uint16_t* compr_shipdate = nullptr;
moodycamel::BlockingConcurrentQueue<FilterChunk> precomp_filter_queue;
size_t morsel_size = 10*1024;

// Wrapper like Eminem
struct CPUKernel {
	Morsel<KernelX100<kMagic, true
#ifdef __AVX512F__
	kPopulationCount
#else
	kNoAvx512
#endif
	>> m;

	CPUKernel(const lineitem& li, bool wo_core0) : m(li, wo_core0) {
	}

	void spawn(size_t offset, size_t num, size_t pushdown_cpu_start_offset) { m.spawn(offset, num, pushdown_cpu_start_offset); }

	void wait(bool active) { m.wait(active); }
};

CoProc::CoProc(const lineitem& li, bool wo_core0)
{
	kernel = new CPUKernel(li, wo_core0);
	table = kernel->m.aggrs0;
}

CoProc::~CoProc()
{
	delete kernel;
}


void
CoProc::operator()(size_t offset, size_t num, size_t pushdown_cpu_start_offset)
{
	kernel->spawn(offset, num, pushdown_cpu_start_offset);
}

void
CoProc::wait()
{
	kernel->wait(false);
}

size_t
CoProc::numExtantGroups() const
{
	unsigned long long num_extant_groups { 0 };
    for (size_t i = 0; i < MAX_GROUPS; i++) {
        if (table[i].count > 0) { num_extant_groups++; }
    }
    return num_extant_groups;
}


void
CoProc::Clear()
{
	kernel->m.Clear();
}
