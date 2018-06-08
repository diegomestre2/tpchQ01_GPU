#include "cpu.h"
#include "cpu/common.hpp"
#include "cpu/vectorized.hpp"
#include "cpu/tpch_kit.hpp"
#include "cpu/kernel_x100.hpp"

// Wrapper like Eminem
struct CPUKernel {
	Morsel<KernelX100<kMagic, true>> m;

	CPUKernel(const lineitem& li, bool wo_core0) : m(li, wo_core0) {
	}

	void spawn(size_t offset, size_t num) { m.spawn(offset, num); }

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
CoProc::operator()(size_t offset, size_t num)
{
	kernel->spawn(offset, num);
}

void
CoProc::wait()
{
	kernel->wait(true);
}

void
CoProc::Clear()
{
	kernel->m.Clear();
}