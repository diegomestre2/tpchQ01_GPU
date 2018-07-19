#ifndef H_CPU
#define H_CPU

struct lineitem;

#include <string>
#include "blockingconcurrentqueue.h"

struct CPUKernel;
struct AggrHashTable;

struct FilterChunk {
	size_t offset;
	size_t num;
};

extern uint32_t* precomp_filter;
extern uint16_t* compr_shipdate;
extern moodycamel::BlockingConcurrentQueue<FilterChunk> precomp_filter_queue;

extern size_t morsel_size;

struct CoProc {
	CPUKernel* kernel;

	AggrHashTable* table;

	CoProc(const lineitem& li, bool wo_core0);
	~CoProc();

	void operator()(size_t offset, size_t num, size_t pushdown_cpu_start_offset);

	void wait();
	void Clear();
	size_t numExtantGroups() const;
		// Avoiding inclusion of anything else.

};

#endif
