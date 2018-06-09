#ifndef H_CPU
#define H_CPU

struct lineitem;

#include <string>

struct CPUKernel;
struct AggrHashTable;

struct CoProc {
	CPUKernel* kernel;

	AggrHashTable* table;

	CoProc(const lineitem& li, bool wo_core0);
	~CoProc();

	void operator()(size_t offset, size_t num);

	void wait();
	void Clear();
};

#endif
