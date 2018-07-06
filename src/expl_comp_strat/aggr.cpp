#include "aggr.hpp"

int run_1()
{
	run_till_2048_groups(kDirect);
}

extern int run_2();
extern int run_3();
extern int run_4();

int main() {
	printf("#type numAggrs numGroups millis cycles\n");

	run_4(); run_1(); run_2(); run_3();
	return 0;
}
