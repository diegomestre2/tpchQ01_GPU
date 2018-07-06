#include <sstream>
#include <vector>
#include <time.h>
#include "common.hpp"
#include "vectorized.hpp"
#include "tpch_kit.hpp"

#include "kernel_naive.hpp"
#include "kernel_naive_compact.hpp"
#include "kernel_hyper.hpp"
#include "kernel_hyper_compact.hpp"
#include "kernel_x100.hpp"
#include "kernel_x100_old.hpp"
#include "kernel_avx512.hpp"

#include <tuple>

#define PRINT_RESULTS

static const size_t REP_COUNT = 25;

static int runIdCounter = 0;

template<typename F, typename... Args>
void run(const lineitem& li, const std::string& name, Args&&... args)
{
	F fun(li, args...);
	size_t total_time = 0;
	double total_millis = 0.0;

	const size_t n = li.l_extendedprice.cardinality;

	for (size_t rep=0; rep<REP_COUNT; rep++) {
		fun.Clear();

		timespec ts_start, ts_end;
		clock_gettime(CLOCK_MONOTONIC, &ts_start);
		const auto start = get_cycles();

		fun();

		const auto time = get_cycles() - start;
		clock_gettime(CLOCK_MONOTONIC, &ts_end);
		double million = 1000000.0;
		int64_t millisec = (ts_end.tv_sec - ts_start.tv_sec) * 1000.0 + ((ts_end.tv_nsec - ts_start.tv_nsec) / million);

		if (rep != 0) { /* Throw cold run away */
			total_time += time;
			total_millis += millisec;
		}
	}

	const double hot_reps = REP_COUNT-1;
	const double total_tuples = hot_reps * n;
	
	printf("%d \t %-40s \t %.1f       \t %.1f       \t %.1f       \t %.1f       \t %.1f\n",
		runIdCounter, name.c_str(), (double)total_time / total_tuples,
		(double)total_millis / hot_reps,
		(double)fun.sum_aggr_time / total_tuples,
		(double)fun.sum_magic_time / total_tuples,
		(double)(total_time - 0 - fun.sum_aggr_time - fun.sum_magic_time) / total_tuples);

	runIdCounter++;
	// fun.Profile(total_tuples);
#ifdef PRINT_RESULTS

	auto print_dec = [] (auto s, auto x) { printf("%s%ld.%ld", s, Decimal64::GetInt(x), Decimal64::GetFrac(x)); };
	// printf("# returnflag|linestatus|sum_qty|sum_base_price|sum_disc_price|sum_charge|count_order\n");

	auto sum_64 = [] (auto& arr, size_t i, size_t n) {
		int64_t start = 0;
		for (size_t k=i; k<i+n; k++) {
			start += arr[k];
		}
		return start;
	};

	auto sum_128 = [] (auto& hi, auto& lo, size_t i, size_t n) {
		__int128 start = 0;
		for (size_t k=i; k<i+n; k++) {
			auto h = (__int128)hi[k] << 64;
			auto l = (__int128)lo[k];
			start += h | l;
		}
		return start;
	};

	for (size_t group=0; group<MAX_GROUPS; group++) {
		if (fun.aggrs0[group].count > 0) {
			char rf = group >> 8;
			char ls = group & std::numeric_limits<unsigned char>::max();

			size_t i = group;

			printf("# %c|%c", rf, ls);
			print_dec("|", fun.aggrs0[i].sum_quantity);
			print_dec("|", fun.aggrs0[i].sum_base_price);
			print_dec("|", fun.aggrs0[i].sum_disc_price);
			print_dec("|", fun.aggrs0[i].sum_charge);
			printf("|%ld\n", fun.aggrs0[i].count);
		}
	}
	size_t i=0;
	for (size_t group=0; group<MAX_GROUPS; group++) {
		char rf = group >> 8;
		char ls = group & std::numeric_limits<unsigned char>::max();

		int64_t count = sum_64(fun.aggr_avx0_count, i, 8);
		
		if (count > 0) {
			char rf = group >> 8;
			char ls = group & std::numeric_limits<unsigned char>::max();

			printf("# %c|%c", rf, ls);
			print_dec("|", sum_64(fun.aggr_avx0_sum_quantity, i, 8));
			print_dec("|", sum_64(fun.aggr_avx0_sum_base_price, i, 8));
			print_dec("|", sum_128(fun.aggr_avx0_sum_disc_price_hi, fun.aggr_avx0_sum_disc_price_lo, i, 8));
			print_dec("|", sum_128(fun.aggr_avx0_sum_charge_hi, fun.aggr_avx0_sum_charge_lo, i, 8));
			printf("|%ld\n", count);
		}

		i+=8;
	}
#endif
}

using namespace std;

inline bool file_exists(const string& name) {
    struct stat buffer;
    return (stat (name.c_str(), &buffer) == 0);
}

inline std::string join_path(std::string a, std::string b) {
    return a + "/" + b;
}

std::pair<string,string> split_once(string delimited, char delimiter) {
    auto pos = delimited.find_first_of(delimiter);
    return { delimited.substr(0, pos), delimited.substr(pos+1) };
}

int main(int argc, const char** argv) {
    double scale_factor = 1;


    for(int i = 1; i < argc; i++) {
        auto arg = string(argv[i]);
        if (arg.substr(0,2) != "--") {
            exit(1);
        }
        arg = arg.substr(2);

        // A  name=value argument
        auto p = split_once(arg, '=');
        auto& arg_name = p.first; auto& arg_value = p.second;
        if (arg_name == "scale-factor") {
            scale_factor = std::stod(arg_value);
            if (scale_factor - 0 < 0.001) {
                std::invalid_argument("Invalid scale factor");
            }
        } else {
            exit(1);
        }
    }
    std::string tpch_directory = join_path(EXPAND_THEN_QUOTE(DATA_FILES_DIR) , std::to_string(scale_factor));
    std::string input_file = join_path(tpch_directory, "lineitem.tbl");

    if (not file_exists(input_file.c_str())) {
        throw std::runtime_error("Cannot locate table text file " + input_file);
        // Not generating it ourselves - that's: 1. Not healthy and 2. Not portable;
        // setup scripts are intended to do that
    }

	/* load data */

    lineitem li((size_t)(7000000 * std::max(scale_factor, 1.0)));
    li.FromFile(input_file.c_str());

	/* start processing */

	printf("ID \t %-40s \t timetuple \t millisec \t aggrtuple \t pshuffletuple \t remainingtuple\n",
		"Configuration");

//	run<KernelWeld>(li, "$\\text{Weld}$");

#if 0
	run<KernelOldX100<kMultiplePrims, true, kSinglePrims, false>>(li, "$\\text{X100 Full NSM Standard}$");
	run<KernelOldX100<kMultiplePrims, false, kSinglePrims, false>>(li, "$\\text{X100 Full DSM Standard}$");
	run<KernelOldX100<k1Step, true, kSinglePrims, false>>(li, "$\\text{X100 Full NSM Standard Fused}$");
	run<KernelOldX100<kMagic, true, kSinglePrims, true>>(li, "$\\text{X100 Full NSM In-Reg}$");

	run<KernelX100<kMultiplePrims, true>>(li, "$\\text{X100 Compact NSM Standard}$");
	run<KernelX100<kMultiplePrims, false>>(li, "$\\text{X100 Compact DSM Standard}$");
	run<KernelX100<k1Step, true>>(li, "$\\text{X100 Compact NSM Standard Fused}$");
#endif
	run<KernelX100<kMagic, true>>(li, "$\\text{X100 Compact NSM In-Reg}$", 0);

	run<Morsel<KernelX100<kMagic, true>, true>>(li, "$\\text{Full system Morsel X100 Compact NSM In-Reg}$");
	run<Morsel<KernelX100<kMagic, true>, false>>(li, "$\\text{One socket Morsel X100 Compact NSM In-Reg}$");
	

	//run<Morsel<KernelNaiveCompact>>(li, "$\\text{HyPer Compact NoOverflow}$");
	//run<KernelNaiveCompact>(li, "$\\text{HyPer Compact NoOverflow}$", 0);
	

#if 0	
#ifdef __AVX512F__
	run<KernelX100<kMagic, true, kPopulationCount>>(li, "$\\text{X100 Compact NSM In-Reg AVX-512}$");
	run<KernelX100<kMagic, true, kCompare>>(li, "$\\text{X100 Compact NSM In-Reg AVX-512 Cmp}$");
#endif
	run<KernelHyPer<true>>(li, "$\\text{HyPer Full}$");
	run<KernelHyPer<false>>(li, "$\\text{HyPer Full OverflowBranch}$");
	run<KernelNaive>(li, "$\\text{HyPer Full NoOverflow}$");

	

	run<KernelHyPerCompact<true>>(li, "$\\text{HyPer Compact}$");
	run<KernelHyPerCompact<false>>(li, "$\\text{HyPer Compact OverflowBranch}$");
	run<KernelNaiveCompact>(li, "$\\text{HyPer Compact NoOverflow}$");

#ifdef __AVX512F__
	run<AVX512<false, false, true>>(li, "$\\text{Handwritten AVX-512}$");
	run<AVX512<false, false, false>>(li, "$\\text{Handwritten AVX-512 Only64BitAggr}$");
#endif
#endif
	return 0;
}
