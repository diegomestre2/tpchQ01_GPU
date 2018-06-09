#ifndef H_KERNEL_WELD
#define H_KERNEL_WELD

#include <weld.h>

template<typename T>
struct WeldVec {
	T* data = nullptr;
	int64_t size = 0;
};


static const char* weld_q1 = 
// inspired by https://raw.githubusercontent.com/weld-project/weld/97567741a8f03c73fbfe7aa3b0855c64400b4512/benches/benchmarks/tpch/q1.weld
"|aggr_table: vec[{i64,i64,i64,i64,i128,i128}], l_returnflag: vec[i8], l_linestatus: vec[i8], l_quantity: vec[i64],"
"    l_ep: vec[i64], l_discount: vec[i64], l_shipdate: vec[i32], l_tax: vec[i64]|"
"    let sums = result(for("
"        filter(zip(l_returnflag, l_linestatus, l_quantity,"
"            l_ep, l_discount, l_shipdate, l_tax),"
"            |e| e.$5 <= 729999"
"        ),"
"        vecmerger[{i64,i64,i64,i64,i128,i128},+](aggr_table),"
"        |b,i,e|"
"            let sum_disc_price = e.$3 * (100l - e.$4);"
"            merge(b, { "
"            i64(i32(e.$0)*255 + i32(e.$1)),"
"            {"
"                e.$2," // quantity
"                1l," // count
"				 e.$3," // ext_price
"                e.$4," // discount
"                i128(sum_disc_price),"
"                i128(sum_disc_price * (100l + e.$6))" // charge
"            }"
"        })"
"    ));"
"    map(sums, |s| {"
"        s.$0," // quan
"        s.$1," // cnt
"        s.$2," // ext_price
"        s.$3," // discount
"        s.$4," // disc_price
"        s.$5," // charge
"    })";


#define weld_def(nme) \
	input.nme.data = li.l_##nme.get(); \
	input.nme.size = li.l_##nme.cardinality;
	

struct KernelWeld : BaseKernel {
	weld_module_t m;
	weld_error_t e;

	const char* kThreads = "weld.threads";

	struct Inputs {
		WeldVec<AggrHashTable> aggr_table;
		WeldVec<char> returnflag;
		WeldVec<char> linestatus;
		WeldVec<int64_t> quantity;
		WeldVec<int64_t> extendedprice;
		WeldVec<int64_t> discount;
		WeldVec<int> shipdate;
		WeldVec<int64_t> tax;
	};

	struct OutputCols {
		WeldVec<int64_t> sum_quantity;
		WeldVec<int64_t> count;

		WeldVec<int64_t> sum_base_price;
		WeldVec<int64_t> sum_disc;
		
		WeldVec<int128_t> sum_disc_price;
		WeldVec<int128_t> sum_charge;
	};

	struct OutputRow {
		int64_t sum_quantity;
		int64_t count;

		int64_t sum_base_price;
		int64_t sum_disc;
		
		int128_t sum_disc_price;
		int128_t sum_charge;
	};

	typedef WeldVec<OutputRow> OutputRows;

	KernelWeld(const lineitem& li) : BaseKernel(li) {
		e = weld_error_new();
		weld_conf_t conf = weld_conf_new();
		// weld_conf_set(conf, kThreads, "1");
		m = weld_module_compile(weld_q1, conf, e);
		weld_conf_free(conf);

		if (weld_error_code(e)) {
			const char *err = weld_error_message(e);
			printf("Error message: %s\n", err);
			assert(false);
		}
	}

	~KernelWeld() {
		weld_error_free(e);
		weld_module_free(m);
	}

	__attribute__((noinline)) void operator()() noexcept {
		Inputs input;

		input.aggr_table.data = &aggrs0[0];
		input.aggr_table.size = MAX_GROUPS;

		weld_def(returnflag);
		weld_def(linestatus);
		weld_def(quantity);

		weld_def(extendedprice);
		weld_def(discount);
		weld_def(shipdate);
		weld_def(tax);

		weld_value_t arg = weld_value_new(&input);
		weld_conf_t conf = weld_conf_new();
		// weld_conf_set(conf, kThreads, "1");
		weld_value_t result = weld_module_run(m, conf, arg, e);
		OutputRows *result_data = (OutputRows*)weld_value_data(result);

		size_t g = result_data->size;
		for (size_t i=0; i<g; i++) {
			OutputRow* r = &result_data->data[i];
			auto cnt = r->count;
			if (cnt <= 0) {
				continue;
			}

			printf("%c|%c|", i / 255, i % 255);
			printf("%lld|%lld|", r->sum_quantity, r->sum_base_price);
			printf("%lld|%lld|%lld\n", (int64_t)r->sum_disc_price, (int64_t)r->sum_charge, cnt);
		}

		weld_value_free(result);
		weld_conf_free(conf);
		weld_value_free(arg);
	}
};

#endif