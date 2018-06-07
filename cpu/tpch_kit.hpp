#ifndef H_TPCH
#define H_TPCH

#include <stdint.h>
#include <string>
#include <cassert>
#include <cmath>
#include <limits>
#include "allocator.hpp"
#include "monetdb.hpp"

struct SkipCol {
	SkipCol(const char* v, int64_t len) {}
};

struct Char {
	char chr_val;

	Char(const char* v, int64_t len) {
		assert(len == 1);
		assert(v[0] && !v[1]);
		chr_val = v[0];
	}
};

struct BaseDecimal {
	static bool parse(const char* first, const char* last, int& intg, int& frac);
};

/* Reinvent standard math function pow() because it couldn't be a constexpr. */
template <typename T>
constexpr T ipow(T num, unsigned int pow)
{
	if (pow == 0) {
		return 1;
	}

	return num * ipow(num, pow - 1);
}

template<size_t PRE = 15, size_t POST = 2, typename ValueT = int64_t>
struct Decimal : BaseDecimal {
	ValueT dec_val;

	static constexpr int64_t int_digits = PRE;
	static constexpr int64_t frac_digits = POST;
	static constexpr ValueT scale = ipow(10, frac_digits);
	static constexpr int64_t highest_num = scale - 1;

	static constexpr ValueT ToValue(int64_t i, int64_t f) {
		return scale * (ValueT)i + f;
	}

	static constexpr int64_t GetFrac(ValueT v) {
		return v % scale;
	}

	static constexpr int64_t GetInt(ValueT v) {
		return v / scale;
	}

	static constexpr ValueT Mul(ValueT a, ValueT b) {
		return a * b; // scale will be wrong
	}

	Decimal(const char* v, int64_t len) {
		const char* begin = v;
		const char* end = v + len;
		int intg = 0;
		int frac = 0;

		if (!parse(begin, end, intg, frac)) {
			assert(false && "parsing failed");
		}

		dec_val = ToValue(intg, frac);

		assert(intg == GetInt(dec_val));
		// printf("org=%d new=%d frac_bits=%d\n", frac, GetFrac(dec_val), shift_amount);
		assert(frac == GetFrac(dec_val));
	}
};

struct Date {
	static bool parse(const char* first, const char* last, int& day, int& month, int& year);

	int dte_val;

	Date(const char* v, int64_t len = -1, int plus_days = 0);
};



template<typename T = int64_t>
struct MinMax {
	int64_t min = std::numeric_limits<T>::max();
	int64_t max = std::numeric_limits<T>::min();

	void operator()(const T& val) {
		if (val <= min) {
			min = val;
		}
		if (val >= max) {
			max = val;
		}
	}
};

template<typename T>
struct Column : Buffer<T> {
	size_t cardinality;

	MinMax<T> minmax;

	Column(size_t init_cap)
	 : Buffer<T>(init_cap), cardinality(0) {
	}

	bool HasSpaceFor(size_t n) const {
		return cardinality + n < Buffer<T>::capacity();
	}

	void Push(const T& val) {
		if (!HasSpaceFor(1)) {
			Buffer<T>::resizeByFactor(1.5);
		}
		assert(HasSpaceFor(1));
		auto data = Buffer<T>::get();
		data[cardinality++] = val;
		minmax(val);
	}
};

// starting from 1
struct lineitem {
	Column<char> l_returnflag; // 9
	Column<char> l_linestatus; // 10
	Column<int64_t> l_quantity; // 5, DECIMAL(15,2)
	Column<int64_t> l_extendedprice; // 6, DECIMAL(15,2)
	Column<int64_t> l_discount; // 7, DECIMAL(15,2)
	Column<int64_t> l_tax; // 8, DECIMAL(15,2)
	Column<int> l_shipdate; // 11
public:
	lineitem(size_t init_cap)
	 : l_returnflag(init_cap), l_linestatus(init_cap), l_quantity(init_cap), l_extendedprice(init_cap), l_discount(init_cap), l_tax(init_cap), l_shipdate(init_cap) {
	}

	void FromFile(const std::string& file);
};

#endif