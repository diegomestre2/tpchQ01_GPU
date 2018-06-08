#include "tpch_kit.hpp"
#include "common.hpp"

#include <fstream>
#include <cassert>
#include <tuple>
#include <ctime>

template<typename ...Types>
class TableReader {
	static constexpr size_t NUM_COLS = sizeof...(Types);

	size_t m_start_pos;

	char* m_word_starts[NUM_COLS];
	int64_t m_lengths[NUM_COLS];

	static constexpr char kTerminator = '\0';
	static constexpr char kSeperator = '|';
	static constexpr char kNewline = '\n';

	static inline bool terminator(char c) {
		return (c == kTerminator) | (c == kNewline);
	}

	static char* tokenize(char* line, int64_t max_words, char** words, int64_t* lengths,
			int64_t& out_num_words) {
		int64_t num_words = 0;
		int64_t word_start = 0;
		int64_t i = 0;

		while (1) {
			char c = line[i];
			if ((terminator(c)) | (c == kSeperator)) {
				words[num_words] = &line[word_start];
				lengths[num_words] = i - word_start;

				// replace kSeperator with kTerminator
				line[i] = kTerminator;

				// saves one register for the iteration variables
				word_start = i + 1;

				num_words++;

				if (terminator(c)) {
					break;
				}
				if (num_words == max_words) {
					out_num_words = num_words;
					return &line[word_start];
				}
			}
			i++;
		}

		out_num_words = num_words;
		return nullptr;
	}

	template <typename T>
	std::tuple<T> parse(int64_t idx) const
	{
		T t(m_word_starts[idx], m_lengths[idx]);
		return std::tuple<T>(std::move(t));
	}

	template <typename T, typename Arg, typename... Args>
	std::tuple<T, Arg, Args...> parse(int64_t idx)  const
	{
		T t(m_word_starts[idx], m_lengths[idx]);
		return std::tuple_cat(std::tuple<T>(std::move(t)),
			parse<Arg, Args...>(idx+1));
	}

public:
	template<typename PushFun>
	void DoLine(char* str, PushFun&& push) {
		int64_t num = 0;
		tokenize(str, NUM_COLS, m_word_starts, m_lengths, num);
		assert(num >= NUM_COLS);

		push(parse<Types...>(0));
	}

	template<typename PushFun>
	void DoFile(const std::string& file, PushFun&& push) {
		std::ifstream f;

		f.open(file, std::ifstream::in);
		assert(f.is_open());
		const auto N = 1024*1024;
		char buf[N];
		while (f.getline(buf, N)) {
			DoLine(buf, std::forward<PushFun>(push));
		}
	}
};

#include <boost/config/warning_disable.hpp>
#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/phoenix_core.hpp>
#include <boost/spirit/include/phoenix_operator.hpp>

bool
BaseDecimal::parse(const char* first, const char* last, int& intg, int& frac)
{
    using boost::spirit::qi::int_;
    using boost::spirit::qi::_1;
    using boost::spirit::qi::phrase_parse;
    using boost::spirit::ascii::space;
    using boost::phoenix::ref;

    bool r = phrase_parse(first, last,
        ( int_[ref(intg) = _1] >> '.' >> int_[ref(frac) = _1] | int_[ref(intg) = _1] ),
        space);

    if (!r || first != last) // fail if we did not get a full match
        return false;
    return r;
}

bool
Date::parse(const char* first, const char* last, int& day, int& month, int& year)
{
    using boost::spirit::qi::int_;
    using boost::spirit::qi::_1;
    using boost::spirit::qi::phrase_parse;
    using boost::spirit::ascii::space;
    using boost::phoenix::ref;

    bool r = phrase_parse(first, last,
        ( int_[ref(year) = _1] >> '-' >> int_[ref(month) = _1] >> '-' >> int_[ref(day) = _1]),
        space);

    if (!r || first != last) // fail if we did not get a full match
        return false;

    return r;
}

Date::Date(const char* v, int64_t len, int plus_days)
{
	if (len < 0) {
		assert(len == -1);
		len = strlen(v);
	}
	int day = 0, month = 0, year = 0;
	if (!parse(v, v + len, day, month, year)) {
		assert(false);
	}
	dte_val = todate(day, month, year);
	dte_val += plus_days;
}

void
lineitem::FromFile(const std::string& file)
{
	using Decimal = Decimal<15, 2>;

	const clock_t begin = clock();
	TableReader<SkipCol, SkipCol, SkipCol, SkipCol, Decimal, Decimal, Decimal, Decimal, Char, Char, Date> reader;
	reader.DoFile(file, [&] (auto t) {
		assert(std::get<8>(t).chr_val);
		assert(std::get<9>(t).chr_val);

		l_quantity.Push(std::get<4>(t).dec_val);
		l_extendedprice.Push(std::get<5>(t).dec_val);
		l_discount.Push(std::get<6>(t).dec_val);
		l_tax.Push(std::get<7>(t).dec_val);
		l_returnflag.Push(std::get<8>(t).chr_val);
		l_linestatus.Push(std::get<9>(t).chr_val);
		l_shipdate.Push(std::get<10>(t).dte_val);

	});

	const clock_t end = clock();
	const double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

#ifdef PRINT_MINMAX
	printf("loaded %zu tuples in %.2fs i.e. %.2f kTuples/s\n", l_quantity.cardinality, elapsed_secs, (double)l_quantity.cardinality / elapsed_secs / 1000.0);


	#define mm(name) printf("%s min=%lld max=%lld diff=%lld\n", #name, l_##name.minmax.min, l_##name.minmax.max, l_##name.minmax.max - l_##name.minmax.min)

	mm(quantity);
	mm(extendedprice);
	mm(discount);
	mm(tax);
	mm(returnflag);
	mm(linestatus);
	mm(shipdate);
	#undef mm
#endif
}