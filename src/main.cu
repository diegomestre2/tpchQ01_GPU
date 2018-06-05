#include <iostream>
#include <cuda/api_wrappers.h>
#include <vector>
#include <iomanip>

#include "kernel.hpp"
#include "../expl_comp_strat/tpch_kit.hpp"

size_t magic_hash(char rf, char ls) {
    return (((rf - 'A')) - (ls - 'F'));
}

#define GIGA (1024 * 1024 * 1024)
#define MEGA (1024 * 1024)
#define KILO (1024)

using timer = std::chrono::high_resolution_clock;

inline bool file_exists (const std::string& name) {
  struct stat buffer;
  return (stat (name.c_str(), &buffer) == 0);
}

#define INITIALIZE_MEMORY(ptrfunc) { \
    auto _shipdate      = ptrfunc< SHIPDATE_TYPE[]       >(cardinality); \
    auto _discount      = ptrfunc< DISCOUNT_TYPE[]       >(cardinality); \
    auto _extendedprice = ptrfunc< EXTENDEDPRICE_TYPE[]  >(cardinality); \
    auto _tax           = ptrfunc< TAX_TYPE[]            >(cardinality); \
    auto _quantity      = ptrfunc< QUANTITY_TYPE[]       >(cardinality); \
    auto _returnflag    = ptrfunc< RETURNFLAG_TYPE[]     >(cardinality); \
    auto _linestatus    = ptrfunc< LINESTATUS_TYPE[]     >(cardinality); \
    shipdate = _shipdate.get(); \
    discount = _discount.get(); \
    extendedprice = _extendedprice.get(); \
    tax = _tax.get(); \
    quantity = _quantity.get(); \
    returnflag = _returnflag.get(); \
    linestatus = _linestatus.get(); \
    _shipdate.release(); \
    _discount.release(); \
    _extendedprice.release(); \
    _tax.release(); \
    _quantity.release(); \
    _returnflag.release(); \
    _linestatus.release(); \
}

int main(int argc, char** argv) {
    if (!file_exists("lineitem.tbl")) {
        fprintf(stderr, "lineitem.tbl not found!\n");
        exit(1);
    }
    std::cout << "TPC-H Query 1" << '\n';
    get_device_properties();
    /* load data */
    auto start_csv = timer::now();
    size_t cardinality;
    lineitem li(7000000ull);
    li.FromFile("lineitem.tbl");
    auto end_csv = timer::now();
    kernel_prologue();

    bool USE_PINNED_MEMORY = true;

    for(int i = 0; i < argc; i++) {
        auto arg = std::string(argv[i]);
        if (arg == "--no-pinned-memory") {
            USE_PINNED_MEMORY = false;
        }
    }

    auto size_per_tuple = sizeof(SHIPDATE_TYPE) + sizeof(DISCOUNT_TYPE) + sizeof(EXTENDEDPRICE_TYPE) + sizeof(TAX_TYPE) + sizeof(QUANTITY_TYPE) + sizeof(RETURNFLAG_TYPE) + sizeof(LINESTATUS_TYPE);

    auto start_preprocess = timer::now();

    SHIPDATE_TYPE* shipdate;
    DISCOUNT_TYPE* discount;
    EXTENDEDPRICE_TYPE* extendedprice;
    TAX_TYPE* tax;
    QUANTITY_TYPE* quantity;
    RETURNFLAG_TYPE* returnflag;
    LINESTATUS_TYPE* linestatus;
    if (USE_PINNED_MEMORY) {
        INITIALIZE_MEMORY(cuda::memory::host::make_unique);
    } else {
        INITIALIZE_MEMORY(std::make_unique);
    }

    for(size_t i = 0; i < cardinality; i++) {
        shipdate[i]      = _shipdate[i] - SHIPDATE_MIN;
        discount[i]      = _discount[i];
        extendedprice[i] = _extendedprice[i];
        linestatus[i]    = _linestatus[i];
        returnflag[i]    = _returnflag[i];
        quantity[i]      = _quantity[i] / 100;
        tax[i]           = _tax[i];

        assert((int)shipdate[i]           == _shipdate[i] - SHIPDATE_MIN);
        assert((int64_t) discount[i]      == _discount[i]);
        assert((int64_t) extendedprice[i] == _extendedprice[i]);
        assert((char) linestatus[i]       == _linestatus[i]);
        assert((char) returnflag[i]       == _returnflag[i]);
        assert((int64_t) quantity[i]      == _quantity[i] / 100);
        assert((int64_t) tax[i]           == _tax[i]);
    }
    auto end_preprocess = timer::now();

    assert(cardinality > 0 && "Prevent BS exception");
    const size_t data_length = cardinality;
    const int nStreams = static_cast<int>((data_length + TUPLES_PER_STREAM - 1) / TUPLES_PER_STREAM);
    clear_tables();

    /* Allocate memory on device */
    auto current_device = cuda::device::current::get();

    auto d_shipdate      = cuda::memory::device::make_unique< SHIPDATE_TYPE[]      >(current_device, data_length);
    auto d_discount      = cuda::memory::device::make_unique< DISCOUNT_TYPE[]      >(current_device, data_length);
    auto d_extendedprice = cuda::memory::device::make_unique< EXTENDEDPRICE_TYPE[] >(current_device, data_length);
    auto d_tax           = cuda::memory::device::make_unique< TAX_TYPE[]           >(current_device, data_length);
    auto d_quantity      = cuda::memory::device::make_unique< QUANTITY_TYPE[]      >(current_device, data_length);
    auto d_returnflag    = cuda::memory::device::make_unique< RETURNFLAG_TYPE[]    >(current_device, data_length);
    auto d_linestatus    = cuda::memory::device::make_unique< LINESTATUS_TYPE[]    >(current_device, data_length);
    auto d_aggregations  = cuda::memory::device::make_unique< AggrHashTable[]      >(current_device, MAX_GROUPS);

    cudaMemset(d_aggregations.get(), 0, sizeof(AggrHashTable)*MAX_GROUPS);

    /* Transfer data to device */
    cudaStream_t streams[nStreams];
    for (int i = 0; i < nStreams; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    double copy_time = 0;
    double computation_time = 0;

    cuda_check_error();
    auto start = timer::now();
    for (int i = 0; i < nStreams; ++i) {

        size_t offset = i * TUPLES_PER_STREAM;
        size_t size = std::min((size_t) TUPLES_PER_STREAM, (size_t) (data_length - offset));

        //std::cout << "Stream " << i << ": " << "[" << offset << " - " << offset + size << "]" << std::endl;

        auto start_copy = timer::now();
        cuda::memory::async::copy(d_shipdate.get()      + offset, shipdate      + offset, size * sizeof(SHIPDATE_TYPE),     streams[i]);
        cuda::memory::async::copy(d_discount.get()      + offset, discount      + offset, size * sizeof(DISCOUNT_TYPE), streams[i]);
        cuda::memory::async::copy(d_extendedprice.get() + offset, extendedprice + offset, size * sizeof(EXTENDEDPRICE_TYPE), streams[i]);
        cuda::memory::async::copy(d_tax.get()           + offset, tax           + offset, size * sizeof(TAX_TYPE), streams[i]);
        cuda::memory::async::copy(d_quantity.get()      + offset, quantity      + offset, size * sizeof(QUANTITY_TYPE), streams[i]);
        cuda::memory::async::copy(d_returnflag.get()    + offset, returnflag    + offset, size * sizeof(RETURNFLAG_TYPE),    streams[i]);
        cuda::memory::async::copy(d_linestatus.get()    + offset, linestatus    + offset, size * sizeof(LINESTATUS_TYPE),    streams[i]);
        auto end_copy = timer::now();
        copy_time += std::chrono::duration<double>(end_copy - start_copy).count();
#if 0
    }

    for (int i = 0; i < nStreams; ++i) {
        size_t offset = i * TUPLES_PER_STREAM;
        assert(offset <= data_length);
        size_t size = std::min((size_t) TUPLES_PER_STREAM, (size_t) (data_length - offset));
#endif
        size_t amount_of_blocks = size / (VALUES_PER_THREAD * THREADS_PER_BLOCK) + 1;
        size_t SHARED_MEMORY = 0; //sizeof(AggrHashTableLocal) * 18 * THREADS_PER_BLOCK;
        auto start_kernel = timer::now();
        //std::cout << "Execution <<<" << amount_of_blocks << "," << THREADS_PER_BLOCK << "," << SHARED_MEMORY << ">>>" << std::endl;
        cuda::thread_local_tpchQ01<<<amount_of_blocks, THREADS_PER_BLOCK, SHARED_MEMORY, streams[i]>>>(
            d_shipdate.get()      + offset,
            d_discount.get()      + offset,
            d_extendedprice.get() + offset,
            d_tax.get()           + offset,
            d_returnflag.get()    + offset,
            d_linestatus.get()    + offset,
            d_quantity.get()      + offset,
            d_aggregations.get(),
            (u64_t) size);
        auto end_kernel = timer::now();
        computation_time += std::chrono::duration<double>(end_kernel - start_kernel).count();
    }

    cuda_check_error();
    for (int i = 0; i < nStreams; ++i) {
        //size_t offset = i * TUPLES_PER_STREAM;

        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }
    cudaDeviceSynchronize();
    auto end = timer::now();

    cuda::memory::copy(aggrs0, d_aggregations.get(), sizeof(AggrHashTable)*MAX_GROUPS);

    std::cout << "\n"
                 "+--------------------------------------------------- Results ---------------------------------------------------+\n";
    std::cout << "|  LS | RF | sum_quantity        | sum_base_price      | sum_disc_price      | sum_charge          | count      |\n";
    std::cout << "+---------------------------------------------------------------------------------------------------------------+\n";
    auto print_dec = [] (auto s, auto x) { printf("%s%16ld.%02ld", s, Decimal64::GetInt(x), Decimal64::GetFrac(x)); };
    for (size_t group=0; group<MAX_GROUPS; group++) {
        if (aggrs0[group].count > 0) {
            size_t i = group;
            char rf = '-', ls = '-';
            if (group == magic_hash('A', 'F')) {
                rf = 'A';
                ls = 'F';
                if (cardinality == 6001215) {
                    assert(aggrs0[i].sum_quantity == 3773410700);
                    assert(aggrs0[i].count == 1478493);
                }
            } else if (group == magic_hash('N', 'F')) {
                rf = 'N';
                ls = 'F';
                if (cardinality == 6001215) {
                    assert(aggrs0[i].sum_quantity == 99141700);
                    assert(aggrs0[i].count == 38854);
                }
            } else if (group == magic_hash('N', 'O')) {
                rf = 'N';
                ls = 'O';
                if (cardinality == 6001215) {
                    assert(aggrs0[i].sum_quantity == 7447604000);
                    assert(aggrs0[i].count == 2920374);
                }
            } else if (group == magic_hash('R', 'F')) {
                rf = 'R';
                ls = 'F';
                if (cardinality == 6001215) {
                    assert(aggrs0[i].sum_quantity == 3771975300);
                    assert(aggrs0[i].count == 1478870);
                }
            }

            printf("| # %c | %c ", rf, ls);
            print_dec(" | ",  aggrs0[i].sum_quantity);
            print_dec(" | ",  aggrs0[i].sum_base_price);
            print_dec(" | ",  aggrs0[i].sum_disc_price);
            print_dec(" | ",  aggrs0[i].sum_charge);
            printf(" | %10llu |\n", aggrs0[i].count);
        }
    }
    std::cout << "+---------------------------------------------------------------------------------------------------------------+\n";

    double sf = cardinality / 6001215.0;
    uint64_t cache_line_size = 128; // bytes
    uint64_t num_loads =  1478493 + 38854 + 2920374 + 1478870 + 6;
    uint64_t num_stores = 19;
    std::chrono::duration<double> duration(end - start);
    uint64_t tuples_per_second               = static_cast<uint64_t>(data_length / duration.count());
    double effective_memory_throughput       = static_cast<double>((tuples_per_second * size_per_tuple) / GIGA);
    double estimated_memory_throughput       = static_cast<double>((tuples_per_second * cache_line_size) / GIGA);
    double effective_memory_throughput_read  = static_cast<double>((tuples_per_second * size_per_tuple) / GIGA);
    double effective_memory_throughput_write = static_cast<double>(tuples_per_second / (size_per_tuple * GIGA));
    double theretical_memory_bandwidth       = static_cast<double>((5505 * 10e06 * (352 / 8) * 2) / 10e09);
    double efective_memory_bandwidth         = static_cast<double>(((data_length * sizeof(SHIPDATE_TYPE)) + (num_loads * size_per_tuple) + (num_loads * num_stores))  / (duration.count() * 10e09));
    double csv_time = std::chrono::duration<double>(end_csv - start_csv).count();
    double pre_process_time = std::chrono::duration<double>(end_preprocess - start_preprocess).count();
    
    std::cout << "\n+------------------------------------------------- Statistics --------------------------------------------------+\n";
    std::cout << "| TPC-H Q01 performance               : ="          << std::fixed 
              << tuples_per_second <<                 " [tuples/sec]" << std::endl;
    std::cout << "| Time taken                          : ~"          << std::setprecision(2)
              << duration.count() <<                  "  [s]"          << std::endl;
    std::cout << "| Estimated time for TPC-H SF100      : ~"          << std::setprecision(2)
              << duration.count() * (100 / sf) <<     "  [s]"          << std::endl;
    std::cout << "| CSV Time                            : ~"          << std::setprecision(2)
              <<  csv_time <<                         "  [s]"          << std::endl;
    std::cout << "| Preprocess Time                     : ~"          << std::setprecision(2)
              <<  pre_process_time <<                 "  [s]"          << std::endl;
    std::cout << "| Copy Time                           : ~"          << std::setprecision(2)
              << copy_time <<                         "  [s]"          << std::endl;
    std::cout << "| Computation Time                    : ~"          << std::setprecision(2)
              << computation_time <<                  "  [s]"          << std::endl;
    std::cout << "| Effective memory throughput (query) : ~"          << std::setprecision(2)
              << effective_memory_throughput <<       "  [GB/s]"       << std::endl;
    std::cout << "| Estimated memory throughput (query) : ~"          << std::setprecision(1)
              << estimated_memory_throughput <<       "  [GB/s]"       << std::endl;
    std::cout << "| Effective memory throughput (read)  : ~"          << std::setprecision(2)
              << effective_memory_throughput_read <<  "  [GB/s]"       << std::endl;
    std::cout << "| Memory throughput (write)           : ~"          << std::setprecision(2)
              << effective_memory_throughput_write << "  [GB/s]"       << std::endl;
    std::cout << "| Theoretical Bandwidth               : ="          << std::setprecision(1)
              << theretical_memory_bandwidth <<       " [GB/s]"       << std::endl;
    std::cout << "| Effective Bandwidth                 : ~"          << std::setprecision(2)
              << efective_memory_bandwidth <<         "  [GB/s]"       << std::endl;
    std::cout << "+---------------------------------------------------------------------------------------------------------------+\n";
}
