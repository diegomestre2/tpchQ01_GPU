#include <iostream>
#include <cuda/api_wrappers.h>
#include <vector>
#include <iomanip>

#include "kernel.hpp"
#include "kernels/naive.hpp"
#include "kernels/local.hpp"
#include "kernels/global.hpp"
#include "kernels/coalesced.hpp"
#include "../expl_comp_strat/tpch_kit.hpp"

size_t magic_hash(char rf, char ls) {
    return (((rf - 'A')) - (ls - 'F'));
}

void assert_always(bool a) {
    if (!a) {
        fprintf(stderr, "Assert always failed!");
        exit(1);
    }
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
    auto _shipdate                       = ptrfunc< SHIPDATE_TYPE[]             >(cardinality);           \
    auto _discount                       = ptrfunc< DISCOUNT_TYPE[]             >(cardinality);           \
    auto _extendedprice                  = ptrfunc< EXTENDEDPRICE_TYPE[]        >(cardinality);           \
    auto _tax                            = ptrfunc< TAX_TYPE[]                  >(cardinality);           \
    auto _quantity                       = ptrfunc< QUANTITY_TYPE[]             >(cardinality);           \
    auto _returnflag                     = ptrfunc< RETURNFLAG_TYPE[]           >(cardinality);           \
    auto _linestatus                     = ptrfunc< LINESTATUS_TYPE[]           >(cardinality);           \
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
    auto _shipdate_small                 = ptrfunc< SHIPDATE_TYPE_SMALL[]       >(cardinality);           \
    auto _discount_small                 = ptrfunc< DISCOUNT_TYPE_SMALL[]       >(cardinality);           \
    auto _extendedprice_small            = ptrfunc< EXTENDEDPRICE_TYPE_SMALL[]  >(cardinality);           \
    auto _tax_small                      = ptrfunc< TAX_TYPE_SMALL[]            >(cardinality);           \
    auto _quantity_small                 = ptrfunc< QUANTITY_TYPE_SMALL[]       >(cardinality);           \
    auto _returnflag_small               = ptrfunc< RETURNFLAG_TYPE_SMALL[]     >((cardinality + 3) / 4); \
    auto _linestatus_small               = ptrfunc< LINESTATUS_TYPE_SMALL[]     >((cardinality + 7) / 8); \
    shipdate_small = _shipdate_small.get(); \
    discount_small = _discount_small.get(); \
    extendedprice_small = _extendedprice_small.get(); \
    tax_small = _tax_small.get(); \
    quantity_small = _quantity_small.get(); \
    returnflag_small = _returnflag_small.get(); \
    linestatus_small = _linestatus_small.get(); \
    _shipdate_small.release(); \
    _discount_small.release(); \
    _extendedprice_small.release(); \
    _tax_small.release(); \
    _quantity_small.release(); \
    _returnflag_small.release(); \
    _linestatus_small.release(); \
}

int main(int argc, char** argv) {
    std::cout << "TPC-H Query 1" << '\n';
    get_device_properties();
    /* load data */
    auto start_csv = timer::now();
    size_t cardinality;

    std::string input_file = "lineitem.tbl";

    bool USE_PINNED_MEMORY = true;
    bool USE_GLOBAL_HT = false;
    bool USE_SMALL_DATATYPES = false;

    std::string input_argument = "--use-input-file=";
    for(int i = 0; i < argc; i++) {
        auto arg = std::string(argv[i]);
        if (arg == "--no-pinned-memory") {
            USE_PINNED_MEMORY = false;
        } else if (arg == "--use-global-ht") {
            USE_GLOBAL_HT = true;
        } else if (arg == "--use-small-datatypes") {
            USE_SMALL_DATATYPES = true;
        } else if (arg.substr(0, input_argument.size()) == input_argument) {
            input_file = arg.substr(input_argument.size());
        }
    }

    if (!file_exists(input_file.c_str())) {
        fprintf(stderr, "lineitem.tbl not found!\n");
        exit(1);
    }

    lineitem li(7000000ull);
    li.FromFile(input_file.c_str());
    auto end_csv = timer::now();
    kernel_prologue();

    auto size_per_tuple = sizeof(SHIPDATE_TYPE) + sizeof(DISCOUNT_TYPE) + sizeof(EXTENDEDPRICE_TYPE) + sizeof(TAX_TYPE) + sizeof(QUANTITY_TYPE) + sizeof(RETURNFLAG_TYPE) + sizeof(LINESTATUS_TYPE);
    if (USE_SMALL_DATATYPES) {
        size_per_tuple = sizeof(SHIPDATE_TYPE_SMALL) + sizeof(DISCOUNT_TYPE_SMALL) + sizeof(EXTENDEDPRICE_TYPE_SMALL) + sizeof(TAX_TYPE_SMALL) + sizeof(QUANTITY_TYPE_SMALL) + sizeof(RETURNFLAG_TYPE_SMALL) + sizeof(LINESTATUS_TYPE_SMALL);
    }
    auto start_preprocess = timer::now();

    SHIPDATE_TYPE* shipdate;
    DISCOUNT_TYPE* discount;
    EXTENDEDPRICE_TYPE* extendedprice;
    TAX_TYPE* tax;
    QUANTITY_TYPE* quantity;
    RETURNFLAG_TYPE* returnflag;
    LINESTATUS_TYPE* linestatus;

    SHIPDATE_TYPE_SMALL* shipdate_small;
    DISCOUNT_TYPE_SMALL* discount_small;
    EXTENDEDPRICE_TYPE_SMALL* extendedprice_small;
    TAX_TYPE_SMALL* tax_small;
    QUANTITY_TYPE_SMALL* quantity_small;
    RETURNFLAG_TYPE_SMALL* returnflag_small;
    LINESTATUS_TYPE_SMALL* linestatus_small;
    if (USE_PINNED_MEMORY) {
        INITIALIZE_MEMORY(cuda::memory::host::make_unique);
    } else {
        INITIALIZE_MEMORY(std::make_unique);
    }

    for(size_t i = 0; i < cardinality; i++) {
        shipdate[i] = _shipdate[i];
        discount[i] = _discount[i];
        extendedprice[i] = _extendedprice[i];
        quantity[i] = _quantity[i];
        tax[i] = _tax[i];
        returnflag[i] = _returnflag[i];
        linestatus[i] = _linestatus[i];

        shipdate_small[i]      = shipdate[i] - SHIPDATE_MIN;
        discount_small[i]      = discount[i];
        extendedprice_small[i] = extendedprice[i];
        quantity_small[i]      = quantity[i] / 100;
        tax_small[i]           = tax[i];
        if (i % 4 == 0) {
            returnflag_small[i / 4] = 0;
            for(size_t j = 0; j < std::min((size_t) 4, cardinality - i); j++) {
                // 'N' = 0x00, 'R' = 0x01, 'A' = 0x10
                returnflag_small[i / 4] |= 
                    (_returnflag[i + j] == 'N' ? 0x00 : (_returnflag[i + j] == 'R' ? 0x01 : 0x02)) << (j * 2);
            }
        }
        if (i % 8 == 0) {
            linestatus_small[i / 8] = 0;
            for(size_t j = 0; j < std::min((size_t) 8, cardinality - i); j++) {
                // 'O' = 0, 'F' = 1
                linestatus_small[i / 8] |= (_linestatus[i + j] == 'F' ? 1 : 0) << j;
            }
        }

        assert_always((int)shipdate_small[i]           == shipdate[i] - SHIPDATE_MIN);
        assert_always((int64_t) discount_small[i]      == discount[i]);
        assert_always((int64_t) extendedprice_small[i] == extendedprice[i]);
        assert_always((int64_t) quantity_small[i]      == quantity[i] / 100);
        assert_always((int64_t) tax_small[i]           == tax[i]);
    }

    constexpr uint8_t RETURNFLAG_MASK[] = { 0x03, 0x0C, 0x30, 0xC0 };
    constexpr uint8_t LINESTATUS_MASK[] = { 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80 };
    for(size_t i = 0; i < cardinality; i++) {
        uint8_t retflag = (returnflag_small[i / 4] & RETURNFLAG_MASK[i % 4]) >> (2 * (i % 4));
        uint8_t lstatus = (linestatus_small[i / 8] & LINESTATUS_MASK[i % 8]) >> (i % 8);
        assert_always(retflag == (returnflag[i] == 'N' ? 0x00 : (returnflag[i] == 'R' ? 0x01 : 0x02)));
        assert_always(lstatus == (linestatus[i] == 'F' ? 1 : 0));
    }
    auto end_preprocess = timer::now();

    assert(cardinality > 0 && "Prevent BS exception");
    const size_t data_length = cardinality;
    const int nStreams = static_cast<int>((data_length + TUPLES_PER_STREAM - 1) / TUPLES_PER_STREAM);
    clear_tables();

    /* Allocate memory on device */
    auto current_device = cuda::device::current::get();

    auto d_shipdate                 = cuda::memory::device::make_unique< SHIPDATE_TYPE[]      >(current_device, data_length);
    auto d_discount                 = cuda::memory::device::make_unique< DISCOUNT_TYPE[]      >(current_device, data_length);
    auto d_extendedprice            = cuda::memory::device::make_unique< EXTENDEDPRICE_TYPE[] >(current_device, data_length);
    auto d_tax                      = cuda::memory::device::make_unique< TAX_TYPE[]           >(current_device, data_length);
    auto d_quantity                 = cuda::memory::device::make_unique< QUANTITY_TYPE[]      >(current_device, data_length);
    auto d_returnflag               = cuda::memory::device::make_unique< RETURNFLAG_TYPE[]    >(current_device, data_length);
    auto d_linestatus               = cuda::memory::device::make_unique< LINESTATUS_TYPE[]    >(current_device, data_length);


    auto d_shipdate_small           = cuda::memory::device::make_unique< SHIPDATE_TYPE_SMALL[]      >(current_device, data_length);
    auto d_discount_small           = cuda::memory::device::make_unique< DISCOUNT_TYPE_SMALL[]      >(current_device, data_length);
    auto d_extendedprice_small      = cuda::memory::device::make_unique< EXTENDEDPRICE_TYPE_SMALL[] >(current_device, data_length);
    auto d_tax_small                = cuda::memory::device::make_unique< TAX_TYPE_SMALL[]           >(current_device, data_length);
    auto d_quantity_small           = cuda::memory::device::make_unique< QUANTITY_TYPE_SMALL[]      >(current_device, data_length);
    auto d_returnflag_small         = cuda::memory::device::make_unique< RETURNFLAG_TYPE_SMALL[]    >(current_device, (data_length + 3) / 4);
    auto d_linestatus_small         = cuda::memory::device::make_unique< LINESTATUS_TYPE_SMALL[]    >(current_device, (data_length + 7) / 8);

    auto d_aggregations             = cuda::memory::device::make_unique< AggrHashTable[]      >(current_device, MAX_GROUPS);

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
        if (USE_SMALL_DATATYPES) {
            cuda::memory::async::copy(d_shipdate_small.get()                 + offset, shipdate_small                     + offset, size * sizeof(SHIPDATE_TYPE_SMALL),     streams[i]);
            cuda::memory::async::copy(d_discount_small.get()                 + offset, discount_small                     + offset, size * sizeof(DISCOUNT_TYPE_SMALL), streams[i]);
            cuda::memory::async::copy(d_extendedprice_small.get()            + offset, extendedprice_small                + offset, size * sizeof(EXTENDEDPRICE_TYPE_SMALL), streams[i]);
            cuda::memory::async::copy(d_tax_small.get()                      + offset, tax_small                          + offset, size * sizeof(TAX_TYPE_SMALL), streams[i]);
            cuda::memory::async::copy(d_quantity_small.get()                 + offset, quantity_small                     + offset, size * sizeof(QUANTITY_TYPE_SMALL), streams[i]);
            cuda::memory::async::copy(d_returnflag_small.get()               + (offset / 4), returnflag_small             + (offset / 4), (size * sizeof(RETURNFLAG_TYPE_SMALL) + 3) / 4,    streams[i]);
            cuda::memory::async::copy(d_linestatus_small.get()               + (offset / 8), linestatus_small             + (offset / 8), (size * sizeof(LINESTATUS_TYPE_SMALL) + 7) / 8,    streams[i]);
        } else {
            cuda::memory::async::copy(d_shipdate.get()      + offset, shipdate      + offset, size * sizeof(SHIPDATE_TYPE),     streams[i]);
            cuda::memory::async::copy(d_discount.get()      + offset, discount      + offset, size * sizeof(DISCOUNT_TYPE), streams[i]);
            cuda::memory::async::copy(d_extendedprice.get() + offset, extendedprice + offset, size * sizeof(EXTENDEDPRICE_TYPE), streams[i]);
            cuda::memory::async::copy(d_tax.get()           + offset, tax           + offset, size * sizeof(TAX_TYPE), streams[i]);
            cuda::memory::async::copy(d_quantity.get()      + offset, quantity      + offset, size * sizeof(QUANTITY_TYPE), streams[i]);
            cuda::memory::async::copy(d_returnflag.get()    + offset, returnflag    + offset, size * sizeof(RETURNFLAG_TYPE),    streams[i]);
            cuda::memory::async::copy(d_linestatus.get()    + offset, linestatus    + offset, size * sizeof(LINESTATUS_TYPE), streams[i]);
        }
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
        if (!USE_GLOBAL_HT) {
            if (USE_SMALL_DATATYPES) {
                cuda::thread_local_tpchQ01_small_datatypes<<<amount_of_blocks, THREADS_PER_BLOCK, SHARED_MEMORY, streams[i]>>>(
                    d_shipdate_small.get()                 + offset,
                    d_discount_small.get()                 + offset,
                    d_extendedprice_small.get()            + offset,
                    d_tax_small.get()                      + offset,
                    d_returnflag_small.get()               + offset / 4,
                    d_linestatus_small.get()               + offset / 8,
                    d_quantity_small.get()                 + offset,
                    d_aggregations.get(),
                    (u64_t) size);
            } else {
                cuda::thread_local_tpchQ01<<<amount_of_blocks, THREADS_PER_BLOCK, SHARED_MEMORY, streams[i]>>>(
                    d_shipdate.get()                 + offset,
                    d_discount.get()                 + offset,
                    d_extendedprice.get()            + offset,
                    d_tax.get()                      + offset,
                    d_returnflag.get()               + offset,
                    d_linestatus.get()               + offset,
                    d_quantity.get()                 + offset,
                    d_aggregations.get(),
                    (u64_t) size);
            }
        } else {
            if (USE_SMALL_DATATYPES) {
                cuda::global_ht_tpchQ01_small_datatypes<<<amount_of_blocks, THREADS_PER_BLOCK, SHARED_MEMORY, streams[i]>>>(
                    d_shipdate_small.get()                 + offset,
                    d_discount_small.get()                 + offset,
                    d_extendedprice_small.get()            + offset,
                    d_tax_small.get()                      + offset,
                    d_returnflag_small.get()               + offset / 4,
                    d_linestatus_small.get()               + offset / 8,
                    d_quantity_small.get()                 + offset,
                    d_aggregations.get(),
                    (u64_t) size);
            } else {
                cuda::global_ht_tpchQ01<<<amount_of_blocks, THREADS_PER_BLOCK, SHARED_MEMORY, streams[i]>>>(
                    d_shipdate.get()                 + offset,
                    d_discount.get()                 + offset,
                    d_extendedprice.get()            + offset,
                    d_tax.get()                      + offset,
                    d_returnflag.get()               + offset,
                    d_linestatus.get()               + offset,
                    d_quantity.get()                 + offset,
                    d_aggregations.get(),
                    (u64_t) size);
            }
        }
        //std::cout << "Execution <<<" << amount_of_blocks << "," << THREADS_PER_BLOCK << "," << SHARED_MEMORY << ">>>" << std::endl;
        
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
    // A/F - N/F - N/O, R/F
    int group_order[4];
    if (USE_SMALL_DATATYPES) {
        group_order[0] = 6;
        group_order[1] = 4;
        group_order[2] = 0;
        group_order[3] = 5;
    } else {
        group_order[0] = magic_hash('A', 'F');
        group_order[1] = magic_hash('N', 'F');
        group_order[2] = magic_hash('N', 'O');
        group_order[3] = magic_hash('R', 'F');
    }
    for (size_t idx=0; idx < 4; idx++) {
        int group = group_order[idx];
        if (aggrs0[group].count > 0) {
            size_t i = group;
            char rf = '-', ls = '-';
            if (idx == 0) { // A, F = 2 + 4
                rf = 'A';
                ls = 'F';
                if (cardinality == 6001215) {
                    assert(aggrs0[i].sum_quantity == 3773410700);
                    assert(aggrs0[i].count == 1478493);
                }
            } else if (idx == 1) { // N, F = 0 + 4
                rf = 'N';
                ls = 'F';
                if (cardinality == 6001215) {
                    assert(aggrs0[i].sum_quantity == 99141700);
                    assert(aggrs0[i].count == 38854);
                }
            } else if (idx == 2) { // N, O = 0 + 0
                rf = 'N';
                ls = 'O';
                if (cardinality == 6001215) {
                    assert(aggrs0[i].sum_quantity == 7447604000);
                    assert(aggrs0[i].count == 2920374);
                }
            } else if (idx == 3) { // R, F = 1 + 4
                rf = 'R';
                ls = 'F';
                if (cardinality == 6001215) {
                    assert(aggrs0[i].sum_quantity == 3771975300);
                    assert(aggrs0[i].count == 1478870);
                }
            } else {
                printf("%d\n", group);
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
