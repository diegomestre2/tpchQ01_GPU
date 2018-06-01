#include <iostream>
#include <cuda/api_wrappers.h>
#include <vector>
#include <iomanip>

#include "kernel.hpp"
#include "../expl_comp_strat/tpch_kit.hpp"

size_t magic_hash(char rf, char ls) {
    return (((rf - 'A')) - (ls - 'F'));
}

using timer = std::chrono::high_resolution_clock;

inline bool file_exists (const std::string& name) {
  struct stat buffer;
  return (stat (name.c_str(), &buffer) == 0);
}

int main(){
    if (!file_exists("lineitem.tbl")) {
        fprintf(stderr, "lineitem.tbl not found!\n");
        exit(1);
    }
    std::cout << "TPC-H Query 1" << '\n';
    get_device_properties();
    /* load data */
    SHIPDATE_TYPE* shipdate;
    DISCOUNT_TYPE* discount;
    EXTENDEDPRICE_TYPE* extendedprice;
    LINESTATUS_TYPE* linestatus;
    RETURNFLAG_TYPE* returnflag;
    QUANTITY_TYPE* quantity;
    TAX_TYPE* tax;
    size_t cardinality;
    {
        lineitem li(7000000ull);
        li.FromFile("lineitem.tbl");

        auto _shipdate = li.l_shipdate.get();
        auto _returnflag = li.l_returnflag.get();
        auto _linestatus = li.l_linestatus.get();
        auto _discount = li.l_discount.get();
        auto _tax = li.l_tax.get();
        auto _extendedprice = li.l_extendedprice.get();
        auto _quantity = li.l_quantity.get();
        cardinality = li.l_extendedprice.cardinality;

        shipdate = (SHIPDATE_TYPE*) malloc(sizeof(SHIPDATE_TYPE) * cardinality);
        discount = (DISCOUNT_TYPE*) malloc(sizeof(DISCOUNT_TYPE) * cardinality);
        extendedprice = (EXTENDEDPRICE_TYPE*) malloc(sizeof(EXTENDEDPRICE_TYPE) * cardinality);
        linestatus = (LINESTATUS_TYPE*) malloc(sizeof(LINESTATUS_TYPE) * cardinality);
        returnflag = (RETURNFLAG_TYPE*) malloc(sizeof(RETURNFLAG_TYPE) * cardinality);
        quantity = (QUANTITY_TYPE*) malloc(sizeof(QUANTITY_TYPE) * cardinality);
        tax = (TAX_TYPE*) malloc(sizeof(TAX_TYPE) * cardinality);
        printf("%d\n", todate_(01, 01,1992));
        for(size_t i = 0; i < cardinality; i++) {
            shipdate[i] = _shipdate[i] - SHIPDATE_MIN;
            discount[i] = _discount[i];
            extendedprice[i] = _extendedprice[i];
            linestatus[i] = _linestatus[i];
            returnflag[i] = _returnflag[i];
            quantity[i] = _quantity[i] / 100;
            tax[i] = _tax[i];

            assert((int)shipdate[i] == _shipdate[i] - SHIPDATE_MIN);
            assert((int64_t) discount[i] == _discount[i]);
            assert((int64_t) extendedprice[i] == _extendedprice[i]);
            assert((char) linestatus[i] == _linestatus[i]);
            assert((char) returnflag[i] == _returnflag[i]);
            assert((int64_t) quantity[i] == _quantity[i] / 100);
            assert((int64_t) tax[i] == _tax[i]);
        }

    }
    //kernel_prologue();

    assert(cardinality > 0 && "Prevent BS exception");
    const size_t data_length = cardinality;
    const int nStreams = static_cast<int>((data_length + TUPLES_PER_STREAM - 1) / TUPLES_PER_STREAM);
    clear_tables();

    /* Allocate memory on device */
    auto current_device = cuda::device::current::get();

    auto d_shipdate      = cuda::memory::device::make_unique< SHIPDATE_TYPE[]           >(current_device,data_length);
    auto d_discount      = cuda::memory::device::make_unique< DISCOUNT_TYPE[]       >(current_device,data_length);
    auto d_extendedprice = cuda::memory::device::make_unique< EXTENDEDPRICE_TYPE[]       >(current_device,data_length);
    auto d_tax           = cuda::memory::device::make_unique< TAX_TYPE[]       >(current_device,data_length);
    auto d_quantity      = cuda::memory::device::make_unique< QUANTITY_TYPE[]       >(current_device,data_length);
    auto d_returnflag    = cuda::memory::device::make_unique< RETURNFLAG_TYPE[]          >(current_device,data_length);
    auto d_linestatus    = cuda::memory::device::make_unique< LINESTATUS_TYPE[]          >(current_device,data_length);
    auto d_aggregations  = cuda::memory::device::make_unique< AggrHashTable[] >(current_device,MAX_GROUPS);

    cudaMemset(d_aggregations.get(), 0, sizeof(AggrHashTable)*MAX_GROUPS);

    /* Transfer data to device */
    cudaStream_t streams[nStreams];
    for (int i = 0; i < nStreams; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    cuda_check_error();
    auto start = timer::now();
    for (int i = 0; i < nStreams; ++i) {
        size_t offset = i * TUPLES_PER_STREAM;
        size_t size = std::min((size_t) TUPLES_PER_STREAM, (size_t) (data_length - offset));

        //std::cout << "Stream " << i << ": " << "[" << offset << " - " << offset + size << "]" << std::endl;

        cuda::memory::async::copy(d_shipdate.get()      + offset, shipdate      + offset, size * sizeof(SHIPDATE_TYPE),     streams[i]);
        cuda::memory::async::copy(d_discount.get()      + offset, discount      + offset, size * sizeof(DISCOUNT_TYPE), streams[i]);
        cuda::memory::async::copy(d_extendedprice.get() + offset, extendedprice + offset, size * sizeof(EXTENDEDPRICE_TYPE), streams[i]);
        cuda::memory::async::copy(d_tax.get()           + offset, tax           + offset, size * sizeof(TAX_TYPE), streams[i]);
        cuda::memory::async::copy(d_quantity.get()      + offset, quantity      + offset, size * sizeof(QUANTITY_TYPE), streams[i]);
        cuda::memory::async::copy(d_returnflag.get()    + offset, returnflag    + offset, size * sizeof(RETURNFLAG_TYPE),    streams[i]);
        cuda::memory::async::copy(d_linestatus.get()    + offset, linestatus    + offset, size * sizeof(LINESTATUS_TYPE),    streams[i]);
#if 1
    }

    for (int i = 0; i < nStreams; ++i) {
        size_t offset = i * TUPLES_PER_STREAM;
        assert(offset <= data_length);
        size_t size = std::min((size_t) TUPLES_PER_STREAM, (size_t) (data_length - offset));
#endif
        size_t amount_of_blocks = size / (VALUES_PER_THREAD * THREADS_PER_BLOCK) + 1;

        //std::cout << "Execution <<<" << amount_of_blocks << "," << THREADS_PER_BLOCK << ">>>" << std::endl;
        cuda::thread_local_tpchQ01<<<amount_of_blocks, THREADS_PER_BLOCK, 0, streams[i]>>>(
            d_shipdate.get() + offset,
            d_discount.get() + offset,
            d_extendedprice.get() + offset,
            d_tax.get() + offset,
            d_returnflag.get() + offset,
            d_linestatus.get() + offset,
            d_quantity.get() + offset,
            d_aggregations.get(),
            (u64_t) size);
    }

    cuda_check_error();
    for (int i = 0; i < nStreams; ++i) {
        //size_t offset = i * TUPLES_PER_STREAM;

        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    cuda::memory::copy(aggrs0, d_aggregations.get(), sizeof(AggrHashTable)*MAX_GROUPS);
    cudaDeviceSynchronize();
    auto end = timer::now();

    auto print_dec = [] (auto s, auto x) { printf("%s%ld.%ld", s, Decimal64::GetInt(x), Decimal64::GetFrac(x)); };
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

            printf("# %c|%c", rf, ls);
            print_dec(" | ", aggrs0[i].sum_quantity);
            print_dec(" | ", aggrs0[i].sum_base_price);
            print_dec(" | ", aggrs0[i].sum_disc_price);
            print_dec(" | ", aggrs0[i].sum_charge);
            printf("|%ld\n", aggrs0[i].count);
        }
    }

    double sf = cardinality / 6001215.0;

    std::chrono::duration<double> duration(end - start);
    uint64_t tuples_per_second = static_cast<uint64_t>(data_length / duration.count());
    auto size_per_tuple = sizeof(int) + sizeof(int64_t) * 4 + sizeof(char) * 2;
    double effective_memory_throughput = tuples_per_second * size_per_tuple / 1024.0 / 1024.0 / 1024.0;
    double effective_memory_throughput_read = tuples_per_second * sizeof(int) / 1024.0 / 1024.0 / 1024.0;
    double effective_memory_throughput_write_output = tuples_per_second / 8.0 / 1024.0 / 1024.0 / 1024.0;
    std::cout << "\n+-------------------------------- Statistics -----------------------------------+\n";
    std::cout << "| TPC-H Q01 performance               : " << std::fixed << tuples_per_second << " [tuples/sec]" << std::endl;
    uint64_t cache_line_size = 128; // bytes

    

    std::cout << "| Time taken                          : ~" << std::setprecision(2)
              << duration.count() << " [s]" << std::endl;
    std::cout << "| Estimated time for TPC-H SF100      : ~" << std::setprecision(2)
              << duration.count() * (100 / sf) << " [s]" << std::endl;
    std::cout << "| Effective memory throughput (query) : ~" << std::setprecision(2)
              << effective_memory_throughput << " [GB/s]" << std::endl;
    std::cout << "| Estimated memory throughput (query) : ~" << std::setprecision(2)
              << (tuples_per_second * cache_line_size / 1024.0 / 1024.0 / 1024.0) << " [GB/s]" << std::endl;
    std::cout << "| Effective memory throughput (read)  : ~" << std::setprecision(2)
              << effective_memory_throughput_read << " [GB/s]" << std::endl;
    std::cout << "| Memory throughput (write)           : ~" << std::setprecision(2)
              << effective_memory_throughput_write_output << " [GB/s]" << std::endl;
    std::cout << "| Theoretical Bandwidth [GB/s]        : " << (5505 * 10e06 * (352 / 8) * 2) / 10e09 << std::endl;
    std::cout << "| Effective Bandwidth [GB/s]          : " << data_length * 25 * 13 / duration.count() << std::endl;
    std::cout << "+-------------------------------------------------------------------------------+\n";
}