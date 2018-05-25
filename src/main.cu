#include <iostream>
#include <cuda/api_wrappers.h>
#include <vector>
#include <iomanip>

#include "kernel.hpp"
#include "../expl_comp_strat/tpch_kit.hpp"

using timer = std::chrono::high_resolution_clock;

int main(){

	std::cout << "TPC-H Query 1" << '\n';

    /* load data */
    lineitem li(7000000ull);
    li.FromFile("lineitem.tbl");

    const size_t data_length = li.l_extendedprice.cardinality;

    kernel_prologue();
    clear_tables();

    /* Allocate memory on device */
    auto current_device = cuda::device::current::get();
    auto d_shipdate      = cuda::memory::device::make_unique< int[]            >(current_device, data_length);
    auto d_discount      = cuda::memory::device::make_unique< int[]            >(current_device, data_length);
    auto d_extendedprice = cuda::memory::device::make_unique< int[]            >(current_device, data_length);
    auto d_tax           = cuda::memory::device::make_unique< int[]            >(current_device, data_length);
    auto d_quantity      = cuda::memory::device::make_unique< int[]            >(current_device, data_length);
    auto d_returnflag    = cuda::memory::device::make_unique< char[]           >(current_device, data_length);
    auto d_linestatus    = cuda::memory::device::make_unique< char[]           >(current_device, data_length);
    auto d_aggregations  = cuda::memory::device::make_unique< AggrHashTable[]  >(current_device, MAX_GROUPS);




    auto size_int          = data_length * sizeof(int);
    auto size_char         = data_length * sizeof(char);
    auto size_aggregations = MAX_GROUPS  * sizeof(AggrHashTable);

    auto start = timer::now();
    /* Transfer data to device */
    cuda::memory::copy(d_shipdate.get(),      shipdate,      size_int);
    cuda::memory::copy(d_discount.get(),      discount,      size_int);
    cuda::memory::copy(d_extendedprice.get(), extendedprice, size_int);
    cuda::memory::copy(d_tax.get(),           tax,           size_int);
    cuda::memory::copy(d_quantity.get(),      quantity,      size_int);
    cuda::memory::copy(d_returnflag.get(),    returnflag,    size_char);
    cuda::memory::copy(d_linestatus.get(),    linestatus,    size_char);
    cuda::memory::copy(d_aggregations.get(),  aggrs0,        size_aggregations);

    /* Setup to launch kernel */
    uint32_t warp_size = 32;
    uint32_t elements_per_thread = warp_size;
    uint32_t block_size = 1024 / elements_per_thread;
    uint32_t elements_per_block = block_size * elements_per_thread;
    uint32_t block_cnt = (data_length + block_size - 1) / block_size;

    /* Launching Kernel */
    //std::cout << "launch params: <<<" << block_cnt << "," << block_size << ">>>" << std::endl;
    cuda::naive_tpchQ01<<<block_cnt, block_size>>>(d_shipdate.get(), 
        d_discount.get(), d_extendedprice.get(), d_tax.get(), d_returnflag.get(), 
        d_linestatus.get(), d_quantity.get(), d_aggregations.get(), data_length);
    cudaDeviceSynchronize();
    cuda_check_error();
    auto end = timer::now();
    cuda::memory::copy(aggrs0, d_aggregations.get(), size_aggregations);

    auto print_dec = [] (auto s, auto x) { printf("%s%ld.%ld", s, Decimal64::GetInt(x), Decimal64::GetFrac(x)); };
    for (size_t group=0; group<MAX_GROUPS; group++) {
        if (aggrs0[group].count > 0) {
            char rf = group >> 8;
            char ls = group & std::numeric_limits<unsigned char>::max();

            size_t i = group;

            printf("# %c|%c", rf, ls);
            print_dec(" | ", aggrs0[i].sum_quantity);
            print_dec(" | ", aggrs0[i].sum_base_price);
            print_dec(" | ", aggrs0[i].sum_disc_price);
            print_dec(" | ", aggrs0[i].sum_charge);
            printf("|%ld\n", aggrs0[i].count);
        }
    }


    std::chrono::duration<double> duration(end - start);
    uint64_t tuples_per_second = static_cast<uint64_t>(data_length / duration.count());
    double effective_memory_throughput = tuples_per_second * sizeof(int) / 1024.0 / 1024.0 / 1024.0;
    double effective_memory_throughput_read = tuples_per_second * sizeof(int) / 1024.0 / 1024.0 / 1024.0;
    double effective_memory_throughput_write_output = tuples_per_second / 8.0 / 1024.0 / 1024.0 / 1024.0;
    std::cout << "\n+-------------------------------------------------------------------------------+\n";
    std::cout << "| TPC-H Q01 performance               : " << std::fixed << tuples_per_second << " [tuples/sec]" << std::endl;
    uint64_t cache_line_size = 128; // bytes
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
    /* Test */
    std::cout << std::endl;
    std::vector<int> in(64, 1);
    std::vector<int> res(64, 0);  
    int  *d_in, *d_out, *d_res;
    cudaMalloc(&d_in,       64 * sizeof(int));
    cudaMalloc(&d_out,      64 * sizeof(int));
    cudaMalloc(&d_res,      64 * sizeof(int));
    cudaMemcpy(d_in,      &in[0],      64 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_res,      &res[0],      64 * sizeof(int), cudaMemcpyHostToDevice);
    cuda::filter_k<<<2,32>>>(d_out,d_res, d_in, 64);
    cudaMemcpy(     &res[0], d_res,      64 * sizeof(int), cudaMemcpyDeviceToHost);
    for(auto &i : res)
            printf(" %d ", i);
    //cuda::deviceReduceKernel<<<1,32>>>(d_out, d_out, 2);
    cudaFree(d_in);
    cudaFree(d_out);

}