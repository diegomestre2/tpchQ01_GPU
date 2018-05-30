#include <iostream>
#include <cuda/api_wrappers.h>
#include <vector>
#include <iomanip>

#include "kernel.hpp"

#include "../expl_comp_strat/tpch_kit.hpp"

using timer = std::chrono::high_resolution_clock;
#define streamSize (32 * 1024)

int main(){

    std::cout << "TPC-H Query 1" << '\n';
    get_device_properties();
    /* load data */
    lineitem li(7000000ull);
    li.FromFile("lineitem.tbl");
    kernel_prologue();
    const size_t data_length = cardinality;
    const int nStreams = static_cast<int>(data_length / streamSize);
    clear_tables();

    /* Allocate memory on device */
    auto current_device = cuda::device::current::get();
    auto d_shipdate      = cuda::memory::device::make_unique< int[]           >(current_device, data_length);
    auto d_discount      = cuda::memory::device::make_unique< int[]           >(current_device, data_length);
    auto d_extendedprice = cuda::memory::device::make_unique< int[]           >(current_device, data_length);
    auto d_tax           = cuda::memory::device::make_unique< int[]           >(current_device, data_length);
    auto d_quantity      = cuda::memory::device::make_unique< int[]           >(current_device, data_length);
    auto d_returnflag    = cuda::memory::device::make_unique< char[]          >(current_device, data_length);
    auto d_linestatus    = cuda::memory::device::make_unique< char[]          >(current_device, data_length);
    auto d_aggregations  = cuda::memory::device::make_unique< AggrHashTable[] >(current_device, MAX_GROUPS);

    auto size_char         = data_length * sizeof(char);
    auto size_aggregations = MAX_GROUPS  * sizeof(AggrHashTable);

    /* Transfer data to device */
    cudaStream_t streams[nStreams];
    auto streamBytes = streamSize * sizeof(int);
    auto stremBytes_char = streamSize * sizeof(char);
    for (int i = 0; i < nStreams; ++i) {
        int offset = i * streamSize;
        cudaStreamCreate(&streams[i]);

        if (offset + stremBytes_char > data_length) {
            stremBytes_char = data_length - offset;
            streamBytes = (data_length - offset) * sizeof(int);
        }
    
        cuda::memory::async::copy(d_shipdate.get()      + offset, shipdate      + offset, streamBytes,     streams[i]);
        cuda::memory::async::copy(d_discount.get()      + offset, discount      + offset, streamBytes,     streams[i]);
        cuda::memory::async::copy(d_extendedprice.get() + offset, extendedprice + offset, streamBytes,     streams[i]);
        cuda::memory::async::copy(d_tax.get()           + offset, tax           + offset, streamBytes,     streams[i]);
        cuda::memory::async::copy(d_quantity.get()      + offset, quantity      + offset, streamBytes,     streams[i]);
        cuda::memory::async::copy(d_returnflag.get()    + offset, returnflag    + offset, stremBytes_char, streams[i]);
        cuda::memory::async::copy(d_linestatus.get()    + offset, linestatus    + offset, stremBytes_char, streams[i]);
        //break;
    }
    cuda::memory::copy(d_aggregations.get(), aggrs0, size_aggregations);

    /* Setup to launch kernel */
    //uint32_t warp_size = 32;
    //uint32_t elements_per_thread = warp_size;
    uint32_t block_size = 128;
    //uint32_t elements_per_block = block_size * elements_per_thread;
    uint32_t block_cnt = ((data_length / 32) + block_size - 1) / block_size;
    int sharedBytes = 18 * sizeof(AggrHashTableKey);

    std::cout << "launch params: <<<" << (streamSize / block_size) << "," << block_size << ">>>" << std::endl;
    auto start = timer::now();
    for (int i = 0; i < nStreams; ++i) {
        auto num = streamSize;
        if 
        int offset = i * streamSize;


        
        cuda::thread_local_tpchQ01<<<(streamSize / (block_size * 32)), block_size, 0, streams[i]>>>(d_shipdate.get(), 
        d_discount.get(), d_extendedprice.get(), d_tax.get(), d_returnflag.get(), 
        d_linestatus.get(), d_quantity.get(), d_aggregations.get(), streamSize); 
        //break;
    }
    cudaDeviceSynchronize();
    cuda_check_error();
    auto end = timer::now();
    for (int i = 0; i < nStreams; ++i) {
        int offset = i * streamSize;
        //cudaMemcpyAsync(&a[offset], &d_a[offset], 
                          //streamBytes, cudaMemcpyDeviceToHost, streams[i]);
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
        //break;
    }


    /* Launching Kernel 
    std::cout << "launch params: <<<" << block_cnt << "," << block_size << ">>>" << std::endl;
    std::cout << todate_(2, 9, 1998) << std::endl;
        auto start = timer::now();
    cuda::thread_local_tpchQ01<<<block_cnt , block_size, sharedBytes>>>(d_shipdate.get(), 
        d_discount.get(), d_extendedprice.get(), d_tax.get(), d_returnflag.get(), 
        d_linestatus.get(), d_quantity.get(), d_aggregations.get(), data_length);
    cudaDeviceSynchronize();
    cuda_check_error();
    auto end = timer::now();
    cuda::memory::copy(aggrs0, d_aggregations.get(), size_aggregations);
*/
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
    std::cout << "\n+-------------------------------- Statistics -----------------------------------+\n";
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
    /*std::cout << std::endl;
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
    cudaFree(d_out);*/

}