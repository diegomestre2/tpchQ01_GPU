#include <iostream>
#include <cuda/api_wrappers.h>
#include <vector>

#include "kernel.hpp"
#include "../expl_comp_strat/tpch_kit.hpp"


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
	uint32_t block_size = 512;
    uint32_t elements_per_thread = warp_size;
	uint32_t elements_per_block = block_size * elements_per_thread;
    uint32_t block_cnt = (data_length + block_size - 1) / block_size;

    /* Launching Kernel */
    std::cout << "launch params: <<<" << block_cnt << "," << block_size << ">>>" << std::endl;
    cuda::naive_tpchQ01<<<block_cnt, block_size>>>(d_shipdate.get(), 
        d_discount.get(), d_extendedprice.get(), d_tax.get(), d_returnflag.get(), 
        d_linestatus.get(), d_quantity.get(), d_aggregations.get(), data_length);
    cudaDeviceSynchronize();
    cuda_check_error();

    /* Test */
    /*std::vector<int> in(64, 1); 
    int  *d_in, *d_out;
    cudaMalloc(&d_in,       64 * sizeof(int));
    cudaMalloc(&d_out,      64 * sizeof(int));
    cudaMemcpy(d_in,      &in[0],      64 * sizeof(int), cudaMemcpyHostToDevice);
    cuda::deviceReduceKernel<<<2,32>>>(d_in, d_out, 64);
    cuda::deviceReduceKernel<<<1,32>>>(d_out, d_out, 2);
    cudaFree(d_in);
    cudaFree(d_out);*/

}