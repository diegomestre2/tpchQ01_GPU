#include <iostream>
#include "kernel.hpp"

int main(){
	std::cout << "TPC-H Query 1" << '\n';
	std::cout << std::endl;

    uint32_t key_cnt = 32;
	uint32_t warp_size = 32;
	uint32_t block_size = warp_size;
    uint32_t elements_per_thread = warp_size;
	uint32_t elements_per_block = block_size * elements_per_thread;
    uint32_t block_cnt = (key_cnt + elements_per_block - 1) / elements_per_block;
    std::cout << "launch params: <<<" << block_cnt << "," << block_size << ">>>" << std::endl;

    cudaDeviceSynchronize();
    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
    cuda::print<<<block_cnt, block_size>>>();
    cuda_check_error();
}