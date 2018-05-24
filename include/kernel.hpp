#include <stdio.h>
#include "helper.hpp"
#include "../expl_comp_strat/common.hpp"
namespace cuda{
  
	__global__
	void print() {
	  	// who am I?
	    int wid = global_warp_id();
	    int lid = warp_local_thread_id();
	    printf(" Global Warp: %d Local Warp: %d \n", wid, lid);

	}

	__inline__ __device__
	int warpReduceSum(int val) {
		for (int offset = warp_size / 2; offset > 0; offset /= 2) 
	    	val += __shfl_down(val, offset);
	   	return val;
	}

	__inline__ __device__
	int blockReduceSum(int val) {

	  	static __shared__ int shared[32]; // Shared mem for 32 partial sums
	  	int lane = warp_local_thread_id();
	  	int wid = block_local_warp_id();

	  	val = cuda::warpReduceSum(val);     // Each warp performs partial reduction

	  	if (lane == 0) shared[wid] = val; // Write reduced value to shared memory

	  	__syncthreads();              // Wait for all partial reductions

	  	//read from shared memory only if that warp existed
	  	val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

	  	if (wid==0) val = cuda::warpReduceSum(val); //Final reduce within first warp

	  	return val;
	}

	__global__ 
	void deviceReduceKernel(int *in, int* out, int N) {
	  	int sum = 0;
	  	//reduce multiple elements per thread
	  	for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
	       	i < N; 
	       	i += blockDim.x * gridDim.x) {
	    		sum += in[i];
	  	}
	  	printf("%d\n", sum);
	  	sum = cuda::blockReduceSum(sum);
	  	if (block_local_thread_id() == 0){
	  		printf("%d %d\n",blockIdx.x, sum);
	    	out[blockIdx.x] = sum;
	  	}
	}

	__global__
	void naive_tpchQ01(int *shipdate, int *discount, int *extendedprice, int *tax, 
		char *returnflag, char *linestatus, int *quantity, AggrHashTable *aggregations, size_t cardinality){

		int index = blockIdx.x * blockDim.x + threadIdx.x;
  		int stride = blockDim.x * gridDim.x;


	    for(size_t i = index ; i < cardinality; i += stride) {
	        if (shipdate[i] <= 19980902) {
	            const auto disc = discount[i];
	            const auto price = extendedprice[i];
	            const auto disc_1 = Decimal64::ToValue(1, 0) - disc;
	            const auto tax_1 = tax[i] + Decimal64::ToValue(1, 0);
	            const auto disc_price = Decimal64::Mul(disc_1, price);
	            const auto charge = Decimal64::Mul(disc_price, tax_1);
	            const idx_t idx = returnflag[i] << 8 | linestatus[i];
	            aggregations[idx].sum_quantity += quantity[i];
	            aggregations[idx].sum_base_price += price;
	            aggregations[idx].sum_disc_price = int128_add64(aggregations[idx].sum_disc_price, disc_price);
	            aggregations[idx].sum_charge = int128_add64(aggregations[idx].sum_charge, charge);
	            aggregations[idx].sum_disc += disc;
	            aggregations[idx].count++;
	        }
	    }

	}


}
