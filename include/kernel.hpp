#include <stdio.h>
#include "helper.hpp"
namespace cuda{
  
  __global__
  void print() {

    // who am I?
    uint32_t wid = global_warp_id();
    uint32_t lid = warp_local_thread_id();
    printf("Global Warp: %zu Local Warp: %zu",wid, lid);

  }
}