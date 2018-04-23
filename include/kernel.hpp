#include <stdio.h>

namespace cuda{
  
  __global__
  void print() {

    // who am I?
    u32 wid = global_warp_id();
    u32 lid = warp_local_thread_id();
    printf("Global Warp: %zu Local Warp: %zu",wid, lid);

  }
}