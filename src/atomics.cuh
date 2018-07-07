#pragma once
#ifndef ATOMICS_CUH_
#define ATOMICS_CUH_

#include "preprocessor_shorthands.cuh"
#include <cuda_runtime_api.h>

// Annoyingly, CUDA - upto and including version 9.2 - provides atomic
// operation wrappers for unsigned int and unsigned long long int, but
// not for the in-between type of unsigned long int. So - we
// have to make our own. Don't worry about the condition checks -
// they are optimized away at compile time and the result is basically
// a single PTX instruction (provided the value is available)
__fd__ unsigned long int atomicAdd(unsigned long int *address, unsigned long int val)
{
    if (sizeof (unsigned long int) == sizeof(unsigned long long int)) {
        return atomicAdd(reinterpret_cast<unsigned long long int*>(address), val);
    }
    else if (sizeof (unsigned long int) == sizeof(unsigned int)) {
        return  atomicAdd(reinterpret_cast<unsigned int*>(address), val);
    }
    else return 0;
}


#endif // #define ATOMICS_CUH_
