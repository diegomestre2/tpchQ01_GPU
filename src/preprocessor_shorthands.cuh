#ifndef PREPROCESSOR_SHORTHANDS_CUH_
#define PREPROCESSOR_SHORTHANDS_CUH_

#pragma once

#ifndef __fhd__
#ifdef __CUDACC__
#define __fhd__  __forceinline__ __host__ __device__
#define __fd__   __forceinline__ __device__
#else
#define __fhd__ inline
#define __fd__  inline
#endif
#endif

#endif // PREPROCESSOR_SHORTHANDS_CUH_
