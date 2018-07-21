/*
 * This file contains more general-purpose code,
 * which is inspecific to the TPC-H Q1 experiments,
 * and didn't fit elsewhere
 */
#pragma once
#include <cuda_runtime.h>
#include <cuda/api_wrappers.h>
#include <iostream>
#include <string>
#include <cassert>
#include <unistd.h>
#include <sstream>
#include <iomanip>
#include <ctime>
#include <chrono>

void get_device_properties(){
    int32_t device_cnt = 0;
    cudaGetDeviceCount(&device_cnt);
    cudaDeviceProp device_prop;

    for (int i = 0; i < device_cnt; i++) {
        cudaGetDeviceProperties(&device_prop, i);
        std::cout << "+---------------------------------------------------------------------------------------------------------------+\n";
        printf("| Device id: %d\t", i);
        printf("  Device name: %s\t",                   device_prop.name);
        printf("  Compute capability: %d.%d\n",         device_prop.major, device_prop.minor);
        std::cout << std::endl;
        printf("| Memory Clock Rate [KHz]: %d\n",       device_prop.memoryClockRate);
        printf("| Memory Bus Width [bits]: %d\n",       device_prop.memoryBusWidth);
        printf("| Peak Memory Bandwidth [GB/s]: %f\n",  2.0*device_prop.memoryClockRate*(device_prop.memoryBusWidth/8)/1.0e6);
        printf("| L2 size [KB]: %d\n",                  device_prop.l2CacheSize/1024);
        printf("| Shared Memory per Block [KB]: %lu\n", device_prop.sharedMemPerBlock/1024);
        std::cout << std::endl;
        printf("| Number of SMs: %d\n",                 device_prop.multiProcessorCount);
        printf("| Max. number of threads per SM: %d\n", device_prop.maxThreadsPerMultiProcessor);
        printf("| Concurrent kernels: %d\n",            device_prop.concurrentKernels);
        printf("| WarpSize: %d\n",                      device_prop.warpSize);
        printf("| MaxThreadsPerBlock: %d\n",            device_prop.maxThreadsPerBlock);
        printf("| MaxThreadsDim[0]: %d\n",              device_prop.maxThreadsDim[0]);
        printf("| MaxGridSize[0]: %d\n",                device_prop.maxGridSize[0]);
        printf("| PageableMemoryAccess: %d\n",          device_prop.pageableMemoryAccess);
        printf("| ConcurrentManagedAccess: %d\n",       device_prop.concurrentManagedAccess);
        printf("| Number of async. engines: %d\n",      device_prop.asyncEngineCount);
        std::cout << "+---------------------------------------------------------------------------------------------------------------+\n";
    }
}

template <typename F, typename... Args>
void for_each_argument(F f, Args&&... args) {
    [](...){}((f(std::forward<Args>(args)), 0)...);
}

void make_sure_we_are_on_cpu_core_0()
{
#if 0
    // CPU affinities are devil's work
    // Make sure we are on core 0
    // TODO: Why not in a function?
    cpu_set_t cpuset;

    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    sched_setaffinity(0, sizeof(cpuset), &cpuset);
#endif
}

std::pair<std::string,std::string> split_once(std::string delimited, char delimiter) {
    auto pos = delimited.find_first_of(delimiter);
    return { delimited.substr(0, pos), delimited.substr(pos+1) };
}

// Would be nice to avoid actually declaring the following 3 types and just using plain aggregates

template <typename T>
using plugged_unique_ptr = std::unique_ptr<T>;

template <typename T>
using plain_ptr = std::conditional_t<std::is_array<T>::value, std::decay_t<T>, std::decay_t<T>*>;

inline void assert_always(bool a) {
    assert(a);
    if (!a) {
        fprintf(stderr, "Assert always failed!");
        exit(EXIT_FAILURE);
    }
}

// Note: This will force casts to int. It's not a problem
// the way our code is written, but otherwise it needs to be generalized
constexpr inline int div_rounding_up(const int& dividend, const int& divisor)
{
    // This is not the fastest implementation, but it's safe, in that there's never overflow
#if __cplusplus >= 201402L
    std::div_t div_result = std::div(dividend, divisor);
    return div_result.quot + !(!div_result.rem);
#else
    // Hopefully the compiler will optimize the two calls away.
    return std::div(dividend, divisor).quot + !(!std::div(dividend, divisor).rem);
#endif
}

std::string host_name()
{
    enum { max_len = 1023 };
    char buffer[max_len + 1];
    gethostname(buffer, max_len);
    buffer[max_len] = '\0'; // to be on the safe side
    return { buffer };
}

std::string timestamp()
{
    auto now = std::chrono::system_clock::now();
    auto now_c = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&now_c), "%F %T");
    return ss.str();
}
