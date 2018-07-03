# TPC-H Query 01 Optimized for GPU execution
We hereby present the source code used to evaluate TPC-H Query 01 optimized for CPU-GPU co-processing. (Paper accepted to ADMS/VLDB 2018)

#### TPC-H Query 01 Versions:

| Implementation Flavor | Split CPU-GPU Computation | Filter Pushdown | Hash table Placement | Compression | Time (sec) |
| --------------------- | ------------------------- | --------------- | -------------------- | ----------- | ---------- |
| Global full           |             -             |         -       |        Global        |      -      |   12.60    |
| In-register full      |             -             |         -       |        Register      |      -      |   12.45    |
| Local full            |             -             |         -       |        Local         |      -      |   12.40    |
| Local fp small        |             -             |         √       |        Local         |      √      |    0.76    |
| In-register fp small  |             -             |         √       |        Register      |      √      |    0.76    |
| Global fp small       |             -             |         √       |        Global        |      √      |    0.76    |
| Global small          |             -             |         -       |        Global        |      √      |    0.74    |
| In-register small     |             -             |         -       |        Register      |      √      |    0.68    | 
| Local small           |             -             |         -       |        Local         |      √      |    0.57    | 
| Global sc small       |             √             |         -       |        Global        |      √      |    0.51    |
| In-register sc small  |             √             |         -       |        Register      |      √      |    0.43    | 
| Local sc small        |             √             |         -       |        Local         |      √      |    0.38    | 
| SharedMemory sc small |             √             |         -       |        Shared        |      √      |    0.37    |


## Prerequisites

- CUDA v9.0 or later is recommended.
- A C++14-capable compiler compatible with your version of CUDA; only GCC has been tested.
- [CMake](http://www.cmake.org/) v3.1 or later
- A Unix-like environment for the (simple) shell scripts; without it, you may need to perform a few tasks manually
- The [cuda-api-wrappers](https://github.com/eyalroz/cuda-api-wrappers) library; it can be obtained 

## Building

Assuming you've cloned into `/path/to/tpchQ1`:

- Configure the build and generate build files using `cmake /path/to/tpchQ1`
- Build using either your default make'ing tool or with `cmake --build /path/to/tpchQ1`; this will also generate input data for Scale Factor 1 (SF 1)
- Optionally: Generate data for more scale factors - SF 10 and SF 100 - using `make -C /path/to/tpchQ1 data_table_sf10` or `make -C /path/to/tpchQ1 data_table_sf100`.

## Running and generating data for other scale factors

- When building, the binary `tpch_01`  is generated; 
- Invoke `bin/tpch_01` in the directory where you've performed the build. That directory should have a `tpch/` subdirectory with  subdir for every scale factor with generated data.
- You can use `scripts/genlineitem.sh` to manually generate a table for arbitrary scale factors, but you must place it under your build dir, under `tpch/123.000000` (for generated scale factor 123). Non-integral scale factors should work as well.

### `tpch_01` command-line options

| Switch                  | Value range                                                          | Default value | Meaning                                                                                                                                                                                                |
|-------------------------|----------------------------------------------------------------------|---------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| --apply-compression     | N/A                                                                  | (off)        | Use the compression schemes described on the Wiki, to reduce the amount of data for transmission over PCI/e                                                                                            |
| --print-results         | N/A                                                                  | (off)         | Print the computed aggregates to std::cout after every run. Useful for debugging result stability issues.                                                                                              |
| --use-filter-pushdown   | N/A                                                                  | (off)         | Have the CPU check the TPC-H Q1 `WHERE` clause condition, passing only that result bit vector to the GPU. It's debatable whether this is actually a "push down"  in the traditional sense of the term. |
|  --use-coprocessing     | N/A                                                                  | (off)         | Schedule some of the work to be done on the CPU and some on the GPU                                                                                                                                    |
| --hash-table-placement  | in-registers, local-mem, per-thread-shared-mem, global               |  in-registers | Memory space + granularity for the aggregation tables; see the paper itself or the code for an explanation of what this means.                                                                         |
| --sf=                   | Integral or fractional number, limited precision                     | 1             | Which scale factor subdirectory to use (to look for the data table or cached column files). For sf 123.456789, data will be expected under `tpch/123.456789`                                           |
| --streams=              | Positive integral value                                              | 4             | The number of concurrent streams to use for scheduling GPU work. You should probably not change this.                                                                                                  |
| --threads-per-block=    | Positive integral number, preferably a multiple of 32                | 256           | Number of CUDA threads per block of a scheduled kernel of the computational work.                                                                                                                      |
| --tuples-per-thread=H   | Positive integral number, preferably high                            | 1024          | The number of tuples each thread processes individually before merging results with other threads                                                                                                      |
| --tuples-per-kernel=    | Positive integral number, preferably a multiple of threads-per-block | 1024          | Every how many input tuples is a new kernel launched?                                                                                                                                                  |


## Important Dates

- Paper Submission: Friday, 8 June, 2018 (Extended to 11 June, 2018)
- Notification of Acceptance: Friday, 29 June, 2018
- Camera-ready Submission: Friday, 20 July, 2018
- Workshop Date: Monday, 27 August, 2018


## What is TPC-H Query 1?

The query text and column information is [on the Wiki](https://github.com/diegomestre2/tpchQ01_GPU/wiki/TPCH-Query-1). For further information about the benchmark it is part of, see the [Transaction Processing Council](http://www.tpc.org/)'s  [page for TPC-H](http://www.tpc.org/tpch/default.asp)

