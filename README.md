# TPC-H Query 01 Optimized for GPU execution
We hereby present the source code used to evaluate TPC-H Query 01 optimized for CPU-GPU co-processing. (Paper submitted to ADMS 2018)

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


## Requirements

- CUDA v9.0 or later is recommended.
- A C++11-capable compiler compatible with your version of CUDA.
- CMake v3.1 or later
- CUB Library (https://github.com/NVlabs/cub)
- CUDA_API_WRAPPERS Library (https://github.com/eyalroz/cuda-api-wrappers)

## How to Run

- Syncronize the project using sshfs in the bricks16
- Type command make all in the Terminal. It will do all the work.

#### Note: For testing purposes use the command make test

## Important Dates

- Paper Submission: Friday, 8 June, 2018 (Extended to 11 June, 2018)
- Notification of Acceptance: Friday, 29 June, 2018
- Camera-ready Submission: Friday, 20 July, 2018
- Workshop Date: Monday, 27 August, 2018

```sql
TPC-H Query 1
SELECT
    l_returnflag,
    l_linestatus,
    sum(l_quantity) as sum_qty,
    sum(l_extendedprice) as sum_base_price,
    sum(l_extendedprice * (1 - l_discount)) as sum_disc_price,
    sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) as sum_charge,
    avg(l_quantity) as avg_qty,
    avg(l_extendedprice) as avg_price,
    avg(l_discount) as avg_disc,
    count(*) as count_order
FROM
    lineitem
WHERE
    l_shipdate <= date '1998-12-01' - interval '90' day
GROUP BY
    l_returnflag,
    l_linestatus
ORDER BY
    l_returnflag,
    l_linestatus;
```
