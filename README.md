# TPC-H Query 01 Optimized for GPU execution
One more way to run TPC-H Query 01...

#### Issues:

- Modify Q01 for different Selectivities
- Compression scheme for the data
- CPU implementation
- GPU implementation

## Requirements

- CUDA v9.0 or later is recommended.
- A C++11-capable compiler compatible with your version of CUDA.
- CMake v3.1 or later
- CUB Library
- CUDA_API_WRAPPERS library

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
