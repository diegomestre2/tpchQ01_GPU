# TPC-H Query 01 Optimized for GPU execution
One more way to run TPC-H Query 01...

#### Issues:

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
- Run the compile.sh script. It will do all the work.

#### Note: In case of error during the build of cuda-api_wrappers, the line include_directories(../cub/cub) should be included in its CMakeLists.txt file.

## Important Dates

- Paper Submission: Friday, 8 June, 2018
- Notification of Acceptance: Friday, 29 June, 2018
- Camera-ready Submission: Friday, 20 July, 2018
- Workshop Date: Monday, 27 August, 2018
