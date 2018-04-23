#!/bin/bash
  
if [[ -z "${CUDA_API_WRAPPERS_DIR}" ]]; then
  echo "Exporting ENV VARIABLE FOR CUDA_API_WRAPPERS..."
  export CUDA_API_WRAPPERS_DIR=/export/scratch1/home/tome/Volume/mnt_mac/cuda-api-wrappers/
fi
rm -R build
mkdir build
cd build
cmake ..
make