#!/bin/bash

# build cuda-api-wrappers
cd cuda-api-wrappers
if [ ! -d "./$lib" ];
then
	cmake .
	make
fi
cd ..
# build project for debugging
rm -R debug
mkdir debug
cd debug
cmake -DCMAKE_BUILD_TYPE=Debug ..
make