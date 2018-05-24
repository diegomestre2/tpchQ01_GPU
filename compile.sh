#!/bin/bash

# build cuda-api-wrappers
cd cuda-api-wrappers
if [ ! -d "./$lib" ] 
then
	cmake .
	make
fi
cd ..
# build project 
rm -R build
mkdir build
cd build
cmake ..
make