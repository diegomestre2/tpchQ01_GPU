#!/bin/bash
./dependencies.sh

# build project for debugging
mkdir -p build
cd build
if [ ! -d "debug" ]
then
	mkdir debug
	cd debug
	cmake -DCMAKE_BUILD_TYPE=Debug ../..
	cd ..
fi
cd ..
# build project 
cd build/debug
make -j 4
