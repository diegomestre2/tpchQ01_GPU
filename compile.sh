#!/bin/bash
./dependencies.sh
# build release project
mkdir -p build
cd build
if [ ! -d "release" ]
then
	mkdir release
	cd release
	cmake -DCMAKE_BUILD_TYPE=Release ../..
	cd ..
fi
cd ..
# build project 
cd build/release
make
