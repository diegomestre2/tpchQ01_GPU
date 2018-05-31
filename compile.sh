#!/bin/bash
./dependencies.sh
# build release project
mkdir -p build
cd build
if [ ! -d "release" ]
then
	mkdir release
	cd release
	cmake ../..
	cd ..
fi
cd ..
# build project 
cd build/release
make -j 4
