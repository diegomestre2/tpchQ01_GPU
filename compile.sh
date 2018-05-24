#!/bin/bash


if grep -q "# -- CONFIGURATION -- #"  ~/.bashrc
then
	echo "Found Configuration";
else
	source configuration.sh
	cat configuration.sh >> ~/.bashrc;
fi

# build cuda-api-wrappers
cd cuda-api-wrappers
if [ ! -d "$lib" ] 
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