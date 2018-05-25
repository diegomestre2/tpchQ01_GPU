#!/bin/bash


if grep -q "# -- CONFIGURATION -- #"  ~/.bashrc
then
	echo "Found Configuration";
else
	source configuration.sh
	cat configuration.sh >> ~/.bashrc;
fi

if [ ! -d cub ]
then
	git clone https://github.com/NVlabs/cub
	cd cub
	git reset --hard c3cceac115c072fb63df1836ff46d8c60d9eb304
	cd ..
fi

if [ ! -d cuda-api-wrappers ]
then
	git clone https://github.com/eyalroz/cuda-api-wrappers
	cp -r cuda-api-wrappers/scripts .
fi

# build cuda-api-wrappers
cd cuda-api-wrappers
if [ ! -d "./lib" ] 
then
	cmake .
	make
fi
cd ..
# build project 
rm -Rf build
mkdir build
cd build
cmake ..
make
