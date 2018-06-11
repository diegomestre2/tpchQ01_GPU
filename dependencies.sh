#!/bin/bash

if grep -q "# -- CONFIGURATION -- #"  ~/.bashrc
then
	echo "Found Configuration in ~/.bashrc , not changing anything";
else
	echo "Applying configuration in configuration.sh (this won't be persistent!)"
	source configuration.sh
fi

if [ ! -d cuda-api-wrappers ]
then
	git clone https://github.com/eyalroz/cuda-api-wrappers
	cp -r cuda-api-wrappers/scripts .
fi

# build cuda-api-wrappers

if [ ! -d "cuda-api-wrappers/lib" ] 
then
	cd cuda-api-wrappers
	cmake .
	make
	cd ..
fi
