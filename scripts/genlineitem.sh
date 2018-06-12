#!/usr/bin/bash
if [ ! -d tpch-dbgen ]
then
	git clone https://github.com/eyalroz/tpch-dbgen
fi
cd tpch-dbgen
cmake -G "Unix Makefiles" .
make
./dbgen -f -T L -s ${1:-1}
mv lineitem.tbl ..
