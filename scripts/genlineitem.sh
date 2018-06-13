#!/bin/bash

scale_factor=${1:-1}
force_overwrite="-f"
generate_only_lineitem_table="-T L"
use_scale_factor="-s $scale_factor"
be_verbose="-v"
	# Generates TPC-H benchmark data:
	# -f    forcing overwrite
	# -T L  only the lineitem table
	# -s    total DBMS size in GB (the lineitem table is over half that)

[ -d tpch-dbgen ] || git clone https://github.com/eyalroz/tpch-dbgen || exit -1
cd tpch-dbgen
cmake .  \
	&& cmake --build . \
	&& echo "Generating lineitem table for TPC-H scale factor $scale_factor" \
	&& ./dbgen $be_verbose $force_overwrite $generate_only_lineitem_table $use_scale_factor \
	&& mv lineitem.tbl .. \
	&& exit 0
exit -1
