


build/release/bin/tpch_01:
	./compile.sh

build/debug/bin/tpch_01:
	./debug.sh

all: build/debug/bin/tpch_01 build/release/bin/tpch_01

test: all
	./build/release/bin/tpch_01

clean:
	rm -rf build

debug: build/debug/bin/tpch_01
	./debug.sh
	gdb -ex run --args ./build/debug/bin/tpch_01
