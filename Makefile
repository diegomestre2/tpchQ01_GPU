

all:
	./compile.sh

test:
	./compile.sh
	./build/release/bin/tpch_01

clean:
	rm -rf build

debug:
	./debug.sh
	gdb -ex run --args ./build/debug/bin/tpch_01
