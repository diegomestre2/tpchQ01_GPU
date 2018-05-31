




all:
	./compile.sh

test: all
	./build/bin/tpch_01

clean:
	rm -rf build