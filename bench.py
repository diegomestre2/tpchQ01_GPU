
import os

binary = "build/release/bin/tpch_01"

default_sf = 100
default_options = "--use-small-datatypes --use-coalescing"
default_streams = 8
default_tuples_per_stream = 131072
default_values_per_thread = 256
default_threads_per_block = 512

sfs = [1, 10, 100]
streams = [1, 2, 4, 8, 16, 32, 64, 128, 256]
tuples_per_stream = [1024, 2*1024, 4*1024, 8*1024, 16*1024, 32*1024, 64*1024, 128*1024, 256*1024, 512*1024, 1024*1024]
values_per_thread = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
threads_per_block = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

options = [
	"",
	"--use-coalescing",
	"--use-small-datatypes",
	"--use-global-ht",
	"--use-small-datatypes --use-global-ht",
	"--use-small-datatypes --use-coalescing",
	"--no-pinned-memory",
	"--no-pinned-memory --use-coalescing",
	"--no-pinned-memory --use-small-datatypes",
	"--no-pinned-memory --use-global-ht",
	"--no-pinned-memory --use-small-datatypes --use-global-ht",
	"--no-pinned-memory --use-small-datatypes --use-coalescing",
]

def syscall(cmd):
	print(cmd)
	os.system(cmd)

def run_test(fname = None, sf = None, streams = None, tpls = None, vals = None, threads = None, options = None):
	if not fname:
		raise Exception("No filename provided")
	if not sf: sf = default_sf
	if not streams: streams = default_streams
	if not tpls: tpls = default_tuples_per_stream
	if not vals: vals = default_values_per_thread
	if not threads: threads = default_threads_per_block
	if not options: options = default_options

	if os.path.isfile(os.path.join('results', fname)):
		return

	syscall("""${BINARY} ${OPTIONS} --streams=${STREAMS} --sf=${SF} --tuples-per-stream=${TUPLES} --values-per-thread=${VALUES} --threads-per-block=${THREADS}""".replace(
		"${BINARY}", binary).replace(
		"${OPTIONS}", options).replace(
		"${STREAMS}", str(streams)).replace(
		"${SF}", str(sf)).replace(
		"${TUPLES}", str(tpls)).replace(
		"${VALUES}", str(vals)).replace(
		"${THREADS}", str(threads)))
	os.system('mv results.csv %s' % os.path.join('results', fname))


os.system('make')
os.system('mkdir -p results')

os.system('mkdir -p results/options')
for opt in options:
	run_test(fname="options/results-%s.csv" % (opt.replace(" ", "")), options = opt)

os.system('mkdir -p results/sf')
for sf in sfs:
	run_test(fname="sf/results-sf%s.csv" % str(sf), sf=sf)

os.system('mkdir -p results/streams')
for stream in streams:
	for tpls in tuples_per_stream:
		run_test(fname="streams/results-streams%s-tuples%s.csv" % (str(stream), str(tpls)), streams=stream, tpls=tpls)

os.system('mkdir -p results/threads')
for vals in values_per_thread:
	for threads in threads_per_block:
		run_test(fname="threads/results-vals%s-threads%s.csv" % (str(vals), str(threads)), vals = vals, threads = threads)


