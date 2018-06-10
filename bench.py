
import os

binary = "build/release/bin/tpch_01"

default_sf = 100
default_options = "--apply-compression"
default_streams = 8
default_tuples_per_launch = 131072
default_tuples_per_thread = 256
default_threads_per_block = 512

sfs = [1, 10, 100]
streams = [1, 2, 3, 4, 6, 8, 16, 32]
tuples_per_launch = [1024, 16*1024, 64*1024, 128*1024, 256*1024, 512*1024, 1024*1024, 2*1024*1024]
tuples_per_thread = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096] # anything below 32 is probably kind of silly  
threads_per_block = [32, 64, 128, 256, 512, 1024] # Note that some kernels do not supported the entire ranges, and need either many or not-too-many

options = [
	"",
	"--apply-compression",
	"--use-global-hash-table",
	"--apply-compression --use-global-hash-table"
]

def syscall(cmd):
	print(cmd)
	os.system(cmd)

def run_test(fname = None, sf = None, streams = None, tpls = None, vals = None, threads = None, options = None):
	if not fname:
		raise Exception("No filename provided")
	if not sf: sf = default_sf
	if not streams: streams = default_streams
	if not tpls: tpls = default_tuples_per_launch
	if not vals: vals = default_tuples_per_thread
	if not threads: threads = default_threads_per_block
	if not options: options = default_options

	if os.path.isfile(os.path.join('results', fname)):
		return

	syscall("""${BINARY} ${OPTIONS} --streams=${STREAMS} --scale-factor=${SF} --tuples-per-launch=${TUPLES} --tuples-per-thread=${VALUES} --threads-per-block=${THREADS}""".replace(
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
	for tpls in tuples_per_launch:
		run_test(fname="streams/results-streams%s-tuples%s.csv" % (str(stream), str(tpls)), streams=stream, tpls=tpls)

os.system('mkdir -p results/threads')
for vals in tuples_per_thread:
	for threads in threads_per_block:
		run_test(fname="threads/results-vals%s-threads%s.csv" % (str(vals), str(threads)), vals = vals, threads = threads)


