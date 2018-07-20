#!/usr/bin/python
#
# Eyal's modified version of Mark's bench.py
# Main changes:
# * All results are prefixed with other (CSV) fields, so that the lines from all files are compatible and can be concatenated together into a single global results file if you so with
# * Result timestamping
# * Calculates the means of runs into separate CSV files (which could still be concatenated with the per-run results; the means have an empty/null run index
# * Not trying kernel grid parameter combinations which result in too few blocks for the (first) GPU on the system
# * Compatibility with plot.R currently broken :-(

from __future__ import print_function
from shutil import copyfile
import datetime
import subprocess
import socket
import os

binary = "bin/tpch_01"
results_dir = "results"

default_sf = 100
default_options = "--apply-compression"
default_streams = 4
default_tuples_per_launch = 1 << 21
default_tuples_per_thread = 1024
default_threads_per_block = 256
default_placement = "local-mem"

streams = [4]
tuples_per_launch = [256*1024, 512*1024, 1024*1024, 2*1024*1024]
tuples_per_thread = [64, 128, 256, 512, 1024, 2048, 4096] # anything below 32 is probably kind of silly
threads_per_block = [32, 64, 128, 256, 512, 1024] # Note that some kernels do not supported the entire ranges, and need either many or not-too-many
placements = ["local-mem", "in-registers", "in-registers-per-thread", "global", "per-thread-shared-mem"]

options = [
	"",
	"--apply-compression",
	"--apply-compression --use-coprocessing",
	"--apply-compression --use-filter-pushdown",
	"--apply-compression --use-coprocessing --use-filter-pushdown",
]

# This next line is pretty fragile; if the binary changes even slightly, it won't work
cores_per_gpu = int((subprocess.check_output(['bash', '-c', "%s --device | grep 'Number of SM' | head -1 | cut -c18-" % (binary)])).rstrip())
min_keep_busy_factor = 2
hostname = (socket.gethostname())
now = datetime.datetime.now()

def mean_of_results_file(filename):
	with open(filename,'r') as f:
		data = [float(line.rstrip()) for line in f.readlines()] 
		f.close()
	return float(sum(data))/len(data) if len(data) > 0 else float('nan')

def syscall(cmd):
	print(cmd)
	os.system(cmd)

def run_test(filename_for_plot = None, raw_results_file = None, mean_results_file = None, sf = None, streams = None, tpls = None, vals = None, threads = None, placement = None, options = None):
	if not sf: sf = default_sf
	if not streams: streams = default_streams
	if not tpls: tpls = default_tuples_per_launch
	if not vals: vals = default_tuples_per_thread
	if not threads: threads = default_threads_per_block
	if not options: options = default_options
	if not placement: placement = default_placement

	if os.path.isfile(os.path.join(results_dir, filename_for_plot)):
		return

	syscall("""${BINARY} ${OPTIONS} --streams=${STREAMS} --scale-factor=${SF} --tuples-per-kernel-launch=${TUPLES} --tuples-per-thread=${VALUES} --threads-per-block=${THREADS} --hash-table-placement=${PLACEMENT}""".replace(
		"${BINARY}", binary).replace(
		"${OPTIONS}", options).replace(
		"${STREAMS}", str(streams)).replace(
		"${SF}", str(sf)).replace(
		"${TUPLES}", str(tpls)).replace(
		"${VALUES}", str(vals)).replace(
		"${THREADS}", str(threads)).replace(
		"${PLACEMENT}", placement))
	if not os.path.isfile('results.csv'):
		return

	full_plot_fn = os.path.join(results_dir, filename_for_plot)
	copyfile('results.csv', full_plot_fn)

	if raw_results_file:
		compressed_data = (options.find("apply-compression") != -1)
		filter_precomputation = (options.find("use-filter-pushdown") != -1)
		data_parallel_coprocessing = (options.find("use-coprocessing") != -1)
		csv_line_prefix_fields = '%s,%04u,%02u,%02u,%02u,%02u,%02u,%u,%u,%u,%u,%u,%s,%u,%u,%u' % ( hostname, now.year, now.month, now.day, now.hour, now.minute, now.second, sf, streams, tpls, vals, threads, placement, compressed_data, filter_precomputation, data_parallel_coprocessing)
		os.system("awk '{printf \"%s,%%d,%%s\\n\", NR, $0}' results.csv >> %s" % (csv_line_prefix_fields, raw_results_file))
	if mean_results_file:
		mean = mean_of_results_file('results.csv')
		with open(mean_results_file,'a') as f:
			f.write('%s,,%f\n' % (csv_line_prefix_fields,mean))
			f.close()
	os.remove('results.csv')

def init_results_files(basename):
	os.system('mkdir -p %s' % os.path.join(results_dir, basename))
	raw_results_filename = os.path.join(results_dir, '%s_raw.csv' % basename)
	mean_results_filename = os.path.join(results_dir, '%s.csv' % basename)
	result_csv_header_line='hostname,year,month,day,hour,minute,second,scale_factor,num_gpu_streams,tuples_per_kernel,tuples_per_thread,threads_per_block,hash_table_placement,compressed_data,filter_precomputation,data_parallel_coprocessing,run_index,time\n'
	with open(raw_results_filename, 'w') as f:
		f.write(result_csv_header_line)
		f.close()
	with open(mean_results_filename, 'w') as f:
		f.write(result_csv_header_line)
		f.close()


os.system('make')
#os.system('rm -rf %s' % results_dir)

options_basename='options'
init_results_files(options_basename)
raw_fn = os.path.join(results_dir, '%s_raw.csv' % options_basename)
mean_fn = os.path.join(results_dir, '%s.csv' % options_basename)
for opt in options:
	for p in placements:
		run_test(filename_for_plot = os.path.join(options_basename,'%s.csv' % p), raw_results_file = raw_fn, mean_results_file = mean_fn, options = opt, placement = p)

gp_basename='grid_params'
raw_fn = os.path.join(results_dir, '%s_raw.csv' % gp_basename)
mean_fn = os.path.join(results_dir, '%s.csv' % gp_basename)
init_results_files(gp_basename)
for vals in tuples_per_thread:
	for threads in threads_per_block:
		for p in placements:
			if default_tuples_per_launch >= vals * threads * cores_per_gpu * min_keep_busy_factor :
				run_test(raw_results_file = raw_fn, mean_results_file = mean_fn, filename_for_plot=os.path.join(gp_basename, 'vals_%s-threads_%s-placement_%s.csv' % (str(vals), str(threads), p)), vals = vals, threads = threads, placement = p)
			else:
				print ("%s tuples per thread and %s threads per block are too many for %s tuples per launch - there would not be enough blocks to keep the GPU busy" % (vals, threads, default_tuples_per_launch))
