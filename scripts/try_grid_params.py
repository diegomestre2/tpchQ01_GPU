#!/usr/bin/python
#
# Eyal's modified version of Mark's bench.py - grid params
#
# * TODO: We must switch this to running nvprof with kernel serialization and 1 stream only, and obtaining the total kernel time (and perhaps launch count)

from __future__ import print_function
from shutil import copyfile
import datetime
import subprocess
import socket
import os
import csv
import sys
import time
import StringIO

binary = "bin/tpch_01"
results_dir = "results"

tuples_per_thread = [32, 64, 128, 256, 512, 1024] # anything below 32 is probably kind of silly, and even 64 has not been optimal in any case
threads_per_block = [32, 64, 128, 160, 256, 512, 1024] # Note that some kernels do not supported the entire ranges, and need either many or not-too-many
placements = ["shared_mem_per_thread", "local_mem", "in_registers", "in_registers_per_thread", "global"]

options = [
#	  apply compression  filter precomputation  use coprocessing
#     -----------------  ---------------------  ----------------
	[ False,             False,                 False             ],
	[ True ,             False,                 False             ],
	[ True ,             False,                 True              ], 
	[ True ,             True,                  False             ],
	[ True ,             True,                  True              ], 
]

default_sf = 100
default_options = [ False, False, False ]
default_streams = 4
default_tuples_per_launch = 1 << 22
default_tuples_per_thread = 128
default_threads_per_block = 256
default_placement = "local_mem"
default_num_runs = 5

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

def check_for_interrupt():
	try:
		time.sleep(0.5)
	except KeyboardInterrupt:
#		print "Interrupted"
		sys.exit()

def run_test(results_file = None, sf = None, streams = None, tpls = None, vals = None, threads = None, placement = None, options = None, runs = None):
	if not runs : runs = default_num_runs
	if not placement: placement = default_placement
	if not sf: sf = default_sf
	if not streams: streams = default_streams
	if not tpls: tpls = default_tuples_per_launch
	if not vals: vals = default_tuples_per_thread
	if not threads: threads = default_threads_per_block
	if not options and options != '': options = default_options
	(compressed_data, filter_precomputation, data_parallel_coprocessing ) = options

	if tpls < vals * threads * cores_per_gpu * min_keep_busy_factor :
		print ("%s tuples per thread and %s threads per block are too many for %s tuples per launch - there would not be enough blocks to keep the GPU busy" % (vals, threads, tpls))
		return

	args = [ 'scripts/nvprof_get_kernel_summary' ]
	args.extend([
		'--runs=1'                                ,
		'--streams=1'                             , # serialize everything for testing the kernels proper,
		'--scale-factor=%f'             % sf      ,
		'--tuples-per-kernel-launch=%u' % tpls    ,
		'--tuples-per-thread=%u'        % vals    ,
		'--threads-per-block=%u'        % threads ,
		'--hash-table-placement=%s'     % p
		])
	if compressed_data:
		args.append('--apply-compression')
	if filter_precomputation:
		args.append('--use-filter-pushdown')
	if data_parallel_coprocessing:
		args.append('--use-coprocessing')

	total_time_of_launches, num_launches, average_execution_time, min_execution_time, max_execution_time = ('', '', '', '', '')
	if not data_parallel_coprocessing:
		print(' '.join(args))
		try:
			nvprof_output = subprocess.check_output(args)
			if nvprof_output.count('\n') != 1:
				print('Run failed; nvprof had %u output lines.' % nvprof_output.count('\n'), file=sys.stderr)
			else:
				as_io = StringIO.StringIO(nvprof_output)
				parsed = csv.reader(as_io)
				single_row = parsed.next()
				(activity_type, percent_of_total_time, total_time_of_launches, num_launches, average_execution_time, min_execution_time, max_execution_time, name) = list(single_row)
		except:
#		print('nvprof wrapper script returned non-zero', file=sys.stderr)
			check_for_interrupt()

	args[0] = binary # as opposed to the nvprof wrapper
	args[1] = \
		'--runs=%u'                     % runs
	args[2] = \
		'--streams=%u'                  % streams

	print(' '.join(args))
	try:
		subprocess.check_output(args)
	except:
		print('Could not complete regular Q1 execution')
		check_for_interrupt()
		return

	if not os.path.isfile('results.csv'):
		print('No results.csv generated')
		return

	if results_file:
		csv_line_prefix = '%s,%04u,%02u,%02u,%02u,%02u,%02u,%u,%u,%u,%u,%u,%u,%s,%u,%u,%u' % ( hostname, now.year, now.month, now.day, now.hour, now.minute, now.second, sf, streams, tpls, vals, threads, runs, placement, compressed_data, filter_precomputation, data_parallel_coprocessing)
		csv_line_nvprof_values = '%s,%s,%s,%s,%s' % (total_time_of_launches, num_launches, average_execution_time, min_execution_time, max_execution_time)
		mean = mean_of_results_file('results.csv')
		with open(results_file,'a') as f:
			f.write('%s,,%s,%s\n' % (csv_line_prefix, csv_line_nvprof_values, mean))
			f.close()
		# The raw results will go into the same file, but have non-empty run index; field layout is the same
		os.system("awk '{printf \"%s,%%d,,,,,,%%s\\n\", NR, $0}' results.csv >> %s" % (csv_line_prefix, results_file))

def init_results_file(filename):
	if not os.path.isdir(results_dir):
		os.makedirs(results_dir)
	result_csv_header_line='hostname,year,month,day,hour,minute,second,scale_factor,num_gpu_streams,tuples_per_kernel,tuples_per_thread,threads_per_block,num_runs,hash_table_placement,compressed_data,filter_precomputation,data_parallel_coprocessing,total_execution_time_of_kernels,num_kernel_launches,average_kernel_execution_time,min_kernel_execution_time,max_kernel_execution_time,overall_execution_time\n'
	with open(filename, 'w') as f:
		f.write(result_csv_header_line)
		f.close()

os.system('make $binary')

gp_basename='grid_params'
results_fn = os.path.join(results_dir, '%s.csv' % gp_basename)
init_results_file(results_fn)
for opt in options:
	for p in placements:
		for vals in tuples_per_thread:
			for threads in threads_per_block:
					run_test(results_file = results_fn, vals = vals, threads = threads, placement = p, options = opt)
