#!/usr/bin/python
#
# Eyal's modified version of Mark's bench.py - get benchmark timings
#
# Use this script after you've discovered the optimal grid parameters for your machine and GPU,
# to get the final end-to-end, normal-execution timings for all execution options

from __future__ import print_function
from shutil import copyfile
import datetime
import subprocess
import socket
import os
import csv
import sys
import StringIO

binary = 'bin/tpch_01'
results_dir = 'results'

tuples_per_thread = [32, 64, 128, 256, 512, 1024] # anything below 32 is probably kind of silly
threads_per_block = [32, 64, 128, 160, 256, 512] # Note that some kernels do not supported the entire ranges, and need either many or not-too-many
placements = ['shared_mem_per_thread', 'local_mem', 'in_registers', 'in_registers_per_thread', 'global']

options = [
#	  apply compression  filter precomputation  use coprocessing
#     -----------------  ---------------------  ----------------
	[ False,             False,                 False             ],
	[ True ,             False,                 False             ],
	[ True ,             False,                 True              ],
	[ True ,             True,                  False             ],
	[ True ,             True,                  True              ]
]

# Key fields are:
#        compression  filter precompute, placement
#
# Data is based on results from bricks16 in the SciLens cluster at CWI - but make your own!
optimal_grid_params = { 
	(False,       False, 'shared_mem_per_thread'  ): { 'tuples_per_thread': 512, 'threads_per_block': 32},
	(True,        False, 'shared_mem_per_thread'  ): { 'tuples_per_thread': 512, 'threads_per_block': 32},
	(True,        True,  'shared_mem_per_thread'  ): { 'tuples_per_thread': 128, 'threads_per_block': 64},
	(False,       False, 'local_mem'              ): { 'tuples_per_thread': 256, 'threads_per_block': 256},
	(True,        False, 'local_mem'              ): { 'tuples_per_thread': 128, 'threads_per_block': 256},
	(True,        True,  'local_mem'              ): { 'tuples_per_thread': 128, 'threads_per_block': 256},
	(False,       False, 'in_registers'           ): { 'tuples_per_thread': 512, 'threads_per_block': 64},
	(True,        False, 'in_registers'           ): { 'tuples_per_thread': 512, 'threads_per_block': 32},
	(True,        True,  'in_registers'           ): { 'tuples_per_thread': 512, 'threads_per_block': 64},
	(False,       False, 'in_registers_per_thread'): { 'tuples_per_thread': 512, 'threads_per_block': 64},
	(True,        False, 'in_registers_per_thread'): { 'tuples_per_thread': 512, 'threads_per_block': 128},
	(True,        True,  'in_registers_per_thread'): { 'tuples_per_thread': 512, 'threads_per_block': 32},
	(False,       False, 'global'                 ): { 'tuples_per_thread': 512, 'threads_per_block': 32},
	(True,        False, 'global'                 ): { 'tuples_per_thread': 512, 'threads_per_block': 32},
	(True,        True,  'global'                 ): { 'tuples_per_thread': 512, 'threads_per_block': 64}
}

default_sf = 100
default_options = [ False, False, False ]
default_streams = 4
default_tuples_per_launch = 1 << 22
default_tuples_per_thread = 128
default_threads_per_block = 256
default_placement = 'local_mem'
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

def run_test(raw_results_file = None, mean_results_file = None, sf = None, streams = None, tpls = None, vals = None, threads = None, placement = None, options = None, runs = None):
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

	args = [ binary ]
	if compressed_data:
		args.append('--apply-compression') 
	if filter_precomputation:
		args.append('--use-filter-pushdown')
	if data_parallel_coprocessing:
		args.append('--use-coprocessing')
	args.extend([
		'--streams=1'                             , # serialize everything for testing the kernels proper,
		'--runs=%u'                     % runs    , # Do we need multiple runs at all? probably not
		'--scale-factor=%f'             % sf      ,
		'--tuples-per-kernel-launch=%u' % tpls    ,
		'--tuples-per-thread=%u'        % vals    ,
		'--threads-per-block=%u'        % threads ,
		'--hash-table-placement=%s'     % p
		])

# Initialize these if you want empty-result lines for failed runs
#	(total_time_of_launches, num_launches, average_execution_time, min_execution_time, max_execution_time) = ('', '', '', '', '')
	print(' '.join(args))
	try:
		subprocess.check_output(args)
	except:
		print('Could not complete execution')
		pass

	if not os.path.isfile('results.csv'):
		print('No results.csv generated')
		return

	if raw_results_file:
		csv_line_prefix = '%s,%04u,%02u,%02u,%02u,%02u,%02u,%u,%u,%u,%u,%u,%u,%s,%u,%u,%u' % ( hostname, now.year, now.month, now.day, now.hour, now.minute, now.second, sf, streams, tpls, vals, threads, runs, placement, compressed_data, filter_precomputation, data_parallel_coprocessing)
		os.system("awk '{printf \"%s,%%d,%%s\\n\", NR, $0}' results.csv >> %s" % (csv_line_prefix, raw_results_file))

	if mean_results_file:
		mean = mean_of_results_file('results.csv')
		with open(mean_results_file,'a') as f:
			f.write('%s,,%s\n' % (csv_line_prefix,mean))
			f.close()


def init_results_file(results_filename):
	if not os.path.isdir(results_dir):
		os.makedirs(results_dir)
	result_csv_header_line = 'hostname,year,month,day,hour,minute,second,scale_factor,num_gpu_streams,tuples_per_kernel,tuples_per_thread,threads_per_block,num_runs,hash_table_placement,compressed_data,filter_precomputation,data_parallel_coprocessing,run_index,execution_time\n'
	with open(results_filename, 'w') as f:
		f.write(result_csv_header_line)
		f.close()




subprocess.check_output(['make', binary])

results_basename='proper_runs'
mean_fn = os.path.join(results_dir, '%s_mean.csv' % results_basename)
raw_fn = os.path.join(results_dir, '%s_raw.csv' % results_basename)
init_results_file(raw_fn)
init_results_file(mean_fn)

for opt in options:
	for p in placements:
		apply_compression = opt[0]
		precompute_filter = opt[1]
		params = optimal_grid_params[(apply_compression, precompute_filter, p)]
		run_test(
			raw_results_file = raw_fn, mean_results_file = mean_fn, 
			sf = default_sf, streams = default_streams, 
			tpls = default_tuples_per_launch,
			vals = params['tuples_per_thread'],
			threads = params['threads_per_block'],
			placement = p, options = opt, runs = default_num_runs)

