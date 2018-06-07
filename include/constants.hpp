#pragma once

#include "data_types.h"
#include <climits>


#define MAX_TUPLES_PER_STREAM MIN_TUPLES_PER_STREAM
#define MIN_TUPLES_PER_STREAM (32*1024)
#define VALUES_PER_THREAD 64

#define THREADS_PER_BLOCK 32

enum : cardinality_t {
    records_per_scheduled_kernel = 1 << 20 // used for scheduling the kernel
};

enum { num_gpu_streams = 4 }; // And 2 might just be enough actually

enum {
    ship_date_frame_of_reference     = 727563, // which date value is this?
    threshold_ship_date              = 729999, // todate_(2, 9, 1998)
    compressed_threshold_ship_date   = threshold_ship_date - ship_date_frame_of_reference,
    return_flag_support_size         = 3,
    line_status_support_size         = 2,
    num_potential_groups             = return_flag_support_size * line_status_support_size,
        // Note we will _not_ concatenate bits here - that would make for 8
        // potential groups, and we don't want that
    return_flag_bits                 = 2,
    line_status_bits                 = 1,
    log_return_flag_bits             = 1,
    log_line_status_bits             = 0,
    bits_per_bit_container           = sizeof(bit_container_t) * CHAR_BIT,
    return_flag_values_per_container = bits_per_bit_container / return_flag_bits,
    line_status_values_per_container = bits_per_bit_container / line_status_bits,
};

static_assert(return_flag_values_per_container * return_flag_bits == bits_per_bit_container,
	"return flags must fill a bit container perfectly");
static_assert(line_status_values_per_container * line_status_bits == bits_per_bit_container,
	"line stati must fill a bit container perfectly");
