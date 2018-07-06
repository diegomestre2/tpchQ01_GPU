#pragma once

#include "data_types.h"
#include <cuda/api/constants.h>
#include <climits>

namespace defaults {

enum {
    num_threads_per_block            = 256,
    num_tuples_per_thread            = 1024,
    num_gpu_streams                  = 4,   // And 2 might just be enough actually
    num_tuples_per_kernel_launch     = 1 << 20, // used for scheduling the kernel
    should_print_results             = false,
    apply_compression                = false,
        // should be true really, but then we'd need a turn-off switch for no compression
    num_query_execution_runs         = 5,
};

constexpr const double scale_factor   = 1.0;
constexpr const char kernel_variant[] = "in-registers";

static_assert(num_tuples_per_kernel_launch % num_threads_per_block == 0,
    "Please allot the same number of records to each thread");
static_assert(num_tuples_per_kernel_launch % cuda::warp_size == 0,
    "Please use a multiple of warp_size for num_tuples_per_kernel_launch");

#ifndef QUOTE
#define QUOTE(x) #x
#endif
#ifndef EXPAND_THEN_QUOTE
#define EXPAND_THEN_QUOTE(str) QUOTE(str)
#endif

constexpr const char tpch_data_subdirectory[]
                                      = EXPAND_THEN_QUOTE(DATA_FILES_DIR);

} // namespace defaults

constexpr const char lineitem_table_file_name[]
                                      = "lineitem.tbl";


enum {
    num_orders_at_scale_factor_1     = 1500000,
    cardinality_of_scale_factor_1    = 6001215,
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
