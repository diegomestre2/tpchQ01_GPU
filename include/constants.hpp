#pragma once

#define VALUES_PER_THREAD 16 // will probably not be used... although that's up to Diego
#define THREADS_PER_BLOCK 64

enum : record_count_t {
    records_per_scheduled_kernel = 1 << 20 // used for scheduling the kernel
};

enum { num_gpu_streams = 4 }; // And 2 might just be enough actually

enum {
    ship_date_frame_of_reference
                             = 727563,
    threshold_ship_date      = 729999 - ship_date_frame_of_reference,
    return_flag_support_size = 2,
    line_status_support_size = 3,
    num_potential_groups     = return_flag_support_size * line_status_support_size,
        // Note we will _not_ concatenate bits here - that would make for 8
        // potential groups, and we don't want that
    return_flag_bits         = 1,
    line_status_bits         = 2,
    return_flag_values_per_container = 32,
    line_status_values_per_container = 16,
};
