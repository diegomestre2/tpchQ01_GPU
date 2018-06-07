#include <iostream>
#include <cuda/api_wrappers.h>
#include <vector>
#include <iomanip>

using record_count_t = uint32_t;

using bit_container_t             = cuda::native_word_t;
static_assert(std::is_same<bit_container_t,uint32_t>{}, "Expecting the bit container to hold 32 bits");
using compressed_date_t           = uint16_t;
using compressed_discount_t       = uint8_t;
using compressed_tax_t            = uint8_t;
using compressed_extended_price_t = uint32_t;

// Most of these happen to be the same, but that is, in a sense, arbitrary;
// they could just as well have had different types.
using sum_quantity_t              = uint64_t;
using sum_base_price_t            = uint64_t;
using sum_discounted_price_t      = uint64_t;
using sum_charge_t                = uint64_t;
using sum_discount_t              = uint64_t;
using record_count_t              = uint32_t;

#include "kernel.hpp"
#include "../expl_comp_strat/tpch_kit.hpp"

#define GIGA (1024 * 1024 * 1024)
#define MEGA (1024 * 1024)
#define KILO (1024)

using timer = std::chrono::high_resolution_clock;

inline bool file_exists (const std::string& name) {
  struct stat buffer;
  return (stat (name.c_str(), &buffer) == 0);
}

#define INITIALIZE_MEMORY(ptrfunc) { \
    auto _shipdate      = ptrfunc< ship_date_t[]      >(cardinality); \
    auto _discount      = ptrfunc< discount_t[]       >(cardinality); \
    auto _extendedprice = ptrfunc< extended_price_t[] >(cardinality); \
    auto _tax           = ptrfunc< tax_t[]            >(cardinality); \
    auto _quantity      = ptrfunc< quantity_t[]       >(cardinality); \
    auto _returnflag    = ptrfunc< return_flag_t[]    >(cardinality); \
    auto _linestatus    = ptrfunc< line_status_t[]    >(cardinality); \
    shipdate = _shipdate.get(); \
    discount = _discount.get(); \
    extendedprice = _extendedprice.get(); \
    tax = _tax.get(); \
    quantity = _quantity.get(); \
    returnflag = _returnflag.get(); \
    linestatus = _linestatus.get(); \
    _shipdate.release(); \
    _discount.release(); \
    _extendedprice.release(); \
    _tax.release(); \
    _quantity.release(); \
    _returnflag.release(); \
    _linestatus.release(); \
}

// Note: This will force casts to int. It's not a problem
// the way our code is written, but otherwise it needs to be generalized
constexpr inline int div_rounding_up(const int& dividend, const int& divisor)
{
    // This is not the fastest implementation, but it's safe, in that there's never overflow
#if __cplusplus >= 201402L
    std::div_t div_result = std::div(dividend, divisor);
    return div_result.quot + !(!div_result.rem);
#else
    // Hopefully the compiler will optimize the two calls away.
    return std::div(dividend, divisor).quot + !(!std::div(dividend, divisor).rem);
#endif
}

int main(int argc, char** argv) {
    if (!file_exists("lineitem.tbl")) {
        fprintf(stderr, "lineitem.tbl not found!\n");
        exit(1);
    }
    std::cout << "TPC-H Query 1" << '\n';
    get_device_properties();
    /* load data */
    auto start_csv = timer::now();
    size_t cardinality;
    lineitem li(7000000ull);
    li.FromFile("lineitem.tbl");
    auto end_csv = timer::now();
    kernel_prologue();

    bool USE_PINNED_MEMORY = true;

    for(int i = 0; i < argc; i++) {
        auto arg = std::string(argv[i]);
        if (arg == "--no-pinned-memory") {
            USE_PINNED_MEMORY = false;
        }
    }

    auto start_preprocess = timer::now();

    ship_date_t* shipdate;
    discount_t* discount;
    extended_price_t* extendedprice;
    tax_t* tax;
    quantity_t* quantity;
    return_flag_t* returnflag;
    line_status_t* linestatus;
    if (USE_PINNED_MEMORY) {
        INITIALIZE_MEMORY(cuda::memory::host::make_unique);
    } else {
        INITIALIZE_MEMORY(std::make_unique);
    }

    for(size_t i = 0; i < cardinality; i++) {
        shipdate[i]      = _shipdate[i] - SHIPDATE_MIN;
        discount[i]      = _discount[i];
        extendedprice[i] = _extendedprice[i];
        linestatus[i]    = _linestatus[i];
        returnflag[i]    = _returnflag[i];
        quantity[i]      = _quantity[i] / 100;
        tax[i]           = _tax[i];

        assert((int)shipdate[i]           == _shipdate[i] - SHIPDATE_MIN);
        assert((int64_t) discount[i]      == _discount[i]);
        assert((int64_t) extendedprice[i] == _extendedprice[i]);
        assert((char) linestatus[i]       == _linestatus[i]);
        assert((char) returnflag[i]       == _returnflag[i]);
        assert((int64_t) quantity[i]      == _quantity[i] / 100);
        assert((int64_t) tax[i]           == _tax[i]);
    }
    auto end_preprocess = timer::now();

    assert(cardinality > 0 && "Prevent BS exception");
    const size_t data_length = cardinality;

    // Note:
    // We are not timing the host-side allocations here. In a real DBMS, these will likely only be
    // a few sub-allocations, which would take very little time (dozens of clock cycles overall) -
    // no system calls.

    struct {
        std::unique_ptr<sum_quantity_t[]        > sum_quantity;
        std::unique_ptr<sum_base_price_t[]      > sum_base_price;
        std::unique_ptr<sum_discounted_price_t[]> sum_discounted_price;
        std::unique_ptr<sum_charge_t[]          > sum_charge;
        std::unique_ptr<sum_discount_t[]        > sum_discount;
        std::unique_ptr<record_count_t[]        > record_count;
        // Why aren't we computing these?
        // struct {
        //     std::unique_ptr<avg_quantity_t[]        > avg_quantity;
        //     std::unique_ptr<avg_extended_price_t[]  > avg_extended_price;
        //     std::unique_ptr<avg_discount_t[]        > avg_discount;
        // } derived;
    } aggregates_on_host = {
        std::make_unique< sum_quantity_t[]         >(num_potential_groups),
        std::make_unique< sum_base_price_t[]       >(num_potential_groups),
        std::make_unique< sum_discounted_price_t[] >(num_potential_groups),
        std::make_unique< sum_charge_t []          >(num_potential_groups),
        std::make_unique< sum_discount_t[]         >(num_potential_groups),
        std::make_unique< record_count_t[]         >(num_potential_groups)
        // ,
        // {
        //      std::make_unique< avg_quantity_t[]         >(num_potential_groups),
        //      std::make_unique< avg_extended_price_t[]   >(num_potential_groups),
        //      std::make_unique< avg_discount_t[]         >(num_potential_groups),
        // }
    };
    // Note:


    /* Allocate memory on device */

    // Note:
    // We are not timing the allocations here. In a real DBMS, actual CUDA allocations would
    // happen with the DBMS is brought up, and when a query is processed, it will only be
    // a few sub-allocations, which would take very little time (dozens of clock cycles overall) -
    // no CUDA API nor system calls. We _will_, however, time the initialization of the buffers.

    auto cuda_device = cuda::device::current::get();

    struct {
        cuda::memory::device::unique_ptr< sum_quantity_t[]         > sum_quantity;
        cuda::memory::device::unique_ptr< sum_base_price_t[]       > sum_base_price;
        cuda::memory::device::unique_ptr< sum_discounted_price_t[] > sum_discounted_price;
        cuda::memory::device::unique_ptr< sum_charge_t[]           > sum_charge;
        cuda::memory::device::unique_ptr< sum_discount_t[]         > sum_discount;
        cuda::memory::device::unique_ptr< record_count_t[]         > record_count;
    } aggregates_on_device = {
        cuda::memory::device::make_unique< sum_quantity_t[]         >(cuda_device, num_potential_groups),
        cuda::memory::device::make_unique< sum_base_price_t[]       >(cuda_device, num_potential_groups),
        cuda::memory::device::make_unique< sum_discounted_price_t[] >(cuda_device, num_potential_groups),
        cuda::memory::device::make_unique< sum_charge_t []          >(cuda_device, num_potential_groups),
        cuda::memory::device::make_unique< sum_discount_t[]         >(cuda_device, num_potential_groups),
        cuda::memory::device::make_unique< record_count_t[]         >(cuda_device, num_potential_groups)
    };

    struct stream_input_buffer_set {
        cuda::memory::device::unique_ptr< ship_date_t[]      > shipdate;
        cuda::memory::device::unique_ptr< discount_t[]       > discount;
        cuda::memory::device::unique_ptr< extended_price_t[] > extendedprice;
        cuda::memory::device::unique_ptr< tax_t[]            > tax;
        cuda::memory::device::unique_ptr< quantity_t[]       > quantity;
        cuda::memory::device::unique_ptr< bit_container_t[]  > returnflag;
        cuda::memory::device::unique_ptr< bit_container_t[]  > linestatus;
    };

    std::vector<stream_input_buffer_set> stream_input_buffer_sets;
    std::vector<cuda::stream_t<>> streams;
	stream_input_buffer_sets.reserve(num_gpu_streams);
	streams.reserve(num_gpu_streams);
        // We'll be scheduling (most of) our work in a round-robin fashion on all of
        // the streams, to prevent the GPU from idling.


    for (int i = 0; i < num_gpu_streams; ++i) {
        auto input_buffers = stream_input_buffer_set{
            cuda::memory::device::make_unique< ship_date_t[]      >(cuda_device, records_per_scheduled_kernel),
            cuda::memory::device::make_unique< discount_t[]       >(cuda_device, records_per_scheduled_kernel),
            cuda::memory::device::make_unique< extended_price_t[] >(cuda_device, records_per_scheduled_kernel),
            cuda::memory::device::make_unique< tax_t[]            >(cuda_device, records_per_scheduled_kernel),
            cuda::memory::device::make_unique< quantity_t[]       >(cuda_device, records_per_scheduled_kernel),
            cuda::memory::device::make_unique< bit_container_t[]  >(cuda_device, div_rounding_up(records_per_scheduled_kernel, return_flag_values_per_container)),
            cuda::memory::device::make_unique< bit_container_t[]  >(cuda_device, div_rounding_up(records_per_scheduled_kernel, line_status_values_per_container))
        };
        stream_input_buffer_sets.emplace_back(std::move(input_buffers));
        auto stream = cuda_device.create_stream(cuda::stream::async);
        streams.emplace_back(std::move(stream));
    }

    double copy_time = 0;
    double computation_time = 0;
    auto start = timer::now();

    // Initialize the aggregates; perhaps we should do this in a single kernel? ... probably not worth it
    streams[0].enqueue.memset(aggregates_on_device.sum_quantity.get(),         0, num_potential_groups * sizeof(sum_quantity_t));
    streams[0].enqueue.memset(aggregates_on_device.sum_base_price.get(),       0, num_potential_groups * sizeof(sum_base_price_t));
    streams[0].enqueue.memset(aggregates_on_device.sum_discounted_price.get(), 0, num_potential_groups * sizeof(sum_discounted_price_t));
    streams[0].enqueue.memset(aggregates_on_device.sum_charge.get(),           0, num_potential_groups * sizeof(sum_charge_t));
    streams[0].enqueue.memset(aggregates_on_device.sum_discount.get(),         0, num_potential_groups * sizeof(sum_discount_t));
    streams[0].enqueue.memset(aggregates_on_device.record_count.get(),         0, num_potential_groups * sizeof(record_count_t));

    cuda::event_t aggregates_initialized_event = cuda_device.create_event(
            cuda::event::sync_by_blocking, cuda::event::dont_record_timings, cuda::event::not_interprocess);
    aggregates_initialized_event.record(streams[0].id());
    for (int i = 1; i < num_gpu_streams; ++i) {
        streams[i].enqueue.event(aggregates_initialized_event);
        // The other streams also require the aggregates to be initialized before doing any work
    }

    for (size_t offset_in_table = 0; offset_in_table < cardinality; offset_in_table += records_per_scheduled_kernel) {

        auto num_records_for_this_launch = std::min<record_count_t>(records_per_scheduled_kernel, cardinality - offset_in_table);

        // auto start_copy = timer::now();  // This can't work, since copying is asynchronous.
        auto stream_index = offset_in_table % num_gpu_streams; // so this loop willl act on streams in a round-robin fashion
        auto& stream = streams[stream_index];
        auto& stream_input_buffers = stream_input_buffer_sets[stream_index];
        stream.enqueue.copy(stream_input_buffers.shipdate.get()     , shipdate      + offset_in_table, num_records_for_this_launch * sizeof(ship_date_t));
        stream.enqueue.copy(stream_input_buffers.discount.get()     , discount      + offset_in_table, num_records_for_this_launch * sizeof(discount_t));
        stream.enqueue.copy(stream_input_buffers.extendedprice.get(), extendedprice + offset_in_table, num_records_for_this_launch * sizeof(extended_price_t));
        stream.enqueue.copy(stream_input_buffers.tax.get()          , tax           + offset_in_table, num_records_for_this_launch * sizeof(tax_t));
        stream.enqueue.copy(stream_input_buffers.quantity.get()     , quantity      + offset_in_table, num_records_for_this_launch * sizeof(quantity_t));
        stream.enqueue.copy(stream_input_buffers.returnflag.get()   , returnflag    + offset_in_table, num_records_for_this_launch * sizeof(return_flag_t));
        stream.enqueue.copy(stream_input_buffers.linestatus.get()   , linestatus    + offset_in_table, num_records_for_this_launch * sizeof(line_status_t));
        // auto end_copy = timer::now();
        // copy_time += std::chrono::duration<double>(end_copy - start_copy).count();

        auto num_blocks = div_rounding_up(num_records_for_this_launch, THREADS_PER_BLOCK);
        auto launch_config = cuda::make_launch_config(num_blocks, THREADS_PER_BLOCK);

        // auto start_kernel = timer::now(); // This won't work either, kernels are asynchronous
        //std::cout << "Execution <<<" << amount_of_blocks << "," << THREADS_PER_BLOCK << "," << SHARED_MEMORY << ">>>" << std::endl;

/*
        stream.enqueue.kernel_launch(
            cuda::thread_local_tpchQ01,
            launch_config,
            aggregates_on_device.sum_quantity.get(),
            aggregates_on_device.sum_base_price.get(),
            aggregates_on_device.sum_discounted_price.get(),
            aggregates_on_device.sum_charge.get(),
            aggregates_on_device.sum_discount.get(),
            aggregates_on_device.record_count.get(),
            stream_input_buffers.shipdate.get(),
            stream_input_buffers.discount.get(),
            stream_input_buffers.extendedprice.get(),
            stream_input_buffers.tax.get(),
            stream_input_buffers.returnflag.get(),
            stream_input_buffers.linestatus.get(),
            stream_input_buffers.quantity.get(),
            num_records_for_this_launch);
*/

        // auto end_kernel = timer::now();
        // computation_time += std::chrono::duration<double>(end_kernel - start_kernel).count();
    }

//    for (auto& stream : streams) { stream.synchronize(); }

    streams[0].enqueue.copy(aggregates_on_host.sum_quantity.get(),         aggregates_on_device.sum_quantity.get(),         num_potential_groups * sizeof(sum_quantity_t));
    streams[0].enqueue.copy(aggregates_on_host.sum_base_price.get(),       aggregates_on_device.sum_base_price.get(),       num_potential_groups * sizeof(sum_base_price_t));
    streams[0].enqueue.copy(aggregates_on_host.sum_discounted_price.get(), aggregates_on_device.sum_discounted_price.get(), num_potential_groups * sizeof(sum_discounted_price_t));
    streams[0].enqueue.copy(aggregates_on_host.sum_charge.get(),           aggregates_on_device.sum_charge.get(),           num_potential_groups * sizeof(sum_charge_t));
    streams[0].enqueue.copy(aggregates_on_host.sum_discount.get(),         aggregates_on_device.sum_discount.get(),         num_potential_groups * sizeof(sum_discount_t));
    streams[0].enqueue.copy(aggregates_on_host.record_count.get(),         aggregates_on_device.record_count.get(),         num_potential_groups * sizeof(record_count_t));

    // What about the remainder of the computation? :
    //
    // 1. Filter out potential groups with no elements (count 0)
    // 2. Normalize values (w.r.t. decimal scaling)
    // 3. Calculate averages using sums

    auto end = timer::now();

    std::cout << "\n"
                 "+--------------------------------------------------- Results ---------------------------------------------------+\n";
    std::cout << "|  LS | RF | sum_quantity        | sum_base_price      | sum_disc_price      | sum_charge          | count      |\n";
    std::cout << "+---------------------------------------------------------------------------------------------------------------+\n";
    auto print_dec = [] (auto s, auto x) { printf("%s%16ld.%02ld", s, Decimal64::GetInt(x), Decimal64::GetFrac(x)); };
    for (size_t group=0; group<num_potential_groups; group++) {
        if (aggregates_on_host.record_count[group] > 0) {
            char rf = '-', ls = '-';
            auto return_flag_group_id = group >> 1;
            auto line_status_group_id = group & 0x1;
            switch(return_flag_group_id) {
            case 0:  rf = 'A';
            case 1:  rf = 'F';
            case 2:  rf = 'N';
            default: rf = '-';
            }
            ls = (line_status_group_id == 0 ? 'F' : 'O');
            if (rf == 'A' and ls == 'F') {
                if (cardinality == 6001215) {
                    assert(aggregates_on_host.sum_quantity[i] == 3773410700);
                    assert(aggregates_on_host.record_count[i] == 1478493);
                }
            } else if (rf == 'N' and ls == 'F') {
                if (cardinality == 6001215) {
                    assert(aggregates_on_host.sum_quantity[i] == 99141700);
                    assert(aggregates_on_host.record_count[i] == 38854);
                }
            } else if (rf == 'N' and ls == 'O') {
                rf = 'N';
                ls = 'O';
                if (cardinality == 6001215) {
                    assert(aggregates_on_host.sum_quantity[i] == 7447604000);
                    assert(aggregates_on_host.record_count[i] == 2920374);
                }
            } else if (rf == 'R' and ls == 'F') {
                if (cardinality == 6001215) {
                    assert(aggregates_on_host.sum_quantity[i]== 3771975300);
                    assert(aggregates_on_host.record_count[i]== 1478870);
                }
            }

            printf("| # %c | %c ", rf, ls);
            print_dec(" | ",  aggregates_on_host.sum_quantity.get()[group]);
            print_dec(" | ",  aggregates_on_host.sum_base_price.get()[group]);
            print_dec(" | ",  aggregates_on_host.sum_discounted_price.get()[group]);
            print_dec(" | ",  aggregates_on_host.sum_charge.get()[group]);
            printf(" | %10u |\n", aggregates_on_host.record_count.get()[group]);
        }
    }
    std::cout << "+---------------------------------------------------------------------------------------------------------------+\n";

    double sf = cardinality / 6001215.0;
    uint64_t cache_line_size = 128; // bytes
    uint64_t num_loads =  1478493 + 38854 + 2920374 + 1478870 + 6;
    uint64_t num_stores = 19;
    std::chrono::duration<double> duration(end - start);
    uint64_t tuples_per_second               = static_cast<uint64_t>(data_length / duration.count());
    auto size_per_tuple = sizeof(ship_date_t) + sizeof(discount_t) + sizeof(extended_price_t) + sizeof(tax_t) + sizeof(quantity_t) + sizeof(return_flag_t) + sizeof(line_status_t);
    double effective_memory_throughput       = static_cast<double>((tuples_per_second * size_per_tuple) / GIGA);
    double estimated_memory_throughput       = static_cast<double>((tuples_per_second * cache_line_size) / GIGA);
    double effective_memory_throughput_read  = static_cast<double>((tuples_per_second * size_per_tuple) / GIGA);
    double effective_memory_throughput_write = static_cast<double>(tuples_per_second / (size_per_tuple * GIGA));
    double theretical_memory_bandwidth       = static_cast<double>((5505 * 10e06 * (352 / 8) * 2) / 10e09);
    double efective_memory_bandwidth         = static_cast<double>(((data_length * sizeof(ship_date_t)) + (num_loads * size_per_tuple) + (num_loads * num_stores))  / (duration.count() * 10e09));
    double csv_time = std::chrono::duration<double>(end_csv - start_csv).count();
    double pre_process_time = std::chrono::duration<double>(end_preprocess - start_preprocess).count();
    
    std::cout << "\n+------------------------------------------------- Statistics --------------------------------------------------+\n";
    std::cout << "| TPC-H Q01 performance               : ="          << std::fixed 
              << tuples_per_second <<                 " [tuples/sec]" << std::endl;
    std::cout << "| Time taken                          : ~"          << std::setprecision(2)
              << duration.count() <<                  "  [s]"          << std::endl;
    std::cout << "| Estimated time for TPC-H SF100      : ~"          << std::setprecision(2)
              << duration.count() * (100 / sf) <<     "  [s]"          << std::endl;
    std::cout << "| CSV Time                            : ~"          << std::setprecision(2)
              <<  csv_time <<                         "  [s]"          << std::endl;
    std::cout << "| Preprocess Time                     : ~"          << std::setprecision(2)
              <<  pre_process_time <<                 "  [s]"          << std::endl;
    std::cout << "| Copy Time                           : ~"          << std::setprecision(2)
              << copy_time <<                         "  [s]"          << std::endl;
    std::cout << "| Computation Time                    : ~"          << std::setprecision(2)
              << computation_time <<                  "  [s]"          << std::endl;
    std::cout << "| Effective memory throughput (query) : ~"          << std::setprecision(2)
              << effective_memory_throughput <<       "  [GB/s]"       << std::endl;
    std::cout << "| Estimated memory throughput (query) : ~"          << std::setprecision(1)
              << estimated_memory_throughput <<       "  [GB/s]"       << std::endl;
    std::cout << "| Effective memory throughput (read)  : ~"          << std::setprecision(2)
              << effective_memory_throughput_read <<  "  [GB/s]"       << std::endl;
    std::cout << "| Memory throughput (write)           : ~"          << std::setprecision(2)
              << effective_memory_throughput_write << "  [GB/s]"       << std::endl;
    std::cout << "| Theoretical Bandwidth               : ="          << std::setprecision(1)
              << theretical_memory_bandwidth <<       " [GB/s]"       << std::endl;
    std::cout << "| Effective Bandwidth                 : ~"          << std::setprecision(2)
              << efective_memory_bandwidth <<         "  [GB/s]"       << std::endl;
    std::cout << "+---------------------------------------------------------------------------------------------------------------+\n";
}
