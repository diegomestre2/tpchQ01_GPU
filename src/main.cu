#include <iostream>
#include <cuda/api_wrappers.h>
#include <vector>
#include <iomanip>
#include <fstream>
#include <chrono>

#include "data_types.h"
#include "constants.hpp"
#include "kernel.hpp"
#include "kernels/naive.hpp"
#include "kernels/local.hpp"
#include "kernels/global.hpp"
#include "kernels/coalesced.hpp"
#include "../expl_comp_strat/tpch_kit.hpp"
#include "../expl_comp_strat/common.hpp"

size_t magic_hash(char rf, char ls) {
    return (((rf - 'A')) - (ls - 'F'));
}

void assert_always(bool a) {
    if (!a) {
        fprintf(stderr, "Assert always failed!");
        exit(1);
    }
}

void syscall(std::string command) {
    auto x = system(command.c_str());
    (void) x;
}

#define GIGA (1024 * 1024 * 1024)
#define MEGA (1024 * 1024)
#define KILO (1024)

using timer = std::chrono::high_resolution_clock;

inline bool file_exists (const std::string& name) {
  struct stat buffer;
  return (stat (name.c_str(), &buffer) == 0);
}

inline std::string join_path(std::string a, std::string b) {
    return a + "/" + b;
}

std::ifstream::pos_type filesize(std::string filename) {
    std::ifstream in(filename.c_str(), std::ifstream::ate | std::ifstream::binary);
    return in.tellg(); 
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


template <typename T>
void load_column_from_binary_file(T* buffer, size_t cardinality, const std::string& directory, const std::string& file_name) {
	// TODO: C++'ify the file access (will also guarantee exception safety)
	auto file_path = join_path(directory, file_name);
    FILE* pFile = fopen(file_path.c_str(), "rb");
    assert_always(pFile);
    auto num_items_read = fread(buffer, sizeof(T), cardinality, pFile);
    if (num_items_read != cardinality) { throw std::runtime_error("Failed reading sufficient data from file"); }
    fclose(pFile);
}

template <typename T>
void write_column_to_binary_file(const T* buffer, size_t cardinality, const std::string& directory, const std::string& file_name) {
	auto file_path = join_path(directory, file_name);
    FILE* pFile = fopen(file_path.c_str(), "wb+");
    assert_always(pFile);
    fwrite(buffer, sizeof(T), cardinality, pFile);
    fclose(pFile);
}

void print_help() {
    fprintf(stderr, "Unrecognized command line option.\n");
    fprintf(stderr, "Usage: tpch_01 [args]\n");
    fprintf(stderr, "   --sf=[sf] (number, e.g. 0.01 - 100)\n");
    fprintf(stderr, "   --use-global-ht\n");
    fprintf(stderr, "   --use-small-datatypes\n");
    fprintf(stderr, "   --use-coalescing\n");
}

int main(int argc, char** argv) {
    std::cout << "TPC-H Query 1" << '\n';
    /* load data */

    size_t cardinality;

    auto start_csv = timer::now();

#if 0
    // of course we always use pinned memory
    bool USE_PINNED_MEMORY = true;
    // not used
    bool USE_GLOBAL_HT = false;
#endif
    bool apply_compression = false;
#if 0
    // of course we want to coalesce memory accesses, always
    bool USE_COALESCING = false;
#endif

    double scale_factor = 1;
    std::string sf_argument = "--sf=";
    for(int i = 1; i < argc; i++) {
        auto arg = std::string(argv[i]);
/*        if (arg == "--use-global-ht") {
            USE_GLOBAL_HT = true;
        } else */
        if (arg == "--use-small-datatypes") {
            apply_compression = true;
        } else if (arg.substr(0, sf_argument.size()) == sf_argument) {
            scale_factor = std::stod(arg.substr(sf_argument.size()));
        } else {
            print_help();
            exit(1);
        }
    }
    lineitem li((size_t)(7000000 * scale_factor));


    std::unique_ptr< ship_date_t[]      > _shipdate;
    std::unique_ptr< return_flag_t[]    > _returnflag;
    std::unique_ptr< line_status_t[]    > _linestatus;
    std::unique_ptr< discount_t[]       > _discount;
    std::unique_ptr< tax_t[]            > _tax;
    std::unique_ptr< extended_price_t[] > _extendedprice;
    std::unique_ptr< quantity_t[]       > _quantity;

    // TODO: Use std::filesystem for the filesystem stuff
    syscall("mkdir -p tpch");
    std::string tpch_directory = join_path("tpch", std::to_string(scale_factor));
    syscall(std::string("mkdir -p ") + tpch_directory);
    if (file_exists(join_path(tpch_directory, "shipdate.bin"))) {
        std::cout << "Loading from binary." << std::endl;
        // binary files exist, load them
        cardinality = filesize(join_path(tpch_directory, "shipdate.bin")) / sizeof(ship_date_t);
        if (cardinality == 0) {
        	throw std::runtime_error("The lineitem table column cardinality should not be 0");
        }
        _shipdate = std::make_unique<ship_date_t[]>(cardinality);
        load_column_from_binary_file(_shipdate.get(),      cardinality, tpch_directory, "shipdate.bin");
        _returnflag = std::make_unique<return_flag_t[]>(cardinality);
        load_column_from_binary_file(_returnflag.get(),    cardinality, tpch_directory, "returnflag.bin");
        _linestatus = std::make_unique<line_status_t[]>(cardinality);
        load_column_from_binary_file(_linestatus.get(),    cardinality, tpch_directory, "linestatus.bin");
        _discount = std::make_unique<discount_t[]>(cardinality);
        load_column_from_binary_file(_discount.get(),      cardinality, tpch_directory, "discount.bin");
        _tax  = std::make_unique<tax_t[]>(cardinality);
        load_column_from_binary_file(_tax.get(),           cardinality, tpch_directory, "tax.bin");
        _extendedprice = std::make_unique<extended_price_t[]>(cardinality);
        load_column_from_binary_file(_extendedprice.get(), cardinality, tpch_directory, "extendedprice.bin");
        _quantity = std::make_unique<quantity_t[]>(cardinality);
        load_column_from_binary_file(_quantity.get(),      cardinality, tpch_directory, "quantity.bin");
    } else {
        std::cout << "Reading CSV file and writing to binary." << std::endl;
        std::string input_file = join_path(tpch_directory, "lineitem.tbl");
        if (!file_exists(input_file.c_str())) {
            // have to generate lineitem file
            syscall("./genlineitem.sh " + std::to_string(scale_factor));
            syscall("mv lineitem.tbl " + input_file);
        }
        li.FromFile(input_file.c_str());
        cardinality = li.l_extendedprice.cardinality;
        if (cardinality == 0) {
        	throw std::runtime_error("The lineitem table column cardinality should not be 0");
        }
        write_column_to_binary_file(_shipdate.get(),      cardinality, tpch_directory, "shipdate.bin");
        write_column_to_binary_file(_returnflag.get(),    cardinality, tpch_directory, "returnflag.bin");
        write_column_to_binary_file(_linestatus.get(),    cardinality, tpch_directory, "linestatus.bin");
        write_column_to_binary_file(_discount.get(),      cardinality, tpch_directory, "discount.bin");
        write_column_to_binary_file(_tax.get(),           cardinality, tpch_directory, "tax.bin");
        write_column_to_binary_file(_extendedprice.get(), cardinality, tpch_directory, "extendedprice.bin");
        write_column_to_binary_file(_quantity.get(),      cardinality, tpch_directory, "quantity.bin");
    }

    auto compressed_ship_date      = std::make_unique< compressed::ship_date_t[]      >(cardinality);
    auto compressed_discount       = std::make_unique< compressed::discount_t[]       >(cardinality);
    auto compressed_extended_price = std::make_unique< compressed::extended_price_t[] >(cardinality);
    auto compressed_tax            = std::make_unique< compressed::tax_t[]            >(cardinality);
    auto compressed_quantity       = std::make_unique< compressed::quantity_t[]       >(cardinality);
    auto compressed_return_flag    = std::make_unique< compressed::return_flag_t[]    >(div_rounding_up(cardinality, return_flag_values_per_container));
    auto compressed_line_status    = std::make_unique< compressed::line_status_t[]    >(div_rounding_up(cardinality, line_status_values_per_container));

    auto end_csv = timer::now();

    auto start_preprocess = timer::now();

    // Eyal says: Drop these copies, we really don't need them AFAICT
    auto shipdate      = _shipdate.get();
    auto returnflag    = _returnflag.get();
    auto linestatus    = _linestatus.get();
    auto discount      = _discount.get();
    auto tax           = _tax.get();
    auto extendedprice = _extendedprice.get();
    auto quantity      = _quantity.get();

    for(size_t i = 0; i < cardinality; i++) {
/*
        shipdate[i] = _shipdate[i];
        discount[i] = _discount[i];
        extendedprice[i] = _extendedprice[i];
        quantity[i] = _quantity[i];
        tax[i] = _tax[i];
        returnflag[i] = _returnflag[i];
        linestatus[i] = _linestatus[i];
*/

        compressed_ship_date[i]      = shipdate[i] - ship_date_frame_of_reference;
        compressed_discount[i]       = discount[i];
        compressed_extended_price[i] = extendedprice[i];
        compressed_quantity[i]       = quantity[i] / 100;
        compressed_tax[i]            = tax[i];
        // TODO: Don't use magic numbers here
        if (i % 4 == 0) {
            compressed_return_flag[i / 4] = 0;
            for(size_t j = 0; j < std::min((size_t) 4, cardinality - i); j++) {
                // 'N' = 0x00, 'R' = 0x01, 'A' = 0x10
                compressed_return_flag[i / 4] |=
                    (_returnflag[i + j] == 'N' ? 0x00 : (_returnflag[i + j] == 'R' ? 0x01 : 0x02)) << (j * 2);
            }
        }
        // TODO: Don't use magic numbers here
        if (i % 8 == 0) {
            compressed_line_status[i / 8] = 0;
            for(size_t j = 0; j < std::min((size_t) 8, cardinality - i); j++) {
                // 'O' = 0, 'F' = 1
                compressed_line_status[i / 8] |= (_linestatus[i + j] == 'F' ? 1 : 0) << j;
            }
        }

        assert((int)     compressed_ship_date[i]       == shipdate[i] - ship_date_frame_of_reference);
        assert((int64_t) compressed_discount[i]       == discount[i]);
        assert((int64_t) compressed_extended_price[i] == extendedprice[i]);
        assert((int64_t) compressed_quantity[i]       == quantity[i] / 100);
        assert((int64_t) compressed_tax[i]            == tax[i]);
    }

    constexpr uint8_t RETURNFLAG_MASK[] = { 0x03, 0x0C, 0x30, 0xC0 };
    constexpr uint8_t LINESTATUS_MASK[] = { 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80 };
    for(size_t i = 0; i < cardinality; i++) {
        uint8_t retflag = (compressed_return_flag[i / 4] & RETURNFLAG_MASK[i % 4]) >> (2 * (i % 4));
        uint8_t lstatus = (compressed_line_status[i / 8] & LINESTATUS_MASK[i % 8]) >> (i % 8);
        assert_always(retflag == (returnflag[i] == 'N' ? 0x00 : (returnflag[i] == 'R' ? 0x01 : 0x02)));
        assert_always(lstatus == (linestatus[i] == 'F' ? 1 : 0));
    }
    auto end_preprocess = timer::now();

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
        cuda::memory::device::unique_ptr< compressed::ship_date_t[]      > shipdate;
        cuda::memory::device::unique_ptr< compressed::discount_t[]       > discount;
        cuda::memory::device::unique_ptr< compressed::extended_price_t[] > extendedprice;
        cuda::memory::device::unique_ptr< compressed::tax_t[]            > tax;
        cuda::memory::device::unique_ptr< compressed::quantity_t[]       > quantity;
        cuda::memory::device::unique_ptr< bit_container_t[]              > returnflag;
        cuda::memory::device::unique_ptr< bit_container_t[]              > linestatus;
    };

    std::vector<stream_input_buffer_set> stream_input_buffer_sets;
    std::vector<cuda::stream_t<>> streams;
    stream_input_buffer_sets.reserve(num_gpu_streams);
    streams.reserve(num_gpu_streams);
        // We'll be scheduling (most of) our work in a round-robin fashion on all of
        // the streams, to prevent the GPU from idling.


    for (int i = 0; i < num_gpu_streams; ++i) {
        auto input_buffers = stream_input_buffer_set{
            cuda::memory::device::make_unique< compressed::ship_date_t[]      >(cuda_device, records_per_scheduled_kernel),
            cuda::memory::device::make_unique< compressed::discount_t[]       >(cuda_device, records_per_scheduled_kernel),
            cuda::memory::device::make_unique< compressed::extended_price_t[] >(cuda_device, records_per_scheduled_kernel),
            cuda::memory::device::make_unique< compressed::tax_t[]            >(cuda_device, records_per_scheduled_kernel),
            cuda::memory::device::make_unique< compressed::quantity_t[]       >(cuda_device, records_per_scheduled_kernel),
            cuda::memory::device::make_unique< bit_container_t[]              >(cuda_device, div_rounding_up(records_per_scheduled_kernel, return_flag_values_per_container)),
            cuda::memory::device::make_unique< bit_container_t[]              >(cuda_device, div_rounding_up(records_per_scheduled_kernel, line_status_values_per_container))
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
    streams[0].enqueue.event(aggregates_initialized_event);
    for (int i = 1; i < num_gpu_streams; ++i) {
        streams[i].enqueue.wait(aggregates_initialized_event);
        // The other streams also require the aggregates to be initialized before doing any work
    }

    for (size_t offset_in_table = 0; offset_in_table < cardinality; offset_in_table += records_per_scheduled_kernel) {

        auto num_records_for_this_launch = std::min<record_count_t>(records_per_scheduled_kernel, cardinality - offset_in_table);

        // auto start_copy = timer::now();  // This can't work, since copying is asynchronous.
        auto stream_index = offset_in_table % num_gpu_streams; // so this loop willl act on streams in a round-robin fashion
        auto& stream = streams[stream_index];
        auto& stream_input_buffers = stream_input_buffer_sets[stream_index];
        stream.enqueue.copy(stream_input_buffers.shipdate.get()     , compressed_ship_date.get()      + offset_in_table, num_records_for_this_launch * sizeof(ship_date_t));
        stream.enqueue.copy(stream_input_buffers.discount.get()     , compressed_discount.get()       + offset_in_table, num_records_for_this_launch * sizeof(discount_t));
        stream.enqueue.copy(stream_input_buffers.extendedprice.get(), compressed_extended_price.get() + offset_in_table, num_records_for_this_launch * sizeof(extended_price_t));
        stream.enqueue.copy(stream_input_buffers.tax.get()          , compressed_tax.get()            + offset_in_table, num_records_for_this_launch * sizeof(tax_t));
        stream.enqueue.copy(stream_input_buffers.quantity.get()     , compressed_quantity.get()       + offset_in_table, num_records_for_this_launch * sizeof(quantity_t));
        stream.enqueue.copy(stream_input_buffers.returnflag.get()   , compressed_return_flag.get()    + offset_in_table / return_flag_values_per_container, num_records_for_this_launch / return_flag_values_per_container * sizeof(bit_container_t));
        stream.enqueue.copy(stream_input_buffers.linestatus.get()   , compressed_line_status.get()    + offset_in_table / line_status_values_per_container, num_records_for_this_launch / line_status_values_per_container * sizeof(bit_container_t));

        auto num_blocks = div_rounding_up(num_records_for_this_launch, THREADS_PER_BLOCK);
        auto launch_config = cuda::make_launch_config(num_blocks, THREADS_PER_BLOCK);
        (void) launch_config;


        stream.enqueue.kernel_launch(
            cuda::thread_local_tpchQ01_snall_datatypes,
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

    }
    std::vector<cuda::event_t> completion_events;
    for(int i = 1; i < num_gpu_streams; i++) {
    	auto event = streams[i].enqueue.event();
    	completion_events.emplace_back(std::move(event));
    }

    // It's probably a better idea to go round-robin on the streams here
    streams[0].enqueue.copy(aggregates_on_host.sum_quantity.get(),         aggregates_on_device.sum_quantity.get(),         num_potential_groups * sizeof(sum_quantity_t));
    streams[0].enqueue.copy(aggregates_on_host.sum_base_price.get(),       aggregates_on_device.sum_base_price.get(),       num_potential_groups * sizeof(sum_base_price_t));
    streams[0].enqueue.copy(aggregates_on_host.sum_discounted_price.get(), aggregates_on_device.sum_discounted_price.get(), num_potential_groups * sizeof(sum_discounted_price_t));
    streams[0].enqueue.copy(aggregates_on_host.sum_charge.get(),           aggregates_on_device.sum_charge.get(),           num_potential_groups * sizeof(sum_charge_t));
    streams[0].enqueue.copy(aggregates_on_host.sum_discount.get(),         aggregates_on_device.sum_discount.get(),         num_potential_groups * sizeof(sum_discount_t));
    streams[0].enqueue.copy(aggregates_on_host.record_count.get(),         aggregates_on_device.record_count.get(),         num_potential_groups * sizeof(record_count_t));

    streams[0].synchronize();


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
            case 0:  rf = 'A'; break;
            case 1:  rf = 'F'; break;
            case 2:  rf = 'N'; break;
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


    uint64_t cache_line_size = 128; // bytes <- not quite true on a GPU, L2 cache is 32 bytes, L1 is 128
    uint64_t num_loads =  1478493 + 38854 + 2920374 + 1478870 + 6;
    uint64_t num_stores = 19;
    std::chrono::duration<double> duration(end - start);
    uint64_t tuples_per_second               = static_cast<uint64_t>(cardinality / duration.count());

    auto size_per_tuple = sizeof(ship_date_t) + sizeof(discount_t) + sizeof(extended_price_t) + sizeof(tax_t) + sizeof(quantity_t) + sizeof(return_flag_t) + sizeof(line_status_t);
    if (apply_compression) {
        size_per_tuple = sizeof(compressed::ship_date_t) + sizeof(compressed::discount_t) + sizeof(compressed::extended_price_t) + sizeof(compressed::tax_t) + sizeof(compressed::quantity_t) + sizeof(compressed::return_flag_t) + sizeof(compressed::line_status_t);
    }

    double effective_memory_throughput       = static_cast<double>((tuples_per_second * size_per_tuple) / GIGA);
    double estimated_memory_throughput       = static_cast<double>((tuples_per_second * cache_line_size) / GIGA);
    double effective_memory_throughput_read  = static_cast<double>((tuples_per_second * size_per_tuple) / GIGA);
    double effective_memory_throughput_write = static_cast<double>(tuples_per_second / (size_per_tuple * GIGA));
    double theretical_memory_bandwidth       = static_cast<double>((5505 * 10e06 * (352 / 8) * 2) / 10e09);
    double efective_memory_bandwidth         = static_cast<double>(((cardinality * sizeof(ship_date_t)) + (num_loads * size_per_tuple) + (num_loads * num_stores))  / (duration.count() * 10e09));
    double csv_time = std::chrono::duration<double>(end_csv - start_csv).count();
    double pre_process_time = std::chrono::duration<double>(end_preprocess - start_preprocess).count();
    
    std::cout << "\n+------------------------------------------------- Statistics --------------------------------------------------+\n";
    std::cout << "| TPC-H Q01 performance               : ="          << std::fixed 
              << tuples_per_second <<                 " [tuples/sec]" << std::endl;
    std::cout << "| Time taken                          : ~"          << std::setprecision(2)
              << duration.count() <<                  "  [s]"          << std::endl;
    std::cout << "| Estimated time for TPC-H SF100      : ~"          << std::setprecision(2)
              << duration.count() * (100 / scale_factor) <<     "  [s]"          << std::endl;
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
