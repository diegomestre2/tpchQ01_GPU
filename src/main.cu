#include <iostream>
#include <cuda/api_wrappers.h>
#include <vector>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <tuple>

#include "data_types.h"
#include "constants.hpp"
#include "bit_operations.h"
#include "kernel.hpp"
//#include "kernels/naive.hpp"
//#include "kernels/local.hpp"
//#include "kernels/global.hpp"
#include "kernels/coalesced.hpp"
#include "../expl_comp_strat/tpch_kit.hpp"
#include "../expl_comp_strat/common.hpp"
#include "cpu/common.hpp"
#include "cpu.h"

#ifndef GPU
#error The GPU preprocessor directive must be defined (ask Tim for the reason)
#endif

using std::tie;
using std::make_pair;
using std::make_unique;
using std::unique_ptr;
using std::cout;
using std::endl;
using std::flush;
using std::string;

using std::cout;
using std::endl;
using std::string;

using std::cout;
using std::endl;
using std::string;

size_t magic_hash(char rf, char ls) {
    return (((rf - 'A')) - (ls - 'F'));
}

inline void assert_always(bool a) {
    if (!a) {
        fprintf(stderr, "Assert always failed!");
        exit(1);
    }
}

void syscall(string command) {
    auto x = system(command.c_str());
    (void) x;
}

#define GIGA (1024 * 1024 * 1024)
#define MEGA (1024 * 1024)
#define KILO (1024)

using timer = std::chrono::high_resolution_clock;

inline bool file_exists(const string& name) {
    struct stat buffer;
    return (stat (name.c_str(), &buffer) == 0);
}

inline string join_path(string a, string b) {
    return a + "/" + b;
}

std::ifstream::pos_type filesize(string filename) {
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

template <typename UniquePtr>
void load_column_from_binary_file(
    UniquePtr&          buffer,
    cardinality_t       cardinality,
    const string&  directory,
    const string&  file_name)
{
    // TODO: C++'ify the file access (will also guarantee exception safety)
    using raw_ptr_type = typename std::decay<decltype(buffer.get())>::type;
    using element_type = typename std::remove_pointer<raw_ptr_type>::type;
    auto file_path = join_path(directory, file_name);
    buffer = std::make_unique<element_type[]>(cardinality);
    cout << "Loading a column from " << file_path << " ... " << flush;
    FILE* pFile = fopen(file_path.c_str(), "rb");
    if (pFile == nullptr) { throw std::runtime_error("Failed opening file " + file_path); }
    auto num_elements_read = fread(buffer.get(), sizeof(element_type), cardinality, pFile);
    if (num_elements_read != cardinality) {
        throw std::runtime_error("Failed reading sufficient data from " +
            file_path + " : expected " + std::to_string(cardinality) + " elements but read only " + std::to_string(num_elements_read) + "."); }
    fclose(pFile);
    cout << "done." << endl;
}

template <typename T>
void write_column_to_binary_file(const T* buffer, cardinality_t cardinality, const string& directory, const string& file_name) {
    auto file_path = join_path(directory, file_name);
    cout << "Writing a column to " << file_path << " ... " << flush;
    FILE* pFile = fopen(file_path.c_str(), "wb+");
    if (pFile == nullptr) { throw std::runtime_error("Failed opening file " + file_path); }
    auto num_elements_written = fwrite(buffer, sizeof(T), cardinality, pFile);
    fclose(pFile);
    if (num_elements_written != cardinality) {
        remove(file_path.c_str());
        throw std::runtime_error("Failed writing all elements to the file - only " +
            std::to_string(num_elements_written) + " written: " + strerror(errno));
    }
    cout << "done." << endl;
}

void print_help() {
    fprintf(stderr, "Unrecognized command line option.\n");
    fprintf(stderr, "Usage: tpch_01 [args]\n");
    fprintf(stderr, "   --sf=[df:1] (number, e.g. 0.01 - 100)\n");
    fprintf(stderr, "   --streams=[streams:8] (number, e.g. 1 - 64)\n");
    fprintf(stderr, "   --threads-per-block=[threads:32] (number, e.g. 32 - 1024)\n");
//    fprintf(stderr, "   --compress\n");
}

template <typename F, typename... Args>
void for_each_argument(F f, Args&&... args) {
    [](...){}((f(std::forward<Args>(args)), 0)...);
}

GPUAggrHashTable aggrs0[num_potential_groups] ALIGN;

#define init_table(ag) memset(&aggrs##ag, 0, sizeof(aggrs##ag))
#define clear(x) memset(x, 0, sizeof(x))

extern "C" void
clear_tables()
{
    init_table(0);
}

void make_sure_we_are_on_cpu_core_0()
{
    // Make sure we are on core 0
    // TODO: Why not in a function?
    cpu_set_t cpuset; 

    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    sched_setaffinity(0, sizeof(cpuset), &cpuset);
}

#include "cpu.h"


int main(int argc, char** argv) {
    cout << "TPC-H Query 1" << '\n';
    make_sure_we_are_on_cpu_core_0();

    cardinality_t cardinality;

    double scale_factor = 1;
    int num_gpu_streams = defaults::num_gpu_streams;
    int num_threads_per_block = defaults::num_threads_per_block;
    int num_records_per_scheduled_kernel = defaults::num_records_per_scheduled_kernel;
        // Make sure it's a multiple of num_threads_per_block, or bad things may happen

    // This is the number of times we run the actual query execution - the part that we time;
    // it will not include initialization/allocations that are not necessary when the DBMS
    // is brought up. Note the allocation vs sub-allocation issue (see further comments below)
    int num_query_execution_runs = 5;

    string sf_argument = "--sf=";
    string streams_argument = "--streams=";
    string threads_per_block_argument = "--threads-per-block=";
    string num_runs_arguments = "--runs=";

    //bool apply_compression = true;
    bool use_coprocessing = false;

    for(int i = 1; i < argc; i++) {
        auto arg = string(argv[i]);
        if (arg == "--device") {
            get_device_properties();
            exit(1);
     //   } else if (arg == "--compress") {
     //       apply_compression = true;
        } else if (arg == "--use-coprocessing") {
            use_coprocessing = true;
        } else if (arg.substr(0, sf_argument.size()) == sf_argument) {
            scale_factor = std::stod(arg.substr(sf_argument.size()));
            if (scale_factor - 0 < 0.001) {
                std::invalid_argument("Invalid scale factor");
            }
        } else if (arg.substr(0, streams_argument.size()) == streams_argument) {
            num_gpu_streams = std::stoi(arg.substr(streams_argument.size()));
        } else if (arg.substr(0, threads_per_block_argument.size()) == threads_per_block_argument) {
            num_threads_per_block = std::stoi(arg.substr(threads_per_block_argument.size()));
        } else if (arg.substr(0, num_runs_arguments.size()) == num_runs_arguments) {
            num_query_execution_runs = std::stoi(arg.substr(num_runs_arguments.size()));
        } else {
            print_help();
            exit(1);
        }
    }
    lineitem li((size_t)(7000000 * std::max(scale_factor, 1.0)));
        // TODO: lineitem should really not need this cap, it should just adjust
        // allocated space as the need arises (and start with an estimate based on
        // the file size

    std::unique_ptr< ship_date_t[]      > _shipdate;
    std::unique_ptr< return_flag_t[]    > _returnflag;
    std::unique_ptr< line_status_t[]    > _linestatus;
    std::unique_ptr< discount_t[]       > _discount;
    std::unique_ptr< tax_t[]            > _tax;
    std::unique_ptr< extended_price_t[] > _extendedprice;
    std::unique_ptr< quantity_t[]       > _quantity;

    // TODO: Use std::filesystem for the filesystem stuff
    syscall("mkdir -p tpch");
    string tpch_directory = join_path("tpch", std::to_string(scale_factor));
    syscall(string("mkdir -p ") + tpch_directory);
    if (file_exists(join_path(tpch_directory, "shipdate.bin"))) {
        // binary files (seem to) exist, load them
        cardinality = filesize(join_path(tpch_directory, "shipdate.bin")) / sizeof(ship_date_t);
        if (cardinality == cardinality_of_scale_factor_1) {
            cardinality = ((double) cardinality) * scale_factor;
        }
        cout << "Lineitem table cardinality for scale factor " << scale_factor << " is " << cardinality << endl;
        if (cardinality == 0) {
            throw std::runtime_error("The lineitem table column cardinality should not be 0");
        }
        load_column_from_binary_file(_shipdate,      cardinality, tpch_directory, "shipdate.bin");
        load_column_from_binary_file(_returnflag,    cardinality, tpch_directory, "returnflag.bin");
        load_column_from_binary_file(_linestatus,    cardinality, tpch_directory, "linestatus.bin");
        load_column_from_binary_file(_discount,      cardinality, tpch_directory, "discount.bin");
        load_column_from_binary_file(_tax,           cardinality, tpch_directory, "tax.bin");
        load_column_from_binary_file(_extendedprice, cardinality, tpch_directory, "extendedprice.bin");
        load_column_from_binary_file(_quantity,      cardinality, tpch_directory, "quantity.bin");

        // See: We don't need no stinkin' macros these days. Actually, we can do something
        // similar with a lot of the replicated code in this file
        for_each_argument(
            [&](auto tup){
                std::get<0>(tup).cardinality = cardinality;
                std::get<0>(tup).m_ptr = std::get<1>(tup).get();
            },
            tie(li.l_shipdate,      _shipdate),
            tie(li.l_returnflag,    _returnflag),
            tie(li.l_linestatus,    _linestatus),
            tie(li.l_discount,      _discount),
            tie(li.l_tax,           _tax),
            tie(li.l_extendedprice, _extendedprice), 
            tie(li.l_quantity,      _quantity)
        );
    } else {
        std::string input_file = join_path(tpch_directory, "lineitem.tbl");
        if (not file_exists(input_file.c_str())) {
            throw std::runtime_error("Cannot locate table text file " + input_file);
            // Not generating it ourselves - that's: 1. Not healthy and 2. Not portable;
            // setup scripts are intended to do that
        }
        cout << "Parsing the lineitem table in file " << input_file << endl;
        li.FromFile(input_file.c_str());
        cardinality = li.l_extendedprice.cardinality;
        if (cardinality == cardinality_of_scale_factor_1) {
            cardinality = ((double) cardinality) * scale_factor;
        }
        if (cardinality == 0) {
            throw std::runtime_error("The lineitem table column cardinality should not be 0");
        }
        cout << "CSV read & parsed; table length: " << cardinality << " records." << endl;
        auto write_to = [&](auto& uptr, const char* filename) {
            using T = typename std::remove_pointer<typename std::decay<decltype(uptr.get())>::type>::type;
            load_column_from_binary_file(uptr.get(), cardinality, tpch_directory, "shipdate.bin");
        };
        write_column_to_binary_file(li.l_shipdate.get(),      cardinality, tpch_directory, "shipdate.bin");
        write_column_to_binary_file(li.l_returnflag.get(),    cardinality, tpch_directory, "returnflag.bin");
        write_column_to_binary_file(li.l_linestatus.get(),    cardinality, tpch_directory, "linestatus.bin");
        write_column_to_binary_file(li.l_discount.get(),      cardinality, tpch_directory, "discount.bin");
        write_column_to_binary_file(li.l_tax.get(),           cardinality, tpch_directory, "tax.bin");
        write_column_to_binary_file(li.l_extendedprice.get(), cardinality, tpch_directory, "extendedprice.bin");
        write_column_to_binary_file(li.l_quantity.get(),      cardinality, tpch_directory, "quantity.bin");    
    }


    clear_tables(); // currently only used by the CPU implementation
    CoProc* cpu = use_coprocessing ?  new CoProc(li, true) : nullptr;

    auto compressed_ship_date      = cuda::memory::host::make_unique< compressed::ship_date_t[]      >(cardinality);
    auto compressed_discount       = cuda::memory::host::make_unique< compressed::discount_t[]       >(cardinality);
    auto compressed_extended_price = cuda::memory::host::make_unique< compressed::extended_price_t[] >(cardinality);
    auto compressed_tax            = cuda::memory::host::make_unique< compressed::tax_t[]            >(cardinality);
    auto compressed_quantity       = cuda::memory::host::make_unique< compressed::quantity_t[]       >(cardinality);
    auto compressed_return_flag    = cuda::memory::host::make_unique< bit_container_t[] >(div_rounding_up(cardinality, return_flag_values_per_container));
    auto compressed_line_status    = cuda::memory::host::make_unique< bit_container_t[] >(div_rounding_up(cardinality, line_status_values_per_container));

    cout << "Preprocessing/compressing column data... " << flush;

    // Eyal says: Drop these copies, we really don't need them AFAICT
    auto shipdate      = _shipdate.get();
    auto returnflag    = _returnflag.get();
    auto linestatus    = _linestatus.get();
    auto discount      = _discount.get();
    auto tax           = _tax.get();
    auto extendedprice = _extendedprice.get();
    auto quantity      = _quantity.get();

    // Man, we really need to have a sub-byte-length-value container class
    std::memset(compressed_return_flag.get(), 0, div_rounding_up(cardinality, return_flag_values_per_container));
    std::memset(compressed_line_status.get(), 0, div_rounding_up(cardinality, line_status_values_per_container));
    for(cardinality_t i = 0; i < cardinality; i++) {
        compressed_ship_date[i]      = shipdate[i] - ship_date_frame_of_reference;
        compressed_discount[i]       = discount[i]; // we're keeping the factor 100 scaling
        compressed_extended_price[i] = extendedprice[i];
        compressed_quantity[i]       = quantity[i] / 100;
        compressed_tax[i]            = tax[i]; // we're keeping the factor 100 scaling
        set_bit_resolution_element<log_return_flag_bits, cardinality_t>(
            compressed_return_flag.get(), i, encode_return_flag(returnflag[i]));
        set_bit_resolution_element<log_line_status_bits, cardinality_t>(
            compressed_line_status.get(), i, encode_line_status(linestatus[i]));
        assert( (ship_date_t)      compressed_ship_date[i]      == shipdate[i] - ship_date_frame_of_reference);
        assert( (discount_t)       compressed_discount[i]       == discount[i]);
        assert( (extended_price_t) compressed_extended_price[i] == extendedprice[i]);
        assert( (quantity_t)       compressed_quantity[i]       == quantity[i] / 100);
            // not keeping the scaling here since we know the data is all integral; you could call this a form
            // of compression
        assert( (tax_t)            compressed_tax[i]            == tax[i]);
    }

//    cout << endl;
    for(cardinality_t i = 0; i < cardinality; i++) {
//        if (i < 16) cout << "Host " << std::setw(4) << i << " : " << returnflag[i] << " | " << linestatus[i] << endl;
        assert(decode_return_flag(get_bit_resolution_element<log_return_flag_bits, cardinality_t>(compressed_return_flag.get(), i)) == returnflag[i]);
        assert(decode_line_status(get_bit_resolution_element<log_line_status_bits, cardinality_t>(compressed_line_status.get(), i)) == linestatus[i]);
    }
    cout << "done." << endl;

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
        std::unique_ptr<cardinality_t[]         > record_count;
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
        std::make_unique< cardinality_t[]          >(num_potential_groups)
        // ,
        // {
        //      std::make_unique< avg_quantity_t[]         >(num_potential_groups),
        //      std::make_unique< avg_extended_price_t[]   >(num_potential_groups),
        //      std::make_unique< avg_discount_t[]         >(num_potential_groups),
        // }
    };

    cuda::profiling::start();


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
        cuda::memory::device::unique_ptr< cardinality_t[]          > record_count;
    } aggregates_on_device = {
        cuda::memory::device::make_unique< sum_quantity_t[]         >(cuda_device, num_potential_groups),
        cuda::memory::device::make_unique< sum_base_price_t[]       >(cuda_device, num_potential_groups),
        cuda::memory::device::make_unique< sum_discounted_price_t[] >(cuda_device, num_potential_groups),
        cuda::memory::device::make_unique< sum_charge_t []          >(cuda_device, num_potential_groups),
        cuda::memory::device::make_unique< sum_discount_t[]         >(cuda_device, num_potential_groups),
        cuda::memory::device::make_unique< cardinality_t[]          >(cuda_device, num_potential_groups)
    };

    struct stream_input_buffer_set {
        cuda::memory::device::unique_ptr< compressed::ship_date_t[]      > ship_date;
        cuda::memory::device::unique_ptr< compressed::discount_t[]       > discount;
        cuda::memory::device::unique_ptr< compressed::extended_price_t[] > extended_price;
        cuda::memory::device::unique_ptr< compressed::tax_t[]            > tax;
        cuda::memory::device::unique_ptr< compressed::quantity_t[]       > quantity;
        cuda::memory::device::unique_ptr< bit_container_t[]              > return_flag;
        cuda::memory::device::unique_ptr< bit_container_t[]              > line_status;
    };

    std::vector<stream_input_buffer_set> stream_input_buffer_sets;
    std::vector<cuda::stream_t<>> streams;
    stream_input_buffer_sets.reserve(num_gpu_streams);
    streams.reserve(num_gpu_streams);
        // We'll be scheduling (most of) our work in a round-robin fashion on all of
        // the streams, to prevent the GPU from idling.


    for (int i = 0; i < num_gpu_streams; ++i) {
        auto input_buffers = stream_input_buffer_set{
            cuda::memory::device::make_unique< compressed::ship_date_t[]      >(cuda_device, num_records_per_scheduled_kernel),
            cuda::memory::device::make_unique< compressed::discount_t[]       >(cuda_device, num_records_per_scheduled_kernel),
            cuda::memory::device::make_unique< compressed::extended_price_t[] >(cuda_device, num_records_per_scheduled_kernel),
            cuda::memory::device::make_unique< compressed::tax_t[]            >(cuda_device, num_records_per_scheduled_kernel),
            cuda::memory::device::make_unique< compressed::quantity_t[]       >(cuda_device, num_records_per_scheduled_kernel),
            cuda::memory::device::make_unique< bit_container_t[]              >(cuda_device, div_rounding_up(num_records_per_scheduled_kernel, return_flag_values_per_container)),
            cuda::memory::device::make_unique< bit_container_t[]              >(cuda_device, div_rounding_up(num_records_per_scheduled_kernel, line_status_values_per_container))
        };
        stream_input_buffer_sets.emplace_back(std::move(input_buffers));
        auto stream = cuda_device.create_stream(cuda::stream::async);
        streams.emplace_back(std::move(stream));
    }

    // You can't measure this from inside the process - without events, which
    // double copy_time = 0;
    // double computation_time = 0;

    // This only works for the overall time, not for anything else, so it's not a good idea:
     std::ofstream results_file;
     results_file.open("results.csv", std::ios::out);

    for(int run_index = 0; run_index < num_query_execution_runs; run_index++) {
        cout << "Executing query, run " << run_index + 1 << " of " << num_query_execution_runs << "... " << flush;
        auto start = timer::now();
        
           auto gpu_end_offset = cardinality; 
        if (use_coprocessing) {
             cpu->Clear();
             // Split the work between the CPU and the GPU at 50% each
             // TODO: 
             // - Double-check the choice of alignment here
             // - The parameters here are weird
             auto cpu_start_offset = cardinality / 2;
             cpu_start_offset = cardinality - cardinality % num_records_per_scheduled_kernel;
             auto num_records_for_cpu_to_process = cardinality - cpu_start_offset;
             (*cpu)(cpu_start_offset, num_records_for_cpu_to_process);
             gpu_end_offset = cpu_start_offset;
        } 

        // Initialize the aggregates; perhaps we should do this in a single kernel? ... probably not worth it
        streams[0].enqueue.memset(aggregates_on_device.sum_quantity.get(),         0, num_potential_groups * sizeof(sum_quantity_t));
        streams[0].enqueue.memset(aggregates_on_device.sum_base_price.get(),       0, num_potential_groups * sizeof(sum_base_price_t));
        streams[0].enqueue.memset(aggregates_on_device.sum_discounted_price.get(), 0, num_potential_groups * sizeof(sum_discounted_price_t));
        streams[0].enqueue.memset(aggregates_on_device.sum_charge.get(),           0, num_potential_groups * sizeof(sum_charge_t));
        streams[0].enqueue.memset(aggregates_on_device.sum_discount.get(),         0, num_potential_groups * sizeof(sum_discount_t));
        streams[0].enqueue.memset(aggregates_on_device.record_count.get(),         0, num_potential_groups * sizeof(cardinality_t));

        cuda::event_t aggregates_initialized_event = cuda_device.create_event(
                cuda::event::sync_by_blocking, cuda::event::dont_record_timings, cuda::event::not_interprocess);
        streams[0].enqueue.event(aggregates_initialized_event);
        for (int i = 1; i < num_gpu_streams; ++i) {
            streams[i].enqueue.wait(aggregates_initialized_event);
            // The other streams also require the aggregates to be initialized before doing any work
        }
        auto stream_index = 0;
        for (size_t offset_in_table = 0;
             offset_in_table < gpu_end_offset;
             offset_in_table += num_records_per_scheduled_kernel,
             stream_index = (stream_index+1) % num_gpu_streams) {

            auto num_records_for_this_launch = std::min<cardinality_t>(num_records_per_scheduled_kernel, gpu_end_offset - offset_in_table);
            auto num_return_flag_bit_containers_for_this_launch = div_rounding_up(num_records_for_this_launch, return_flag_values_per_container);
            cout << "num_return_flag_bit_containers_for_this_launch = " << num_return_flag_bit_containers_for_this_launch << endl;
            auto num_line_status_bit_containers_for_this_launch = div_rounding_up(num_records_for_this_launch, line_status_values_per_container);

            // auto start_copy = timer::now();  // This can't work, since copying is asynchronous.
            auto& stream = streams[stream_index];
            auto& stream_input_buffers = stream_input_buffer_sets[stream_index];
            stream.enqueue.copy(stream_input_buffers.ship_date.get()     , compressed_ship_date.get()      + offset_in_table, num_records_for_this_launch * sizeof(compressed::ship_date_t));
            stream.enqueue.copy(stream_input_buffers.discount.get()     , compressed_discount.get()       + offset_in_table, num_records_for_this_launch * sizeof(compressed::discount_t));
            stream.enqueue.copy(stream_input_buffers.extended_price.get(), compressed_extended_price.get() + offset_in_table, num_records_for_this_launch * sizeof(compressed::extended_price_t));
            stream.enqueue.copy(stream_input_buffers.tax.get()          , compressed_tax.get()            + offset_in_table, num_records_for_this_launch * sizeof(compressed::tax_t));
            stream.enqueue.copy(stream_input_buffers.quantity.get()     , compressed_quantity.get()       + offset_in_table, num_records_for_this_launch * sizeof(compressed::quantity_t));
            stream.enqueue.copy(stream_input_buffers.return_flag.get()   , compressed_return_flag.get()    + offset_in_table / return_flag_values_per_container, num_return_flag_bit_containers_for_this_launch * sizeof(bit_container_t));
            stream.enqueue.copy(stream_input_buffers.line_status.get()   , compressed_line_status.get()    + offset_in_table / line_status_values_per_container, num_line_status_bit_containers_for_this_launch * sizeof(bit_container_t));

            auto num_blocks = div_rounding_up(num_records_for_this_launch, num_threads_per_block);
            auto launch_config = cuda::make_launch_config(num_blocks, num_threads_per_block);
            (void) launch_config;

            stream.enqueue.kernel_launch(
                cuda::thread_local_tpchQ01_small_datatypes_coalesced,
                launch_config,
                aggregates_on_device.sum_quantity.get(),
                aggregates_on_device.sum_base_price.get(),
                aggregates_on_device.sum_discounted_price.get(),
                aggregates_on_device.sum_charge.get(),
                aggregates_on_device.sum_discount.get(),
                aggregates_on_device.record_count.get(),
                stream_input_buffers.ship_date.get(),
                stream_input_buffers.discount.get(),
                stream_input_buffers.extended_price.get(),
                stream_input_buffers.tax.get(),
                stream_input_buffers.quantity.get(),
                stream_input_buffers.return_flag.get(),
                stream_input_buffers.line_status.get(),
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
        streams[0].enqueue.copy(aggregates_on_host.record_count.get(),         aggregates_on_device.record_count.get(),         num_potential_groups * sizeof(cardinality_t));

        streams[0].synchronize();

        if (cpu) {
            cpu->wait();

/*            // merge
            int group_order[4];
            if (apply_compression) {
                group_order[0] = 6;
                group_order[1] = 4;
                group_order[2] = 0;
                group_order[3] = 5;
            } else {
                group_order[0] = magic_hash('A', 'F');
                group_order[1] = magic_hash('N', 'F');
                group_order[2] = magic_hash('N', 'O');
                group_order[3] = magic_hash('R', 'F');
            }
*/
            size_t idx = 0;
            for (size_t i=0; i<num_potential_groups; i++) {
                auto& e = cpu->table[i];
                if (e.count <= 0) {
                    continue;
                }
/*
                auto group = group_order[idx];

                #define B(i)  aggrs0[group].i += e.i; printf("set %s group %d  parti %d\n", #i, group, e.i)

                B(sum_quantity);
                B(count);
                B(sum_base_price);
                B(sum_disc);
                B(sum_disc_price);
                B(sum_charge);
*/
                idx++;
            }
            assert_always(idx == 4);
        }

        auto end = timer::now();
        std::chrono::duration<double> duration(end - start);
        cout << "done." << endl;
        results_file << duration.count() << '\n';
    }

    cuda::profiling::stop();

    if (num_query_execution_runs == 1) {
        cout << "\n"
                "+--------------------------------------------------- Results ---------------------------------------------------+\n";
        cout << "|  LS | RF | sum_quantity        | sum_base_price      | sum_disc_price      | sum_charge          | count      |\n";
        cout << "+---------------------------------------------------------------------------------------------------------------+\n";
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
                        assert(aggregates_on_host.sum_quantity[group] == 3773410700);
                        assert(aggregates_on_host.record_count[group] == 1478493);
                    }
                } else if (rf == 'N' and ls == 'F') {
                    if (cardinality == 6001215) {
                        assert(aggregates_on_host.sum_quantity[group] == 99141700);
                        assert(aggregates_on_host.record_count[group] == 38854);
                    }
                } else if (rf == 'N' and ls == 'O') {
                    rf = 'N';
                    ls = 'O';
                    if (cardinality == 6001215) {
                        assert(aggregates_on_host.sum_quantity[group] == 7447604000);
                        assert(aggregates_on_host.record_count[group] == 2920374);
                    }
                } else if (rf == 'R' and ls == 'F') {
                    if (cardinality == 6001215) {
                        assert(aggregates_on_host.sum_quantity[group]== 3771975300);
                        assert(aggregates_on_host.record_count[group]== 1478870);
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

        cout << "+---------------------------------------------------------------------------------------------------------------+\n";
    }
    results_file.close();
}
