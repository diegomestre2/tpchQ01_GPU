#include <iostream>
#include <cuda/api_wrappers.h>
#include <vector>
#include <iomanip>
#include <fstream>

#include "kernel.hpp"
#include "kernels/naive.hpp"
#include "kernels/local.hpp"
#include "kernels/global.hpp"
#include "kernels/coalesced.hpp"
#include "../expl_comp_strat/tpch_kit.hpp"

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
    system(command.c_str());
}


int MAX_TUPLES_PER_STREAM = 32 * 1024;
int VALUES_PER_THREAD = 64;
int THREADS_PER_BLOCK = 32;

#define GIGA (1024 * 1024 * 1024)
#define MEGA (1024 * 1024)
#define KILO (1024)

using timer = std::chrono::high_resolution_clock;

inline bool file_exists(const std::string& name) {
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

#define INITIALIZE_MEMORY(ptrfunc) { \
    auto _shipdate                       = ptrfunc< SHIPDATE_TYPE[]             >(cardinality);           \
    auto _discount                       = ptrfunc< DISCOUNT_TYPE[]             >(cardinality);           \
    auto _extendedprice                  = ptrfunc< EXTENDEDPRICE_TYPE[]        >(cardinality);           \
    auto _tax                            = ptrfunc< TAX_TYPE[]                  >(cardinality);           \
    auto _quantity                       = ptrfunc< QUANTITY_TYPE[]             >(cardinality);           \
    auto _returnflag                     = ptrfunc< RETURNFLAG_TYPE[]           >(cardinality);           \
    auto _linestatus                     = ptrfunc< LINESTATUS_TYPE[]           >(cardinality);           \
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
    auto _shipdate_small                 = ptrfunc< SHIPDATE_TYPE_SMALL[]       >(cardinality);           \
    auto _discount_small                 = ptrfunc< DISCOUNT_TYPE_SMALL[]       >(cardinality);           \
    auto _extendedprice_small            = ptrfunc< EXTENDEDPRICE_TYPE_SMALL[]  >(cardinality);           \
    auto _tax_small                      = ptrfunc< TAX_TYPE_SMALL[]            >(cardinality);           \
    auto _quantity_small                 = ptrfunc< QUANTITY_TYPE_SMALL[]       >(cardinality);           \
    auto _returnflag_small               = ptrfunc< RETURNFLAG_TYPE_SMALL[]     >((cardinality + 3) / 4); \
    auto _linestatus_small               = ptrfunc< LINESTATUS_TYPE_SMALL[]     >((cardinality + 7) / 8); \
    shipdate_small = _shipdate_small.get(); \
    discount_small = _discount_small.get(); \
    extendedprice_small = _extendedprice_small.get(); \
    tax_small = _tax_small.get(); \
    quantity_small = _quantity_small.get(); \
    returnflag_small = _returnflag_small.get(); \
    linestatus_small = _linestatus_small.get(); \
    _shipdate_small.release(); \
    _discount_small.release(); \
    _extendedprice_small.release(); \
    _tax_small.release(); \
    _quantity_small.release(); \
    _returnflag_small.release(); \
    _linestatus_small.release(); \
}

struct Stream {
    cudaStream_t stream;
    size_t id;
#define EXPAND(A) \
    A(shipdate, SHIPDATE_TYPE) \
    A(discount, DISCOUNT_TYPE) \
    A(eprice, EXTENDEDPRICE_TYPE) \
    A(tax, TAX_TYPE) \
    A(quantity, QUANTITY_TYPE) \
    A(rf, RETURNFLAG_TYPE) \
    A(ls, LINESTATUS_TYPE)

#define DECLARE(name, type) type* name;
    EXPAND(DECLARE)
#undef DECLARE

    Stream(size_t size, size_t id) : id(id) {
        cudaStreamCreate(&stream);
#define ALLOC(name, type) cudaMalloc((void**)&name, size * sizeof(type));
    EXPAND(ALLOC)
#undef ALLOC

    }

    void Sync() {
        cudaStreamSynchronize(stream);
    }

    ~Stream() {
        Sync();
        cudaStreamDestroy(stream);
#define DEALLOC(name, type) cudaFree((void**)name);
    EXPAND(DEALLOC)
#undef DEALLOC

    }
};

struct StreamManager {
private:
    std::vector<Stream> streams;
    size_t pos;

public:
    StreamManager(size_t size, size_t max_streams) {
        streams.reserve(max_streams);
        for (size_t i=0; i<max_streams; i++) {
            streams.emplace_back(size, i);    
        }
        pos = 0;
    }

    Stream& GetNewStream() {
        return streams[pos++ % streams.size()];
    }
};


#define LOAD_BINARY(variable, tpe, fname) { \
        std::string fpath = join_path(tpch_directory, fname); \
        FILE* pFile = fopen(fpath.c_str(), "rb"); \
        variable = (tpe*) malloc(sizeof(tpe) * cardinality); \
        assert_always(variable && pFile); \
        fread(variable, sizeof(tpe), cardinality, pFile); \
        fclose(pFile); \
    }
    

#define WRITE_BINARY(variable, tpe, fname) { \
        std::string fpath = join_path(tpch_directory, fname); \
        FILE* pFile = fopen(fpath.c_str(), "wb+"); \
        assert_always(pFile); \
        fwrite(variable, sizeof(tpe), cardinality, pFile); \
        fclose(pFile); \
    }

void print_help() {
    fprintf(stderr, "Unrecognized command line option.\n");
    fprintf(stderr, "Usage: tpch_01 [args]\n");
    fprintf(stderr, "   --sf=[sf:1] (number, e.g. 0.01 - 100)\n");
    fprintf(stderr, "   --streams=[streams:8] (number, e.g. 1 - 64)\n");
    fprintf(stderr, "   --tuples-per-stream=[tuples:32768] (number, e.g. 16384 - 131072)\n");
    fprintf(stderr, "   --values-per-thread=[values:64] (number, e.g. 16 - 128)\n");
    fprintf(stderr, "   --threads-per-block=[threads:32] (number, e.g. 32 - 1024)\n");
    fprintf(stderr, "   --use-global-ht\n");
    fprintf(stderr, "   --no-pinned-memory\n");
    fprintf(stderr, "   --use-small-datatypes\n");
    fprintf(stderr, "   --use-coalescing\n");
}

#include "cpu.h"


GPUAggrHashTable aggrs0[MAX_GROUPS] ALIGN;

#define init_table(ag) memset(&aggrs##ag, 0, sizeof(aggrs##ag))
#define clear(x) memset(x, 0, sizeof(x))

extern "C" void
clear_tables()
{
    init_table(0);
}

#include "cpu/common.hpp"

int main(int argc, char** argv) {
    /* load data */
    auto start_csv = timer::now();
    SHIPDATE_TYPE* _shipdate;
    RETURNFLAG_TYPE* _returnflag;
    LINESTATUS_TYPE* _linestatus;
    DISCOUNT_TYPE* _discount;
    TAX_TYPE* _tax;
    EXTENDEDPRICE_TYPE* _extendedprice;
    QUANTITY_TYPE* _quantity;
    size_t cardinality;

    bool USE_PINNED_MEMORY = true;
    bool USE_GLOBAL_HT = false;
    bool USE_SMALL_DATATYPES = false;
    bool USE_COALESCING = false;
    bool USE_COPROCESSING = true;

    double sf = 1;
    int nr_streams = 8;
    int nruns = 5;
    std::string sf_argument = "--sf=";
    std::string streams_argument = "--streams=";
    std::string tuples_per_stream_argument = "--tuples-per-stream=";
    std::string values_per_thread_argument = "--values-per-thread=";
    std::string threads_per_block_argument = "--threads-per-block=";
    std::string nruns_argument = "--nruns=";
    for(int i = 1; i < argc; i++) {
        auto arg = std::string(argv[i]);
        if (arg == "--device") {
            get_device_properties();
            exit(1);
        } else if (arg == "--no-pinned-memory") {
            USE_PINNED_MEMORY = false;
        } else if (arg == "--use-global-ht") {
            USE_GLOBAL_HT = true;
        } else if (arg == "--use-small-datatypes") {
            USE_SMALL_DATATYPES = true;
        } else if (arg == "--use-coalescing") {
            USE_COALESCING = true;
        } else if (arg == "--use-coprocessing") {
            USE_COPROCESSING = true;
        } else if (arg.substr(0, sf_argument.size()) == sf_argument) {
            sf = std::stod(arg.substr(sf_argument.size()));
        } else if (arg.substr(0, streams_argument.size()) == streams_argument) {
            nr_streams = std::stoi(arg.substr(streams_argument.size()));
        } else if (arg.substr(0, tuples_per_stream_argument.size()) == tuples_per_stream_argument) {
            MAX_TUPLES_PER_STREAM = std::stoi(arg.substr(tuples_per_stream_argument.size()));
        } else if (arg.substr(0, values_per_thread_argument.size()) == values_per_thread_argument) {
            VALUES_PER_THREAD = std::stoi(arg.substr(values_per_thread_argument.size()));
        } else if (arg.substr(0, threads_per_block_argument.size()) == threads_per_block_argument) {
            THREADS_PER_BLOCK = std::stoi(arg.substr(threads_per_block_argument.size()));
        } else if (arg.substr(0, nruns_argument.size()) == nruns_argument) {
            nruns = std::stoi(arg.substr(nruns_argument.size()));
        } else {
            print_help();
            exit(1);
        }
    }
    lineitem li((size_t)(7000000 * sf));
    syscall("mkdir -p tpch");
    std::string tpch_directory = join_path("tpch", std::to_string(sf));
    syscall(std::string("mkdir -p ") + tpch_directory);
    if (file_exists(join_path(tpch_directory, "shipdate.bin"))) {
        std::cout << "Loading from binary." << std::endl;
        // binary files exist, load them
        cardinality = filesize(join_path(tpch_directory, "shipdate.bin")) / sizeof(SHIPDATE_TYPE);
        LOAD_BINARY(_shipdate, SHIPDATE_TYPE, "shipdate.bin");
        LOAD_BINARY(_returnflag, RETURNFLAG_TYPE, "returnflag.bin");
        LOAD_BINARY(_linestatus, LINESTATUS_TYPE, "linestatus.bin");
        LOAD_BINARY(_discount, DISCOUNT_TYPE, "discount.bin");
        LOAD_BINARY(_tax, TAX_TYPE, "tax.bin");
        LOAD_BINARY(_extendedprice, EXTENDEDPRICE_TYPE, "extendedprice.bin");
        LOAD_BINARY(_quantity, QUANTITY_TYPE, "quantity.bin");

        #define A(name) li.l_##name.cardinality = cardinality; li.l_##name.m_ptr = _##name

        A(shipdate);
        A(returnflag);
        A(linestatus);
        A(discount);
        A(tax);
        A(extendedprice);
        A(quantity);
        
    } else {
        std::cout << "Reading CSV file and writing to binary." << std::endl;
        std::string input_file = join_path(tpch_directory, "lineitem.tbl");
        if (!file_exists(input_file.c_str())) {
            // have to generate lineitem file
            syscall("./genlineitem.sh " + std::to_string(sf));
            syscall("mv lineitem.tbl " + input_file);
        }
        li.FromFile(input_file.c_str());
        _shipdate = li.l_shipdate.get();
        _returnflag = li.l_returnflag.get();
        _linestatus = li.l_linestatus.get();
        _discount = li.l_discount.get();
        _tax = li.l_tax.get();
        _extendedprice = li.l_extendedprice.get();
        _quantity = li.l_quantity.get();
        cardinality = li.l_extendedprice.cardinality;
        WRITE_BINARY(_shipdate, SHIPDATE_TYPE, "shipdate.bin");
        WRITE_BINARY(_returnflag, RETURNFLAG_TYPE, "returnflag.bin");
        WRITE_BINARY(_linestatus, LINESTATUS_TYPE, "linestatus.bin");
        WRITE_BINARY(_discount, DISCOUNT_TYPE, "discount.bin");
        WRITE_BINARY(_tax, TAX_TYPE, "tax.bin");
        WRITE_BINARY(_extendedprice, EXTENDEDPRICE_TYPE, "extendedprice.bin");
        WRITE_BINARY(_quantity, QUANTITY_TYPE, "quantity.bin");
    }

    CoProc* cpu = nullptr;
    if (USE_COPROCESSING) {
        cpu = new CoProc(li);
    }

    auto end_csv = timer::now();

    auto size_per_tuple = sizeof(SHIPDATE_TYPE) + sizeof(DISCOUNT_TYPE) + sizeof(EXTENDEDPRICE_TYPE) + sizeof(TAX_TYPE) + sizeof(QUANTITY_TYPE) + sizeof(RETURNFLAG_TYPE) + sizeof(LINESTATUS_TYPE);
    if (USE_SMALL_DATATYPES) {
        size_per_tuple = sizeof(SHIPDATE_TYPE_SMALL) + sizeof(DISCOUNT_TYPE_SMALL) + sizeof(EXTENDEDPRICE_TYPE_SMALL) + sizeof(TAX_TYPE_SMALL) + sizeof(QUANTITY_TYPE_SMALL) + sizeof(RETURNFLAG_TYPE_SMALL) + sizeof(LINESTATUS_TYPE_SMALL);
    }
    auto start_preprocess = timer::now();

    SHIPDATE_TYPE* shipdate;
    DISCOUNT_TYPE* discount;
    EXTENDEDPRICE_TYPE* extendedprice;
    TAX_TYPE* tax;
    QUANTITY_TYPE* quantity;
    RETURNFLAG_TYPE* returnflag;
    LINESTATUS_TYPE* linestatus;

    SHIPDATE_TYPE_SMALL* shipdate_small;
    DISCOUNT_TYPE_SMALL* discount_small;
    EXTENDEDPRICE_TYPE_SMALL* extendedprice_small;
    TAX_TYPE_SMALL* tax_small;
    QUANTITY_TYPE_SMALL* quantity_small;
    RETURNFLAG_TYPE_SMALL* returnflag_small;
    LINESTATUS_TYPE_SMALL* linestatus_small;
    if (USE_PINNED_MEMORY) {
        INITIALIZE_MEMORY(cuda::memory::host::make_unique);
    } else {
        INITIALIZE_MEMORY(std::make_unique);
    }

    for(size_t i = 0; i < cardinality; i++) {
        shipdate[i] = _shipdate[i];
        discount[i] = _discount[i];
        extendedprice[i] = _extendedprice[i];
        quantity[i] = _quantity[i];
        tax[i] = _tax[i];
        returnflag[i] = _returnflag[i];
        linestatus[i] = _linestatus[i];

        shipdate_small[i]      = shipdate[i] - SHIPDATE_MIN;
        discount_small[i]      = discount[i];
        extendedprice_small[i] = extendedprice[i];
        quantity_small[i]      = quantity[i] / 100;
        tax_small[i]           = tax[i];
        if (i % 4 == 0) {
            returnflag_small[i / 4] = 0;
            for(size_t j = 0; j < std::min((size_t) 4, cardinality - i); j++) {
                // 'N' = 0x00, 'R' = 0x01, 'A' = 0x10
                returnflag_small[i / 4] |= 
                    (_returnflag[i + j] == 'N' ? 0x00 : (_returnflag[i + j] == 'R' ? 0x01 : 0x02)) << (j * 2);
            }
        }
        if (i % 8 == 0) {
            linestatus_small[i / 8] = 0;
            for(size_t j = 0; j < std::min((size_t) 8, cardinality - i); j++) {
                // 'O' = 0, 'F' = 1
                linestatus_small[i / 8] |= (_linestatus[i + j] == 'F' ? 1 : 0) << j;
            }
        }

        assert_always((int)shipdate_small[i]           == shipdate[i] - SHIPDATE_MIN);
        assert_always((int64_t) discount_small[i]      == discount[i]);
        assert_always((int64_t) extendedprice_small[i] == extendedprice[i]);
        assert_always((int64_t) quantity_small[i]      == quantity[i] / 100);
        assert_always((int64_t) tax_small[i]           == tax[i]);
    }

    constexpr uint8_t RETURNFLAG_MASK[] = { 0x03, 0x0C, 0x30, 0xC0 };
    constexpr uint8_t LINESTATUS_MASK[] = { 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80 };
    for(size_t i = 0; i < cardinality; i++) {
        uint8_t retflag = (returnflag_small[i / 4] & RETURNFLAG_MASK[i % 4]) >> (2 * (i % 4));
        uint8_t lstatus = (linestatus_small[i / 8] & LINESTATUS_MASK[i % 8]) >> (i % 8);
        assert_always(retflag == (returnflag[i] == 'N' ? 0x00 : (returnflag[i] == 'R' ? 0x01 : 0x02)));
        assert_always(lstatus == (linestatus[i] == 'F' ? 1 : 0));
    }
    auto end_preprocess = timer::now();

    assert(cardinality > 0 && "Prevent BS exception");
    const size_t data_length = cardinality / 2;
    clear_tables();

    /* Allocate memory on device */
    auto current_device = cuda::device::current::get();
    auto d_aggregations  = cuda::memory::device::make_unique< GPUAggrHashTable[]      >(current_device, MAX_GROUPS);

    StreamManager streams(MAX_TUPLES_PER_STREAM, nr_streams);
    cuda_check_error();
    std::ofstream myfile;
    myfile.open("results.csv", std::ios::out);
    assert_always(myfile.is_open());
    for(int k = 0; k < nruns + 1; k++) {
        cudaMemset(d_aggregations.get(), 0, sizeof(GPUAggrHashTable)*MAX_GROUPS);

        double copy_time = 0;
        double computation_time = 0;

        size_t offset = 0;
        auto start = timer::now();

        if (cpu) {
            (*cpu)(data_length, cardinality - data_length);
        }        

        while (offset < data_length) {
            size_t size = std::min((size_t) MAX_TUPLES_PER_STREAM, (size_t) (data_length - offset));

            auto& stream = streams.GetNewStream();
            auto& s = stream;

            if (USE_SMALL_DATATYPES) {
                cuda::memory::async::copy(stream.shipdate, shipdate_small      + offset, size * sizeof(SHIPDATE_TYPE_SMALL),     stream.stream);
                cuda::memory::async::copy(stream.discount, discount_small      + offset, size * sizeof(DISCOUNT_TYPE_SMALL), stream.stream);
                cuda::memory::async::copy(stream.eprice, extendedprice_small   + offset, size * sizeof(EXTENDEDPRICE_TYPE_SMALL), stream.stream);
                cuda::memory::async::copy(stream.tax, tax_small                + offset, size * sizeof(TAX_TYPE_SMALL), stream.stream);
                cuda::memory::async::copy(stream.quantity, quantity_small      + offset, size * sizeof(QUANTITY_TYPE_SMALL), stream.stream);
                cuda::memory::async::copy(stream.rf, returnflag_small          + offset / 4, (size * sizeof(RETURNFLAG_TYPE_SMALL) + 3) / 4,    stream.stream);
                cuda::memory::async::copy(stream.ls, linestatus_small          + offset / 8, (size * sizeof(LINESTATUS_TYPE_SMALL) + 7) / 8,    stream.stream);
            } else {
                cuda::memory::async::copy(stream.shipdate, shipdate      + offset, size * sizeof(SHIPDATE_TYPE),     stream.stream);
                cuda::memory::async::copy(stream.discount, discount      + offset, size * sizeof(DISCOUNT_TYPE), stream.stream);
                cuda::memory::async::copy(stream.eprice, extendedprice   + offset, size * sizeof(EXTENDEDPRICE_TYPE), stream.stream);
                cuda::memory::async::copy(stream.tax, tax                + offset, size * sizeof(TAX_TYPE), stream.stream);
                cuda::memory::async::copy(stream.quantity, quantity      + offset, size * sizeof(QUANTITY_TYPE), stream.stream);
                cuda::memory::async::copy(stream.rf, returnflag          + offset, size * sizeof(RETURNFLAG_TYPE),    stream.stream);
                cuda::memory::async::copy(stream.ls, linestatus          + offset, size * sizeof(LINESTATUS_TYPE),    stream.stream);
            }
            

            size_t amount_of_blocks = size / (VALUES_PER_THREAD * THREADS_PER_BLOCK) + 1;
            size_t SHARED_MEMORY = 0; //sizeof(AggrHashTableLocal) * 18 * THREADS_PER_BLOCK;
            if (!USE_GLOBAL_HT) {
                if (USE_SMALL_DATATYPES) {
                    if(USE_COALESCING){
                        cuda::thread_local_tpchQ01_small_datatypes_coalesced<<<amount_of_blocks, THREADS_PER_BLOCK, SHARED_MEMORY, s.stream>>>(
                            (SHIPDATE_TYPE_SMALL*) s.shipdate,
                            (DISCOUNT_TYPE_SMALL*) s.discount,
                            (EXTENDEDPRICE_TYPE_SMALL*) s.eprice,
                            (TAX_TYPE_SMALL*) s.tax,
                            (RETURNFLAG_TYPE_SMALL*)s.rf,
                            (LINESTATUS_TYPE_SMALL*)s.ls,
                            (QUANTITY_TYPE_SMALL*) s.quantity,
                            d_aggregations.get(),
                            (u64_t) size,
                            VALUES_PER_THREAD);
                    } else {
                        cuda::thread_local_tpchQ01_small_datatypes<<<amount_of_blocks, THREADS_PER_BLOCK, SHARED_MEMORY, s.stream>>>(
                            (SHIPDATE_TYPE_SMALL*) s.shipdate,
                            (DISCOUNT_TYPE_SMALL*) s.discount,
                            (EXTENDEDPRICE_TYPE_SMALL*) s.eprice,
                            (TAX_TYPE_SMALL*) s.tax,
                            (RETURNFLAG_TYPE_SMALL*)s.rf,
                            (LINESTATUS_TYPE_SMALL*)s.ls,
                            (QUANTITY_TYPE_SMALL*) s.quantity,
                            d_aggregations.get(),
                            (u64_t) size,
                            VALUES_PER_THREAD);
                    }
                } else {
                    if(USE_COALESCING){
                        cuda::thread_local_tpchQ01_coalesced<<<amount_of_blocks, THREADS_PER_BLOCK, SHARED_MEMORY, s.stream>>>(
                            s.shipdate,
                            s.discount,
                            s.eprice,
                            s.tax,
                            s.rf,
                            s.ls,
                            s.quantity,
                            d_aggregations.get(),
                            (u64_t) size,
                            VALUES_PER_THREAD);
                    } else {
                        cuda::thread_local_tpchQ01<<<amount_of_blocks, THREADS_PER_BLOCK, SHARED_MEMORY, s.stream>>>(
                            s.shipdate,
                            s.discount,
                            s.eprice,
                            s.tax,
                            s.rf,
                            s.ls,
                            s.quantity,
                            d_aggregations.get(),
                            (u64_t) size,
                            VALUES_PER_THREAD);
                    }
                }
            } else {
                if (USE_SMALL_DATATYPES) {
                    cuda::global_ht_tpchQ01_small_datatypes<<<amount_of_blocks, THREADS_PER_BLOCK, SHARED_MEMORY, s.stream>>>(
                        (SHIPDATE_TYPE_SMALL*) s.shipdate,
                        (DISCOUNT_TYPE_SMALL*) s.discount,
                        (EXTENDEDPRICE_TYPE_SMALL*) s.eprice,
                        (TAX_TYPE_SMALL*) s.tax,
                        (RETURNFLAG_TYPE_SMALL*)s.rf,
                        (LINESTATUS_TYPE_SMALL*)s.ls,
                        (QUANTITY_TYPE_SMALL*) s.quantity,
                        d_aggregations.get(),
                        (u64_t) size,
                        VALUES_PER_THREAD);
                } else {
                    cuda::global_ht_tpchQ01<<<amount_of_blocks, THREADS_PER_BLOCK, SHARED_MEMORY, s.stream>>>(
                        s.shipdate,
                        s.discount,
                        s.eprice,
                        s.tax,
                        s.rf,
                        s.ls,
                        s.quantity,
                        d_aggregations.get(),
                        (u64_t) size,
                        VALUES_PER_THREAD);
                }
            }

            offset += size;
        }
        cudaDeviceSynchronize();
        cuda::memory::copy(aggrs0, d_aggregations.get(), sizeof(GPUAggrHashTable)*MAX_GROUPS);

        if (cpu) {
            cpu->wait();

            // merge
            int group_order[4];
            if (USE_SMALL_DATATYPES) {
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

            size_t idx = 0;
            for (size_t i=0; i<MAX_GROUPS; i++) {
                auto& e = cpu->table[i];
                if (e.count <= 0) {
                    continue;
                }

                auto group = group_order[idx];

                #define B(i)  aggrs0[group].i += e.i; printf("set %s group %d  old %d parti %d\n", #i, group, aggrs0[group].i, e.i)

                B(sum_quantity);
                B(count);
                B(sum_base_price);
                B(sum_disc);
                B(sum_disc_price);
                B(sum_charge);

                idx++;
            }

            assert_always(idx == 4);
        }

        auto end = timer::now();  
        cuda_check_error();
        if (k > 0) {
            std::chrono::duration<double> duration(end - start);
            myfile << duration.count() << std::endl;
        }
        if (k == 1) {
            std::cout << "\n"
                         "+--------------------------------------------------- Results ---------------------------------------------------+\n";
            std::cout << "|  LS | RF | sum_quantity        | sum_base_price      | sum_disc_price      | sum_charge          | count      |\n";
            std::cout << "+---------------------------------------------------------------------------------------------------------------+\n";
            auto print_dec = [] (auto s, auto x) { printf("%s%16ld.%02ld", s, Decimal64::GetInt(x), Decimal64::GetFrac(x)); };
            // A/F - N/F - N/O, R/F
            int group_order[4];
            if (USE_SMALL_DATATYPES) {
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
            for (size_t idx=0; idx < 4; idx++) {
                int group = group_order[idx];
                if (aggrs0[group].count > 0) {
                    size_t i = group;
                    char rf = '-', ls = '-';
                    if (idx == 0) { // A, F = 2 + 4
                        rf = 'A';
                        ls = 'F';
                        if (cardinality == 6001215) {
                            assert(aggrs0[i].count == 1478493);
                            assert(aggrs0[i].sum_quantity == 3773410700);
                            
                        }
                    } else if (idx == 1) { // N, F = 0 + 4
                        rf = 'N';
                        ls = 'F';
                        if (cardinality == 6001215) {
                            assert(aggrs0[i].count == 38854);
                            assert(aggrs0[i].sum_quantity == 99141700);
                            
                        }
                    } else if (idx == 2) { // N, O = 0 + 0
                        rf = 'N';
                        ls = 'O';
                        if (cardinality == 6001215) {
                            assert(aggrs0[i].sum_quantity == 7447604000);
                            assert(aggrs0[i].count == 2920374);
                        }
                    } else if (idx == 3) { // R, F = 1 + 4
                        rf = 'R';
                        ls = 'F';
                        if (cardinality == 6001215) {
                            assert(aggrs0[i].sum_quantity == 3771975300);
                            assert(aggrs0[i].count == 1478870);
                        }
                    } else {
                        printf("%d\n", group);
                    }

                    printf("| # %c | %c ", rf, ls);
                    print_dec(" | ",  aggrs0[i].sum_quantity);
                    print_dec(" | ",  aggrs0[i].sum_base_price);
                    print_dec(" | ",  aggrs0[i].sum_disc_price);
                    print_dec(" | ",  aggrs0[i].sum_charge);
                    printf(" | %10llu |\n", aggrs0[i].count);
                }
            }
            std::cout << "+---------------------------------------------------------------------------------------------------------------+\n";

            uint64_t cache_line_size = 128; // bytes
            uint64_t num_loads =  1478493 + 38854 + 2920374 + 1478870 + 6;
            uint64_t num_stores = 19;
            std::chrono::duration<double> duration(end - start);
            uint64_t tuples_per_second               = static_cast<uint64_t>(data_length / duration.count());
            double effective_memory_throughput       = static_cast<double>((tuples_per_second * size_per_tuple) / GIGA);
            double estimated_memory_throughput       = static_cast<double>((tuples_per_second * cache_line_size) / GIGA);
            double effective_memory_throughput_read  = static_cast<double>((tuples_per_second * size_per_tuple) / GIGA);
            double effective_memory_throughput_write = static_cast<double>(tuples_per_second / (size_per_tuple * GIGA));
            double theretical_memory_bandwidth       = static_cast<double>((5505 * 10e06 * (352 / 8) * 2) / 10e09);
            double efective_memory_bandwidth         = static_cast<double>(((data_length * sizeof(SHIPDATE_TYPE)) + (num_loads * size_per_tuple) + (num_loads * num_stores))  / (duration.count() * 10e09));
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
    }
    myfile.close();
}
