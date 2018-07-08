#include "data_types.h"
#include "constants.hpp"
#include "bit_operations.hpp"

#if __cplusplus >= 201703L
#include <filesystem>
using namespace filesystem = std::filesystem;
#elif __cplusplus >= 201402L
#include <experimental/filesystem>
namespace filesystem = std::experimental::filesystem;
#else
#error This code must be compiled using the C++14 language started or later
#endif

#include <iostream>
#include <fstream>
#include <ios>

template <typename UniquePtr>
void load_column_from_binary_file(
    UniquePtr&               buffer,
    cardinality_t            cardinality,
    const filesystem::path&  directory,
    const std::string&       file_name)
{
    // TODO: C++'ify the file access (will also guarantee exception safety)
    using raw_ptr_type = typename std::decay<decltype(buffer.get())>::type;
    using element_type = typename std::remove_pointer<raw_ptr_type>::type;
    auto file_path = directory / file_name;
    buffer = std::make_unique<element_type[]>(cardinality);
    std::cout << "Loading a column from " << file_path << " ... " << std::flush;
    FILE* pFile = fopen(file_path.c_str(), "rb");
    if (pFile == nullptr) { throw std::runtime_error("Failed opening file " + file_path.string()); }
    auto num_elements_read = fread(buffer.get(), sizeof(element_type), cardinality, pFile);
    if (num_elements_read != cardinality) {
        throw std::runtime_error("Failed reading sufficient data from " +
            file_path.string() + " : expected " + std::to_string(cardinality) + " elements but read only " + std::to_string(num_elements_read) + "."); }
    fclose(pFile);
    std::cout << "done." << std::endl;
}

template <typename T>
void write_column_to_binary_file(
    const T*                buffer,
    cardinality_t           cardinality,
    const filesystem::path& directory,
    const std::string&      file_name)
{
    auto file_path = directory / file_name;
    std::cout << "Writing a column to " << file_path << " ... " << std::flush;
    FILE* pFile = fopen(file_path.c_str(), "wb+");
    if (pFile == nullptr) { throw std::runtime_error("Failed opening file " + file_path.string()); }
    auto num_elements_written = fwrite(buffer, sizeof(T), cardinality, pFile);
    fclose(pFile);
    if (num_elements_written != cardinality) {
        remove(file_path.c_str());
        throw std::runtime_error("Failed writing all elements to the file - only " +
            std::to_string(num_elements_written) + " written: " + strerror(errno));
    }
    std::cout << "done." << std::endl;
}
