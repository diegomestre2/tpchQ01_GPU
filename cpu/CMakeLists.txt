cmake_minimum_required(VERSION 2.6.2)

SET (CMAKE_CXX_STANDARD 14)

find_package(Threads)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/CMakeModules/")

ADD_DEFINITIONS(
  -std=c++14
  -march=native
  -O3
  # -pedantic -Wall
  -ftree-vectorize -ftree-slp-vectorize -funroll-all-loops -fpeel-loops -fomit-frame-pointer 
  -g
)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)


find_package(Numa REQUIRED)
include_directories(${NUMA_INCLUDE_DIR})

# 
find_package(Boost 1.45.0 COMPONENTS filesystem)

include_directories( ${Boost_INCLUDE_DIRS} )

add_executable(q1 q1.cpp tpch_kit.cpp vectorized.cpp common.cpp monetdb.cpp)
target_link_libraries(q1 ${CMAKE_THREAD_LIBS_INIT} ${NUMA_LIBRARY} ${PAPI_LIBRARIES} ${NUMA_LIBRARY})
