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

# Weld
set(WELD_HOME "~/weld/weld/")
set(WELD_INC "${WELD_HOME}/c")
set(WELD_LIB "-lweld")

include_directories(${WELD_INC})
link_directories(${WELD_HOME}/target/release/)
link_directories(${WELD_HOME}/weld_rt/target/release/)

# 
find_package(Boost 1.45.0 COMPONENTS filesystem)

include_directories( ${Boost_INCLUDE_DIRS} )

add_executable(q1 q1.cpp tpch_kit.cpp vectorized.cpp common.cpp monetdb.cpp)
add_executable(aggr aggr.cpp aggr_2.cpp aggr_3.cpp aggr_4.cpp tpch_kit.cpp vectorized.cpp common.cpp monetdb.cpp)
target_link_libraries(q1 ${CMAKE_THREAD_LIBS_INIT} ${NUMA_LIBRARY} ${PAPI_LIBRARIES} ${WELD_LIB})