# Benchmark build configuration for OpenEMS FDTD performance testing
cmake_minimum_required(VERSION 3.10)

project(openems_benchmarks CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Google Highway
find_package(hwy QUIET)

# Optimization flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")

# Basic and SSE benchmark (no external dependencies)
add_executable(fdtd_benchmark fdtd_benchmark.cpp)
target_compile_options(fdtd_benchmark PRIVATE -O3 -march=native -msse4.2)

# Highway benchmarks (require libhwy)
if(hwy_FOUND)
    message(STATUS "Google Highway found - building Highway benchmarks")

    add_executable(fdtd_benchmark_highway fdtd_benchmark_highway.cpp)
    target_link_libraries(fdtd_benchmark_highway hwy::hwy)
    target_compile_options(fdtd_benchmark_highway PRIVATE -O3 -march=native)

    add_executable(fdtd_benchmark_all fdtd_benchmark_all.cpp)
    target_link_libraries(fdtd_benchmark_all hwy::hwy)
    target_compile_options(fdtd_benchmark_all PRIVATE -O3 -march=native)
else()
    message(STATUS "Google Highway not found - skipping Highway benchmarks")
    message(STATUS "Install with: apt install libhwy-dev")
endif()
