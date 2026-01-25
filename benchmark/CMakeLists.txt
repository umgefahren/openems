cmake_minimum_required(VERSION 3.14)
project(fdtd_benchmark CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable optimizations for release builds
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -fsanitize=address")

# Default to Release build
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Find threads
find_package(Threads REQUIRED)

# Basic benchmark without Highway
add_executable(fdtd_benchmark fdtd_benchmark.cpp)
target_link_libraries(fdtd_benchmark PRIVATE Threads::Threads)

# Optional: Highway-enabled benchmark
option(WITH_HIGHWAY "Build with Google Highway SIMD library" OFF)

if(WITH_HIGHWAY)
    find_package(hwy CONFIG)
    if(hwy_FOUND)
        message(STATUS "Highway found, building Highway-enabled benchmark")
        add_executable(fdtd_benchmark_hwy fdtd_benchmark_hwy.cpp)
        target_link_libraries(fdtd_benchmark_hwy PRIVATE hwy::hwy Threads::Threads)
        target_compile_definitions(fdtd_benchmark_hwy PRIVATE HWY_ENABLED)
    else()
        message(WARNING "Highway not found. Install with: apt install libhwy-dev")
    endif()
endif()

# Print build info
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CXX flags: ${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE}}")
