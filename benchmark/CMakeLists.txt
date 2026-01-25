cmake_minimum_required(VERSION 3.10)
project(fdtd_benchmark CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Enable SSE/AVX based on architecture
include(CheckCXXCompilerFlag)

check_cxx_compiler_flag("-msse2" COMPILER_SUPPORTS_SSE2)
check_cxx_compiler_flag("-mavx" COMPILER_SUPPORTS_AVX)
check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
check_cxx_compiler_flag("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)

if(COMPILER_SUPPORTS_MARCH_NATIVE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
elseif(COMPILER_SUPPORTS_AVX2)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma")
elseif(COMPILER_SUPPORTS_AVX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx")
elseif(COMPILER_SUPPORTS_SSE2)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse2")
endif()

# Find OpenMP
find_package(OpenMP)

# Add executables
add_executable(fdtd_benchmark fdtd_benchmark.cpp)
add_executable(fdtd_optimized fdtd_optimized.cpp)

# Enable more optimization flags
target_compile_options(fdtd_benchmark PRIVATE
    -ffast-math
    -funroll-loops
)

target_compile_options(fdtd_optimized PRIVATE
    -ffast-math
    -funroll-loops
)

# Link OpenMP if available
if(OpenMP_CXX_FOUND)
    target_link_libraries(fdtd_optimized OpenMP::OpenMP_CXX)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CXX flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "CXX flags release: ${CMAKE_CXX_FLAGS_RELEASE}")
message(STATUS "OpenMP found: ${OpenMP_CXX_FOUND}")
