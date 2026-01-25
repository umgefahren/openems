/*
 * FDTD Engine Comprehensive Benchmark
 * Compares: Basic (scalar), SSE (4 floats), Highway/AVX-512 (16 floats)
 */

#include <iostream>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <iomanip>

// Highway includes
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "fdtd_benchmark_all.cpp"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace fdtd_all {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

// Grid dimensions - use multiple of 16 for AVX-512 alignment
constexpr unsigned int NX = 100;
constexpr unsigned int NY = 100;
constexpr unsigned int NZ = 128;
constexpr unsigned int NUM_VECTORS_SSE = (NZ + 3) / 4;
constexpr unsigned int TIMESTEPS = 1000;

// ============== Basic Engine (Scalar) ==============
#ifdef __GNUC__
typedef float v4sf __attribute__ ((vector_size (16)));
union f4vector {
    v4sf v;
    float f[4];
};
#else
#include <emmintrin.h>
union f4vector {
    __m128 v;
    float f[4];
};
inline __m128 operator + (__m128 a, __m128 b) { return _mm_add_ps(a, b); }
inline __m128 operator - (__m128 a, __m128 b) { return _mm_sub_ps(a, b); }
inline __m128 operator * (__m128 a, __m128 b) { return _mm_mul_ps(a, b); }
inline __m128& operator *= (__m128& a, __m128 b) { a = a * b; return a; }
inline __m128& operator += (__m128& a, __m128 b) { a = a + b; return a; }
#endif

// Basic arrays
float**** volt_basic = nullptr;
float**** curr_basic = nullptr;
float**** vv_basic = nullptr;
float**** vi_basic = nullptr;
float**** ii_basic = nullptr;
float**** iv_basic = nullptr;

// SSE arrays
f4vector**** volt_sse = nullptr;
f4vector**** curr_sse = nullptr;
f4vector**** vv_sse = nullptr;
f4vector**** vi_sse = nullptr;
f4vector**** ii_sse = nullptr;
f4vector**** iv_sse = nullptr;

// Highway arrays (flat)
float* volt_hwy = nullptr;
float* curr_hwy = nullptr;
float* vv_hwy = nullptr;
float* vi_hwy = nullptr;
float* ii_hwy = nullptr;
float* iv_hwy = nullptr;

// ============== Basic Engine Functions ==============
float**** allocate_basic_array(unsigned int nx, unsigned int ny, unsigned int nz) {
    float**** arr = new float***[3];
    for (int n = 0; n < 3; n++) {
        arr[n] = new float**[nx];
        for (unsigned int x = 0; x < nx; x++) {
            arr[n][x] = new float*[ny];
            for (unsigned int y = 0; y < ny; y++) {
                arr[n][x][y] = new float[nz];
                std::memset(arr[n][x][y], 0, nz * sizeof(float));
            }
        }
    }
    return arr;
}

void free_basic_array(float**** arr, unsigned int nx, unsigned int ny) {
    for (int n = 0; n < 3; n++) {
        for (unsigned int x = 0; x < nx; x++) {
            for (unsigned int y = 0; y < ny; y++) {
                delete[] arr[n][x][y];
            }
            delete[] arr[n][x];
        }
        delete[] arr[n];
    }
    delete[] arr;
}

void init_basic_arrays() {
    volt_basic = allocate_basic_array(NX, NY, NZ);
    curr_basic = allocate_basic_array(NX, NY, NZ);
    vv_basic = allocate_basic_array(NX, NY, NZ);
    vi_basic = allocate_basic_array(NX, NY, NZ);
    ii_basic = allocate_basic_array(NX, NY, NZ);
    iv_basic = allocate_basic_array(NX, NY, NZ);

    for (int n = 0; n < 3; n++) {
        for (unsigned int x = 0; x < NX; x++) {
            for (unsigned int y = 0; y < NY; y++) {
                for (unsigned int z = 0; z < NZ; z++) {
                    vv_basic[n][x][y][z] = 0.99f;
                    vi_basic[n][x][y][z] = 0.01f;
                    ii_basic[n][x][y][z] = 0.99f;
                    iv_basic[n][x][y][z] = 0.01f;
                }
            }
        }
    }
    volt_basic[0][NX/2][NY/2][NZ/2] = 1.0f;
}

void cleanup_basic_arrays() {
    free_basic_array(volt_basic, NX, NY);
    free_basic_array(curr_basic, NX, NY);
    free_basic_array(vv_basic, NX, NY);
    free_basic_array(vi_basic, NX, NY);
    free_basic_array(ii_basic, NX, NY);
    free_basic_array(iv_basic, NX, NY);
}

void UpdateVoltages_Basic() {
    for (unsigned int x = 1; x < NX-1; x++) {
        for (unsigned int y = 1; y < NY-1; y++) {
            for (unsigned int z = 1; z < NZ-1; z++) {
                volt_basic[0][x][y][z] *= vv_basic[0][x][y][z];
                volt_basic[0][x][y][z] += vi_basic[0][x][y][z] * (
                    curr_basic[2][x][y][z] - curr_basic[2][x][y-1][z] -
                    curr_basic[1][x][y][z] + curr_basic[1][x][y][z-1]);

                volt_basic[1][x][y][z] *= vv_basic[1][x][y][z];
                volt_basic[1][x][y][z] += vi_basic[1][x][y][z] * (
                    curr_basic[0][x][y][z] - curr_basic[0][x][y][z-1] -
                    curr_basic[2][x][y][z] + curr_basic[2][x-1][y][z]);

                volt_basic[2][x][y][z] *= vv_basic[2][x][y][z];
                volt_basic[2][x][y][z] += vi_basic[2][x][y][z] * (
                    curr_basic[1][x][y][z] - curr_basic[1][x-1][y][z] -
                    curr_basic[0][x][y][z] + curr_basic[0][x][y-1][z]);
            }
        }
    }
}

void UpdateCurrents_Basic() {
    for (unsigned int x = 1; x < NX-1; x++) {
        for (unsigned int y = 1; y < NY-2; y++) {
            for (unsigned int z = 0; z < NZ-1; z++) {
                curr_basic[0][x][y][z] *= ii_basic[0][x][y][z];
                curr_basic[0][x][y][z] += iv_basic[0][x][y][z] * (
                    volt_basic[2][x][y][z] - volt_basic[2][x][y+1][z] -
                    volt_basic[1][x][y][z] + volt_basic[1][x][y][z+1]);

                curr_basic[1][x][y][z] *= ii_basic[1][x][y][z];
                curr_basic[1][x][y][z] += iv_basic[1][x][y][z] * (
                    volt_basic[0][x][y][z] - volt_basic[0][x][y][z+1] -
                    volt_basic[2][x][y][z] + volt_basic[2][x+1][y][z]);

                curr_basic[2][x][y][z] *= ii_basic[2][x][y][z];
                curr_basic[2][x][y][z] += iv_basic[2][x][y][z] * (
                    volt_basic[1][x][y][z] - volt_basic[1][x+1][y][z] -
                    volt_basic[0][x][y][z] + volt_basic[0][x][y+1][z]);
            }
        }
    }
}

// ============== SSE Engine Functions ==============
f4vector**** allocate_sse_array(unsigned int nx, unsigned int ny, unsigned int nz_vec) {
    f4vector**** arr = new f4vector***[3];
    for (int n = 0; n < 3; n++) {
        arr[n] = new f4vector**[nx];
        for (unsigned int x = 0; x < nx; x++) {
            arr[n][x] = new f4vector*[ny];
            for (unsigned int y = 0; y < ny; y++) {
                arr[n][x][y] = new f4vector[nz_vec];
                for (unsigned int z = 0; z < nz_vec; z++) {
                    for (int i = 0; i < 4; i++) arr[n][x][y][z].f[i] = 0.0f;
                }
            }
        }
    }
    return arr;
}

void free_sse_array(f4vector**** arr, unsigned int nx, unsigned int ny) {
    for (int n = 0; n < 3; n++) {
        for (unsigned int x = 0; x < nx; x++) {
            for (unsigned int y = 0; y < ny; y++) {
                delete[] arr[n][x][y];
            }
            delete[] arr[n][x];
        }
        delete[] arr[n];
    }
    delete[] arr;
}

void init_sse_arrays() {
    volt_sse = allocate_sse_array(NX, NY, NUM_VECTORS_SSE);
    curr_sse = allocate_sse_array(NX, NY, NUM_VECTORS_SSE);
    vv_sse = allocate_sse_array(NX, NY, NUM_VECTORS_SSE);
    vi_sse = allocate_sse_array(NX, NY, NUM_VECTORS_SSE);
    ii_sse = allocate_sse_array(NX, NY, NUM_VECTORS_SSE);
    iv_sse = allocate_sse_array(NX, NY, NUM_VECTORS_SSE);

    for (int n = 0; n < 3; n++) {
        for (unsigned int x = 0; x < NX; x++) {
            for (unsigned int y = 0; y < NY; y++) {
                for (unsigned int z = 0; z < NUM_VECTORS_SSE; z++) {
                    for (int i = 0; i < 4; i++) {
                        vv_sse[n][x][y][z].f[i] = 0.99f;
                        vi_sse[n][x][y][z].f[i] = 0.01f;
                        ii_sse[n][x][y][z].f[i] = 0.99f;
                        iv_sse[n][x][y][z].f[i] = 0.01f;
                    }
                }
            }
        }
    }
    volt_sse[0][NX/2][NY/2][NUM_VECTORS_SSE/2].f[0] = 1.0f;
}

void cleanup_sse_arrays() {
    free_sse_array(volt_sse, NX, NY);
    free_sse_array(curr_sse, NX, NY);
    free_sse_array(vv_sse, NX, NY);
    free_sse_array(vi_sse, NX, NY);
    free_sse_array(ii_sse, NX, NY);
    free_sse_array(iv_sse, NX, NY);
}

void UpdateVoltages_SSE() {
    for (unsigned int x = 1; x < NX-1; x++) {
        for (unsigned int y = 1; y < NY-1; y++) {
            for (unsigned int z = 1; z < NUM_VECTORS_SSE; z++) {
                volt_sse[0][x][y][z].v *= vv_sse[0][x][y][z].v;
                volt_sse[0][x][y][z].v += vi_sse[0][x][y][z].v * (
                    curr_sse[2][x][y][z].v - curr_sse[2][x][y-1][z].v -
                    curr_sse[1][x][y][z].v + curr_sse[1][x][y][z-1].v);

                volt_sse[1][x][y][z].v *= vv_sse[1][x][y][z].v;
                volt_sse[1][x][y][z].v += vi_sse[1][x][y][z].v * (
                    curr_sse[0][x][y][z].v - curr_sse[0][x][y][z-1].v -
                    curr_sse[2][x][y][z].v + curr_sse[2][x-1][y][z].v);

                volt_sse[2][x][y][z].v *= vv_sse[2][x][y][z].v;
                volt_sse[2][x][y][z].v += vi_sse[2][x][y][z].v * (
                    curr_sse[1][x][y][z].v - curr_sse[1][x-1][y][z].v -
                    curr_sse[0][x][y][z].v + curr_sse[0][x][y-1][z].v);
            }
        }
    }
}

void UpdateCurrents_SSE() {
    for (unsigned int x = 1; x < NX-1; x++) {
        for (unsigned int y = 1; y < NY-2; y++) {
            for (unsigned int z = 0; z < NUM_VECTORS_SSE-1; z++) {
                curr_sse[0][x][y][z].v *= ii_sse[0][x][y][z].v;
                curr_sse[0][x][y][z].v += iv_sse[0][x][y][z].v * (
                    volt_sse[2][x][y][z].v - volt_sse[2][x][y+1][z].v -
                    volt_sse[1][x][y][z].v + volt_sse[1][x][y][z+1].v);

                curr_sse[1][x][y][z].v *= ii_sse[1][x][y][z].v;
                curr_sse[1][x][y][z].v += iv_sse[1][x][y][z].v * (
                    volt_sse[0][x][y][z].v - volt_sse[0][x][y][z+1].v -
                    volt_sse[2][x][y][z].v + volt_sse[2][x+1][y][z].v);

                curr_sse[2][x][y][z].v *= ii_sse[2][x][y][z].v;
                curr_sse[2][x][y][z].v += iv_sse[2][x][y][z].v * (
                    volt_sse[1][x][y][z].v - volt_sse[1][x+1][y][z].v -
                    volt_sse[0][x][y][z].v + volt_sse[0][x][y+1][z].v);
            }
        }
    }
}

// ============== Highway Engine Functions ==============
inline size_t idx(int pol, unsigned int x, unsigned int y, unsigned int z) {
    return static_cast<size_t>(pol) * NX * NY * NZ + x * NY * NZ + y * NZ + z;
}

void init_hwy_arrays() {
    size_t total = 3ULL * NX * NY * NZ;
    volt_hwy = static_cast<float*>(aligned_alloc(64, total * sizeof(float)));
    curr_hwy = static_cast<float*>(aligned_alloc(64, total * sizeof(float)));
    vv_hwy = static_cast<float*>(aligned_alloc(64, total * sizeof(float)));
    vi_hwy = static_cast<float*>(aligned_alloc(64, total * sizeof(float)));
    ii_hwy = static_cast<float*>(aligned_alloc(64, total * sizeof(float)));
    iv_hwy = static_cast<float*>(aligned_alloc(64, total * sizeof(float)));

    std::memset(volt_hwy, 0, total * sizeof(float));
    std::memset(curr_hwy, 0, total * sizeof(float));

    for (size_t i = 0; i < total; i++) {
        vv_hwy[i] = 0.99f;
        vi_hwy[i] = 0.01f;
        ii_hwy[i] = 0.99f;
        iv_hwy[i] = 0.01f;
    }
    volt_hwy[idx(0, NX/2, NY/2, NZ/2)] = 1.0f;
}

void cleanup_hwy_arrays() {
    free(volt_hwy);
    free(curr_hwy);
    free(vv_hwy);
    free(vi_hwy);
    free(ii_hwy);
    free(iv_hwy);
}

void UpdateVoltages_Highway() {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);

    for (unsigned int x = 1; x < NX-1; x++) {
        for (unsigned int y = 1; y < NY-1; y++) {
            for (unsigned int z = N; z < NZ - N; z += N) {
                // x-polarization
                size_t i0 = idx(0, x, y, z);
                auto v0 = hn::Load(d, &volt_hwy[i0]);
                auto vv0 = hn::Load(d, &vv_hwy[i0]);
                auto vi0 = hn::Load(d, &vi_hwy[i0]);
                auto c2_0 = hn::Load(d, &curr_hwy[idx(2, x, y, z)]);
                auto c2_ym1 = hn::Load(d, &curr_hwy[idx(2, x, y-1, z)]);
                auto c1_0 = hn::Load(d, &curr_hwy[idx(1, x, y, z)]);
                auto c1_zm1 = hn::Load(d, &curr_hwy[idx(1, x, y, z) - N]);
                auto curl = hn::Sub(hn::Sub(c2_0, c2_ym1), hn::Sub(c1_0, c1_zm1));
                v0 = hn::MulAdd(vi0, curl, hn::Mul(v0, vv0));
                hn::Store(v0, d, &volt_hwy[i0]);

                // y-polarization
                size_t i1 = idx(1, x, y, z);
                auto v1 = hn::Load(d, &volt_hwy[i1]);
                auto vv1 = hn::Load(d, &vv_hwy[i1]);
                auto vi1 = hn::Load(d, &vi_hwy[i1]);
                auto c0_0 = hn::Load(d, &curr_hwy[idx(0, x, y, z)]);
                auto c0_zm1 = hn::Load(d, &curr_hwy[idx(0, x, y, z) - N]);
                auto c2_curr = hn::Load(d, &curr_hwy[idx(2, x, y, z)]);
                auto c2_xm1 = hn::Load(d, &curr_hwy[idx(2, x-1, y, z)]);
                curl = hn::Sub(hn::Sub(c0_0, c0_zm1), hn::Sub(c2_curr, c2_xm1));
                v1 = hn::MulAdd(vi1, curl, hn::Mul(v1, vv1));
                hn::Store(v1, d, &volt_hwy[i1]);

                // z-polarization
                size_t i2 = idx(2, x, y, z);
                auto v2 = hn::Load(d, &volt_hwy[i2]);
                auto vv2 = hn::Load(d, &vv_hwy[i2]);
                auto vi2 = hn::Load(d, &vi_hwy[i2]);
                auto c1_curr = hn::Load(d, &curr_hwy[idx(1, x, y, z)]);
                auto c1_xm1 = hn::Load(d, &curr_hwy[idx(1, x-1, y, z)]);
                auto c0_curr = hn::Load(d, &curr_hwy[idx(0, x, y, z)]);
                auto c0_ym1 = hn::Load(d, &curr_hwy[idx(0, x, y-1, z)]);
                curl = hn::Sub(hn::Sub(c1_curr, c1_xm1), hn::Sub(c0_curr, c0_ym1));
                v2 = hn::MulAdd(vi2, curl, hn::Mul(v2, vv2));
                hn::Store(v2, d, &volt_hwy[i2]);
            }
        }
    }
}

void UpdateCurrents_Highway() {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);

    for (unsigned int x = 1; x < NX-1; x++) {
        for (unsigned int y = 1; y < NY-2; y++) {
            for (unsigned int z = 0; z < NZ - N; z += N) {
                // x-polarization
                size_t i0 = idx(0, x, y, z);
                auto c0 = hn::Load(d, &curr_hwy[i0]);
                auto ii0 = hn::Load(d, &ii_hwy[i0]);
                auto iv0 = hn::Load(d, &iv_hwy[i0]);
                auto v2_0 = hn::Load(d, &volt_hwy[idx(2, x, y, z)]);
                auto v2_yp1 = hn::Load(d, &volt_hwy[idx(2, x, y+1, z)]);
                auto v1_0 = hn::Load(d, &volt_hwy[idx(1, x, y, z)]);
                auto v1_zp1 = hn::Load(d, &volt_hwy[idx(1, x, y, z) + N]);
                auto curl = hn::Sub(hn::Sub(v2_0, v2_yp1), hn::Sub(v1_0, v1_zp1));
                c0 = hn::MulAdd(iv0, curl, hn::Mul(c0, ii0));
                hn::Store(c0, d, &curr_hwy[i0]);

                // y-polarization
                size_t i1 = idx(1, x, y, z);
                auto c1 = hn::Load(d, &curr_hwy[i1]);
                auto ii1 = hn::Load(d, &ii_hwy[i1]);
                auto iv1 = hn::Load(d, &iv_hwy[i1]);
                auto v0_0 = hn::Load(d, &volt_hwy[idx(0, x, y, z)]);
                auto v0_zp1 = hn::Load(d, &volt_hwy[idx(0, x, y, z) + N]);
                auto v2_curr = hn::Load(d, &volt_hwy[idx(2, x, y, z)]);
                auto v2_xp1 = hn::Load(d, &volt_hwy[idx(2, x+1, y, z)]);
                curl = hn::Sub(hn::Sub(v0_0, v0_zp1), hn::Sub(v2_curr, v2_xp1));
                c1 = hn::MulAdd(iv1, curl, hn::Mul(c1, ii1));
                hn::Store(c1, d, &curr_hwy[i1]);

                // z-polarization
                size_t i2 = idx(2, x, y, z);
                auto c2 = hn::Load(d, &curr_hwy[i2]);
                auto ii2 = hn::Load(d, &ii_hwy[i2]);
                auto iv2 = hn::Load(d, &iv_hwy[i2]);
                auto v1_curr = hn::Load(d, &volt_hwy[idx(1, x, y, z)]);
                auto v1_xp1 = hn::Load(d, &volt_hwy[idx(1, x+1, y, z)]);
                auto v0_curr = hn::Load(d, &volt_hwy[idx(0, x, y, z)]);
                auto v0_yp1 = hn::Load(d, &volt_hwy[idx(0, x, y+1, z)]);
                curl = hn::Sub(hn::Sub(v1_curr, v1_xp1), hn::Sub(v0_curr, v0_yp1));
                c2 = hn::MulAdd(iv2, curl, hn::Mul(c2, ii2));
                hn::Store(c2, d, &curr_hwy[i2]);
            }
        }
    }
}

// ============== Benchmark Runner ==============
struct BenchResult {
    const char* name;
    double time_ms;
    double mc_per_sec;
    size_t simd_width;
};

void RunBenchmark() {
    const hn::ScalableTag<float> d;
    size_t hwy_lanes = hn::Lanes(d);

    std::cout << "================================================================" << std::endl;
    std::cout << "       FDTD Engine Comprehensive Performance Benchmark" << std::endl;
    std::cout << "================================================================" << std::endl;
    std::cout << std::endl;
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Grid size:    " << NX << " x " << NY << " x " << NZ << std::endl;
    std::cout << "  Total cells:  " << (unsigned long long)NX * NY * NZ << std::endl;
    std::cout << "  Timesteps:    " << TIMESTEPS << std::endl;
    std::cout << "  Highway SIMD: " << hwy_lanes << " floats per vector" << std::endl;
    std::cout << std::endl;

    BenchResult results[3];

    // Basic Engine Benchmark
    std::cout << "Running Basic (scalar) benchmark..." << std::endl;
    init_basic_arrays();
    auto start = std::chrono::high_resolution_clock::now();
    for (unsigned int ts = 0; ts < TIMESTEPS; ts++) {
        UpdateVoltages_Basic();
        UpdateCurrents_Basic();
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    results[0] = {"Basic (scalar)", static_cast<double>(duration),
                  ((double)NX * NY * NZ * TIMESTEPS) / (duration / 1000.0) / 1e6, 1};
    cleanup_basic_arrays();

    // SSE Engine Benchmark
    std::cout << "Running SSE (128-bit) benchmark..." << std::endl;
    init_sse_arrays();
    start = std::chrono::high_resolution_clock::now();
    for (unsigned int ts = 0; ts < TIMESTEPS; ts++) {
        UpdateVoltages_SSE();
        UpdateCurrents_SSE();
    }
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    results[1] = {"SSE (128-bit)", static_cast<double>(duration),
                  ((double)NX * NY * NZ * TIMESTEPS) / (duration / 1000.0) / 1e6, 4};
    cleanup_sse_arrays();

    // Highway Engine Benchmark
    std::cout << "Running Highway (auto-vectorized) benchmark..." << std::endl;
    init_hwy_arrays();
    start = std::chrono::high_resolution_clock::now();
    for (unsigned int ts = 0; ts < TIMESTEPS; ts++) {
        UpdateVoltages_Highway();
        UpdateCurrents_Highway();
    }
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    results[2] = {"Highway (AVX-512)", static_cast<double>(duration),
                  ((double)NX * NY * NZ * TIMESTEPS) / (duration / 1000.0) / 1e6, hwy_lanes};
    cleanup_hwy_arrays();

    // Print Results
    std::cout << std::endl;
    std::cout << "================================================================" << std::endl;
    std::cout << "                         RESULTS" << std::endl;
    std::cout << "================================================================" << std::endl;
    std::cout << std::endl;
    std::cout << std::left << std::setw(20) << "Engine"
              << std::right << std::setw(12) << "Time (ms)"
              << std::setw(15) << "Speed (MC/s)"
              << std::setw(12) << "SIMD Width"
              << std::setw(12) << "Speedup" << std::endl;
    std::cout << std::string(71, '-') << std::endl;

    double baseline = results[0].mc_per_sec;
    for (int i = 0; i < 3; i++) {
        std::cout << std::left << std::setw(20) << results[i].name
                  << std::right << std::setw(12) << std::fixed << std::setprecision(0) << results[i].time_ms
                  << std::setw(15) << std::setprecision(2) << results[i].mc_per_sec
                  << std::setw(12) << results[i].simd_width
                  << std::setw(11) << std::setprecision(2) << (results[i].mc_per_sec / baseline) << "x" << std::endl;
    }

    std::cout << std::endl;
    std::cout << "================================================================" << std::endl;
    std::cout << "                      PERFORMANCE GAINS" << std::endl;
    std::cout << "================================================================" << std::endl;
    std::cout << std::endl;
    std::cout << "Highway vs SSE speedup:   " << std::fixed << std::setprecision(2)
              << (results[2].mc_per_sec / results[1].mc_per_sec) << "x" << std::endl;
    std::cout << "Highway vs Basic speedup: " << std::fixed << std::setprecision(2)
              << (results[2].mc_per_sec / results[0].mc_per_sec) << "x" << std::endl;
    std::cout << std::endl;

    // Memory bandwidth analysis
    double bytes_per_cell = 12 * sizeof(float) * 2;  // 6 arrays, read+write for volt and curr
    double bandwidth_hwy = (bytes_per_cell * NX * NY * NZ * TIMESTEPS) / (results[2].time_ms / 1000.0) / 1e9;
    std::cout << "Estimated memory bandwidth (Highway): " << std::fixed << std::setprecision(2)
              << bandwidth_hwy << " GB/s" << std::endl;
}

}  // namespace HWY_NAMESPACE
}  // namespace fdtd_all
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace fdtd_all {
HWY_EXPORT(RunBenchmark);

void RunBenchmarkDispatcher() {
    return HWY_DYNAMIC_DISPATCH(RunBenchmark)();
}
}  // namespace fdtd_all

int main() {
    fdtd_all::RunBenchmarkDispatcher();
    return 0;
}
#endif
