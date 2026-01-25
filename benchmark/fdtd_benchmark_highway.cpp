/*
 * FDTD Engine Benchmark with Google Highway
 * Measures performance with portable SIMD (auto-detects AVX-512/AVX2/SSE)
 */

#include <iostream>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdlib>

// Highway includes
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "fdtd_benchmark_highway.cpp"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace fdtd_hwy {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

// Grid dimensions
constexpr unsigned int NX = 100;  // X dimension
constexpr unsigned int NY = 100;  // Y dimension
constexpr unsigned int NZ = 128;  // Z dimension (multiple of 16 for AVX-512)
constexpr unsigned int TIMESTEPS = 1000;

// Arrays stored as flat contiguous memory for better SIMD performance
// Layout: volt[pol][x][y][z] -> volt[pol * NX*NY*NZ + x * NY*NZ + y * NZ + z]
float* volt_hwy = nullptr;
float* curr_hwy = nullptr;
float* vv_hwy = nullptr;
float* vi_hwy = nullptr;
float* ii_hwy = nullptr;
float* iv_hwy = nullptr;

inline size_t idx(int pol, unsigned int x, unsigned int y, unsigned int z) {
    return static_cast<size_t>(pol) * NX * NY * NZ + x * NY * NZ + y * NZ + z;
}

void init_hwy_arrays() {
    size_t total = 3ULL * NX * NY * NZ;

    // Use aligned allocation for SIMD
    volt_hwy = static_cast<float*>(aligned_alloc(64, total * sizeof(float)));
    curr_hwy = static_cast<float*>(aligned_alloc(64, total * sizeof(float)));
    vv_hwy = static_cast<float*>(aligned_alloc(64, total * sizeof(float)));
    vi_hwy = static_cast<float*>(aligned_alloc(64, total * sizeof(float)));
    ii_hwy = static_cast<float*>(aligned_alloc(64, total * sizeof(float)));
    iv_hwy = static_cast<float*>(aligned_alloc(64, total * sizeof(float)));

    // Initialize to zero
    std::memset(volt_hwy, 0, total * sizeof(float));
    std::memset(curr_hwy, 0, total * sizeof(float));

    // Initialize coefficients
    for (size_t i = 0; i < total; i++) {
        vv_hwy[i] = 0.99f;
        vi_hwy[i] = 0.01f;
        ii_hwy[i] = 0.99f;
        iv_hwy[i] = 0.01f;
    }

    // Initialize source
    volt_hwy[idx(0, NX/2, NY/2, NZ/2)] = 1.0f;
}

void free_hwy_arrays() {
    free(volt_hwy);
    free(curr_hwy);
    free(vv_hwy);
    free(vi_hwy);
    free(ii_hwy);
    free(iv_hwy);
}

// Highway vectorized UpdateVoltages
void UpdateVoltages_Highway() {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);

    for (unsigned int x = 1; x < NX-1; x++) {
        for (unsigned int y = 1; y < NY-1; y++) {
            // Process Z dimension in SIMD-width chunks
            for (unsigned int z = N; z < NZ - N; z += N) {
                // x-polarization: volt[0] depends on curr[2] and curr[1]
                size_t i0 = idx(0, x, y, z);
                size_t i2_0 = idx(2, x, y, z);
                size_t i2_ym1 = idx(2, x, y-1, z);
                size_t i1_0 = idx(1, x, y, z);
                size_t i1_zm1 = idx(1, x, y, z) - N;

                auto v0 = hn::Load(d, &volt_hwy[i0]);
                auto vv0 = hn::Load(d, &vv_hwy[i0]);
                auto vi0 = hn::Load(d, &vi_hwy[i0]);
                auto c2_0 = hn::Load(d, &curr_hwy[i2_0]);
                auto c2_ym1 = hn::Load(d, &curr_hwy[i2_ym1]);
                auto c1_0 = hn::Load(d, &curr_hwy[i1_0]);
                auto c1_zm1 = hn::Load(d, &curr_hwy[i1_zm1]);

                auto curl_x = hn::Sub(hn::Sub(c2_0, c2_ym1), hn::Sub(c1_0, c1_zm1));
                v0 = hn::MulAdd(vi0, curl_x, hn::Mul(v0, vv0));
                hn::Store(v0, d, &volt_hwy[i0]);

                // y-polarization
                size_t i1_pol = idx(1, x, y, z);
                size_t i0_curr = idx(0, x, y, z);
                size_t i0_curr_zm1 = idx(0, x, y, z) - N;
                size_t i2_curr = idx(2, x, y, z);
                size_t i2_curr_xm1 = idx(2, x-1, y, z);

                auto v1 = hn::Load(d, &volt_hwy[i1_pol]);
                auto vv1 = hn::Load(d, &vv_hwy[i1_pol]);
                auto vi1 = hn::Load(d, &vi_hwy[i1_pol]);
                auto c0_0 = hn::Load(d, &curr_hwy[i0_curr]);
                auto c0_zm1 = hn::Load(d, &curr_hwy[i0_curr_zm1]);
                auto c2_curr = hn::Load(d, &curr_hwy[i2_curr]);
                auto c2_xm1 = hn::Load(d, &curr_hwy[i2_curr_xm1]);

                auto curl_y = hn::Sub(hn::Sub(c0_0, c0_zm1), hn::Sub(c2_curr, c2_xm1));
                v1 = hn::MulAdd(vi1, curl_y, hn::Mul(v1, vv1));
                hn::Store(v1, d, &volt_hwy[i1_pol]);

                // z-polarization
                size_t i2_pol = idx(2, x, y, z);
                size_t i1_curr = idx(1, x, y, z);
                size_t i1_curr_xm1 = idx(1, x-1, y, z);
                size_t i0_curr_pol = idx(0, x, y, z);
                size_t i0_curr_ym1 = idx(0, x, y-1, z);

                auto v2 = hn::Load(d, &volt_hwy[i2_pol]);
                auto vv2 = hn::Load(d, &vv_hwy[i2_pol]);
                auto vi2 = hn::Load(d, &vi_hwy[i2_pol]);
                auto c1_curr = hn::Load(d, &curr_hwy[i1_curr]);
                auto c1_xm1 = hn::Load(d, &curr_hwy[i1_curr_xm1]);
                auto c0_curr = hn::Load(d, &curr_hwy[i0_curr_pol]);
                auto c0_ym1 = hn::Load(d, &curr_hwy[i0_curr_ym1]);

                auto curl_z = hn::Sub(hn::Sub(c1_curr, c1_xm1), hn::Sub(c0_curr, c0_ym1));
                v2 = hn::MulAdd(vi2, curl_z, hn::Mul(v2, vv2));
                hn::Store(v2, d, &volt_hwy[i2_pol]);
            }
        }
    }
}

// Highway vectorized UpdateCurrents
void UpdateCurrents_Highway() {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);

    for (unsigned int x = 1; x < NX-1; x++) {
        for (unsigned int y = 1; y < NY-2; y++) {
            for (unsigned int z = 0; z < NZ - N; z += N) {
                // x-polarization
                size_t i0 = idx(0, x, y, z);
                size_t i2_0 = idx(2, x, y, z);
                size_t i2_yp1 = idx(2, x, y+1, z);
                size_t i1_0 = idx(1, x, y, z);
                size_t i1_zp1 = idx(1, x, y, z) + N;

                auto c0 = hn::Load(d, &curr_hwy[i0]);
                auto ii0 = hn::Load(d, &ii_hwy[i0]);
                auto iv0 = hn::Load(d, &iv_hwy[i0]);
                auto v2_0 = hn::Load(d, &volt_hwy[i2_0]);
                auto v2_yp1 = hn::Load(d, &volt_hwy[i2_yp1]);
                auto v1_0 = hn::Load(d, &volt_hwy[i1_0]);
                auto v1_zp1 = hn::Load(d, &volt_hwy[i1_zp1]);

                auto curl_x = hn::Sub(hn::Sub(v2_0, v2_yp1), hn::Sub(v1_0, v1_zp1));
                c0 = hn::MulAdd(iv0, curl_x, hn::Mul(c0, ii0));
                hn::Store(c0, d, &curr_hwy[i0]);

                // y-polarization
                size_t i1_pol = idx(1, x, y, z);
                size_t i0_volt = idx(0, x, y, z);
                size_t i0_volt_zp1 = idx(0, x, y, z) + N;
                size_t i2_volt = idx(2, x, y, z);
                size_t i2_volt_xp1 = idx(2, x+1, y, z);

                auto c1 = hn::Load(d, &curr_hwy[i1_pol]);
                auto ii1 = hn::Load(d, &ii_hwy[i1_pol]);
                auto iv1 = hn::Load(d, &iv_hwy[i1_pol]);
                auto v0_0 = hn::Load(d, &volt_hwy[i0_volt]);
                auto v0_zp1 = hn::Load(d, &volt_hwy[i0_volt_zp1]);
                auto v2_curr = hn::Load(d, &volt_hwy[i2_volt]);
                auto v2_xp1 = hn::Load(d, &volt_hwy[i2_volt_xp1]);

                auto curl_y = hn::Sub(hn::Sub(v0_0, v0_zp1), hn::Sub(v2_curr, v2_xp1));
                c1 = hn::MulAdd(iv1, curl_y, hn::Mul(c1, ii1));
                hn::Store(c1, d, &curr_hwy[i1_pol]);

                // z-polarization
                size_t i2_pol = idx(2, x, y, z);
                size_t i1_volt = idx(1, x, y, z);
                size_t i1_volt_xp1 = idx(1, x+1, y, z);
                size_t i0_volt_pol = idx(0, x, y, z);
                size_t i0_volt_yp1 = idx(0, x, y+1, z);

                auto c2 = hn::Load(d, &curr_hwy[i2_pol]);
                auto ii2 = hn::Load(d, &ii_hwy[i2_pol]);
                auto iv2 = hn::Load(d, &iv_hwy[i2_pol]);
                auto v1_curr = hn::Load(d, &volt_hwy[i1_volt]);
                auto v1_xp1 = hn::Load(d, &volt_hwy[i1_volt_xp1]);
                auto v0_curr = hn::Load(d, &volt_hwy[i0_volt_pol]);
                auto v0_yp1 = hn::Load(d, &volt_hwy[i0_volt_yp1]);

                auto curl_z = hn::Sub(hn::Sub(v1_curr, v1_xp1), hn::Sub(v0_curr, v0_yp1));
                c2 = hn::MulAdd(iv2, curl_z, hn::Mul(c2, ii2));
                hn::Store(c2, d, &curr_hwy[i2_pol]);
            }
        }
    }
}

void RunBenchmark() {
    const hn::ScalableTag<float> d;
    std::cout << "====== FDTD Highway Benchmark ======" << std::endl;
    std::cout << "Grid size: " << NX << " x " << NY << " x " << NZ << std::endl;
    std::cout << "Timesteps: " << TIMESTEPS << std::endl;
    std::cout << "Total cells: " << (unsigned long long)NX * NY * NZ << std::endl;
    std::cout << "SIMD width: " << hn::Lanes(d) << " floats" << std::endl;
    std::cout << std::endl;

    std::cout << "Initializing Highway arrays..." << std::endl;
    init_hwy_arrays();

    std::cout << "Running Highway benchmark..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    for (unsigned int ts = 0; ts < TIMESTEPS; ts++) {
        UpdateVoltages_Highway();
        UpdateCurrents_Highway();
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    double cells_per_sec = ((double)NX * NY * NZ * TIMESTEPS) / (duration / 1000.0);
    double mc_per_sec = cells_per_sec / 1e6;

    std::cout << "Highway Engine:" << std::endl;
    std::cout << "  Time: " << duration << " ms" << std::endl;
    std::cout << "  Speed: " << mc_per_sec << " MC/s" << std::endl;

    free_hwy_arrays();
}

}  // namespace HWY_NAMESPACE
}  // namespace fdtd_hwy
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace fdtd_hwy {
HWY_EXPORT(RunBenchmark);

void RunBenchmarkDispatcher() {
    return HWY_DYNAMIC_DISPATCH(RunBenchmark)();
}
}  // namespace fdtd_hwy

int main() {
    fdtd_hwy::RunBenchmarkDispatcher();
    return 0;
}
#endif
