/*
 * FDTD Benchmark using Google Highway
 * Compares Highway (portable SIMD) against hand-written SSE/AVX
 *
 * Uses static dispatch - Highway picks best SIMD from compile flags (-march=native)
 */

#include "hwy/highway.h"
#include "hwy/aligned_allocator.h"

#include <iostream>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>

#ifdef __SSE2__
#include <emmintrin.h>
#endif

#ifdef __AVX__
#include <immintrin.h>
#endif

#ifdef __SSE__
#include <xmmintrin.h>
#endif

using namespace std;
using namespace std::chrono;

namespace hn = hwy::HWY_NAMESPACE;

// Use the widest available SIMD
using D = hn::ScalableTag<float>;

// Memory allocation
template<typename T, size_t Align = 64>
T* aligned_alloc_array(size_t n) {
    void* ptr = nullptr;
    if (posix_memalign(&ptr, Align, n * sizeof(T)) != 0) {
        throw bad_alloc();
    }
    memset(ptr, 0, n * sizeof(T));
    return static_cast<T*>(ptr);
}

template<typename T>
void aligned_free(T* ptr) { free(ptr); }

class Timer {
public:
    void start() { m_start = high_resolution_clock::now(); }
    void stop() { m_end = high_resolution_clock::now(); }
    double elapsed_us() const {
        return duration_cast<microseconds>(m_end - m_start).count();
    }
private:
    time_point<high_resolution_clock> m_start, m_end;
};

//=============================================================================
// Highway SIMD Implementation
//=============================================================================
class Engine_Highway {
public:
    unsigned int nx, ny, nz;
    float *volt, *curr, *vv, *vi, *ii, *iv;

    Engine_Highway(unsigned int _nx, unsigned int _ny, unsigned int _nz)
        : nx(_nx), ny(_ny), nz(_nz) {
        size_t total = 3ULL * nx * ny * nz;
        volt = aligned_alloc_array<float>(total);
        curr = aligned_alloc_array<float>(total);
        vv = aligned_alloc_array<float>(total);
        vi = aligned_alloc_array<float>(total);
        ii = aligned_alloc_array<float>(total);
        iv = aligned_alloc_array<float>(total);

        for (size_t i = 0; i < total; i++) {
            volt[i] = 0.001f * (i % 1000);
            curr[i] = 0.001f * ((i + 500) % 1000);
            vv[i] = 0.999f;
            vi[i] = 0.001f;
            ii[i] = 0.999f;
            iv[i] = 0.001f;
        }
    }

    ~Engine_Highway() {
        aligned_free(volt);
        aligned_free(curr);
        aligned_free(vv);
        aligned_free(vi);
        aligned_free(ii);
        aligned_free(iv);
    }

    void UpdateVoltages() {
        const D d;
        const size_t lanes = hn::Lanes(d);

        // N-I-J-K memory layout
        const size_t stride_n = (size_t)nx * ny * nz;
        const size_t stride_x = (size_t)ny * nz;
        const size_t stride_y = nz;

        for (unsigned int x = 0; x < nx; ++x) {
            const size_t shift_x = (x > 0) ? stride_x : 0;

            for (unsigned int y = 0; y < ny; ++y) {
                const size_t shift_y = (y > 0) ? stride_y : 0;
                const size_t base_xy = x * stride_x + y * stride_y;

                // Main loop - vectorized
                for (size_t z = lanes; z < nz; z += lanes) {
                    const size_t idx = base_xy + z;

                    // X-component
                    auto v0 = hn::Load(d, volt + idx);
                    auto vv0 = hn::Load(d, vv + idx);
                    auto vi0 = hn::Load(d, vi + idx);

                    auto c2_y = hn::Load(d, curr + 2*stride_n + idx);
                    auto c2_ym = hn::Load(d, curr + 2*stride_n + idx - shift_y);
                    auto c1_z = hn::Load(d, curr + stride_n + idx);
                    auto c1_zm = hn::Load(d, curr + stride_n + idx - lanes);

                    auto curl0 = hn::Sub(hn::Sub(hn::Add(c2_y, c1_zm), c2_ym), c1_z);
                    v0 = hn::MulAdd(vi0, curl0, hn::Mul(v0, vv0));
                    hn::Store(v0, d, volt + idx);

                    // Y-component
                    auto v1 = hn::Load(d, volt + stride_n + idx);
                    auto vv1 = hn::Load(d, vv + stride_n + idx);
                    auto vi1 = hn::Load(d, vi + stride_n + idx);

                    auto c0_z = hn::Load(d, curr + idx);
                    auto c0_zm = hn::Load(d, curr + idx - lanes);
                    auto c2_x = hn::Load(d, curr + 2*stride_n + idx);
                    auto c2_xm = hn::Load(d, curr + 2*stride_n + idx - shift_x);

                    auto curl1 = hn::Sub(hn::Sub(hn::Add(c0_z, c2_xm), c0_zm), c2_x);
                    v1 = hn::MulAdd(vi1, curl1, hn::Mul(v1, vv1));
                    hn::Store(v1, d, volt + stride_n + idx);

                    // Z-component
                    auto v2 = hn::Load(d, volt + 2*stride_n + idx);
                    auto vv2 = hn::Load(d, vv + 2*stride_n + idx);
                    auto vi2 = hn::Load(d, vi + 2*stride_n + idx);

                    auto c1_x = hn::Load(d, curr + stride_n + idx);
                    auto c1_xm = hn::Load(d, curr + stride_n + idx - shift_x);
                    auto c0_y = hn::Load(d, curr + idx);
                    auto c0_ym = hn::Load(d, curr + idx - shift_y);

                    auto curl2 = hn::Sub(hn::Sub(hn::Add(c1_x, c0_ym), c1_xm), c0_y);
                    v2 = hn::MulAdd(vi2, curl2, hn::Mul(v2, vv2));
                    hn::Store(v2, d, volt + 2*stride_n + idx);
                }

                // Boundary at z=0 (simplified)
                for (size_t z = 0; z < lanes && z < nz; z += lanes) {
                    const size_t idx = base_xy + z;

                    auto v0 = hn::Load(d, volt + idx);
                    auto v1 = hn::Load(d, volt + stride_n + idx);
                    auto v2 = hn::Load(d, volt + 2*stride_n + idx);

                    auto vv0 = hn::Load(d, vv + idx);
                    auto vv1 = hn::Load(d, vv + stride_n + idx);
                    auto vv2 = hn::Load(d, vv + 2*stride_n + idx);

                    v0 = hn::Mul(v0, vv0);
                    v1 = hn::Mul(v1, vv1);
                    v2 = hn::Mul(v2, vv2);

                    hn::Store(v0, d, volt + idx);
                    hn::Store(v1, d, volt + stride_n + idx);
                    hn::Store(v2, d, volt + 2*stride_n + idx);
                }
            }
        }
    }

    void UpdateCurrents() {
        const D d;
        const size_t lanes = hn::Lanes(d);

        const size_t stride_n = (size_t)nx * ny * nz;
        const size_t stride_x = (size_t)ny * nz;
        const size_t stride_y = nz;

        for (unsigned int x = 0; x < nx - 1; ++x) {
            const size_t shift_x = stride_x;

            for (unsigned int y = 0; y < ny - 1; ++y) {
                const size_t shift_y = stride_y;
                const size_t base_xy = x * stride_x + y * stride_y;

                for (size_t z = 0; z + lanes <= nz - 1; z += lanes) {
                    const size_t idx = base_xy + z;

                    // X-component
                    auto c0 = hn::Load(d, curr + idx);
                    auto ii0 = hn::Load(d, ii + idx);
                    auto iv0 = hn::Load(d, iv + idx);

                    auto v2_y = hn::Load(d, volt + 2*stride_n + idx);
                    auto v2_yp = hn::Load(d, volt + 2*stride_n + idx + shift_y);
                    auto v1_z = hn::Load(d, volt + stride_n + idx);
                    auto v1_zp = hn::Load(d, volt + stride_n + idx + lanes);

                    auto curl0 = hn::Sub(hn::Sub(hn::Add(v2_y, v1_zp), v2_yp), v1_z);
                    c0 = hn::MulAdd(iv0, curl0, hn::Mul(c0, ii0));
                    hn::Store(c0, d, curr + idx);

                    // Y-component
                    auto c1 = hn::Load(d, curr + stride_n + idx);
                    auto ii1 = hn::Load(d, ii + stride_n + idx);
                    auto iv1 = hn::Load(d, iv + stride_n + idx);

                    auto v0_z = hn::Load(d, volt + idx);
                    auto v0_zp = hn::Load(d, volt + idx + lanes);
                    auto v2_x = hn::Load(d, volt + 2*stride_n + idx);
                    auto v2_xp = hn::Load(d, volt + 2*stride_n + idx + shift_x);

                    auto curl1 = hn::Sub(hn::Sub(hn::Add(v0_z, v2_xp), v0_zp), v2_x);
                    c1 = hn::MulAdd(iv1, curl1, hn::Mul(c1, ii1));
                    hn::Store(c1, d, curr + stride_n + idx);

                    // Z-component
                    auto c2 = hn::Load(d, curr + 2*stride_n + idx);
                    auto ii2 = hn::Load(d, ii + 2*stride_n + idx);
                    auto iv2 = hn::Load(d, iv + 2*stride_n + idx);

                    auto v1_x = hn::Load(d, volt + stride_n + idx);
                    auto v1_xp = hn::Load(d, volt + stride_n + idx + shift_x);
                    auto v0_y = hn::Load(d, volt + idx);
                    auto v0_yp = hn::Load(d, volt + idx + shift_y);

                    auto curl2 = hn::Sub(hn::Sub(hn::Add(v1_x, v0_yp), v1_xp), v0_y);
                    c2 = hn::MulAdd(iv2, curl2, hn::Mul(c2, ii2));
                    hn::Store(c2, d, curr + 2*stride_n + idx);
                }
            }
        }
    }
};

//=============================================================================
// Reference SSE Implementation
//=============================================================================
#ifdef __SSE2__
class Engine_SSE_Ref {
public:
    unsigned int nx, ny, nz;
    float *volt, *curr, *vv, *vi, *ii, *iv;

    Engine_SSE_Ref(unsigned int _nx, unsigned int _ny, unsigned int _nz)
        : nx(_nx), ny(_ny), nz(_nz) {
        size_t total = 3ULL * nx * ny * nz;
        volt = aligned_alloc_array<float>(total);
        curr = aligned_alloc_array<float>(total);
        vv = aligned_alloc_array<float>(total);
        vi = aligned_alloc_array<float>(total);
        ii = aligned_alloc_array<float>(total);
        iv = aligned_alloc_array<float>(total);

        for (size_t i = 0; i < total; i++) {
            volt[i] = 0.001f * (i % 1000);
            curr[i] = 0.001f * ((i + 500) % 1000);
            vv[i] = 0.999f;
            vi[i] = 0.001f;
            ii[i] = 0.999f;
            iv[i] = 0.001f;
        }
    }

    ~Engine_SSE_Ref() {
        aligned_free(volt);
        aligned_free(curr);
        aligned_free(vv);
        aligned_free(vi);
        aligned_free(ii);
        aligned_free(iv);
    }

    void UpdateVoltages() {
        const size_t stride_n = (size_t)nx * ny * nz;
        const size_t stride_x = (size_t)ny * nz;
        const size_t stride_y = nz;

        for (unsigned int x = 0; x < nx; ++x) {
            size_t shift_x = (x > 0) ? stride_x : 0;
            for (unsigned int y = 0; y < ny; ++y) {
                size_t shift_y = (y > 0) ? stride_y : 0;
                size_t base_xy = x * stride_x + y * stride_y;

                for (unsigned int z = 4; z < nz; z += 4) {
                    size_t idx = base_xy + z;

                    __m128 v0 = _mm_load_ps(volt + idx);
                    __m128 vv0 = _mm_load_ps(vv + idx);
                    __m128 vi0 = _mm_load_ps(vi + idx);

                    __m128 c2_y = _mm_load_ps(curr + 2*stride_n + idx);
                    __m128 c2_ym = _mm_load_ps(curr + 2*stride_n + idx - shift_y);
                    __m128 c1_z = _mm_load_ps(curr + stride_n + idx);
                    __m128 c1_zm = _mm_load_ps(curr + stride_n + idx - 4);

                    __m128 curl0 = _mm_sub_ps(_mm_sub_ps(_mm_add_ps(c2_y, c1_zm), c2_ym), c1_z);
                    v0 = _mm_add_ps(_mm_mul_ps(v0, vv0), _mm_mul_ps(vi0, curl0));
                    _mm_store_ps(volt + idx, v0);

                    __m128 v1 = _mm_load_ps(volt + stride_n + idx);
                    __m128 vv1 = _mm_load_ps(vv + stride_n + idx);
                    __m128 vi1 = _mm_load_ps(vi + stride_n + idx);

                    __m128 c0_z = _mm_load_ps(curr + idx);
                    __m128 c0_zm = _mm_load_ps(curr + idx - 4);
                    __m128 c2_x = _mm_load_ps(curr + 2*stride_n + idx);
                    __m128 c2_xm = _mm_load_ps(curr + 2*stride_n + idx - shift_x);

                    __m128 curl1 = _mm_sub_ps(_mm_sub_ps(_mm_add_ps(c0_z, c2_xm), c0_zm), c2_x);
                    v1 = _mm_add_ps(_mm_mul_ps(v1, vv1), _mm_mul_ps(vi1, curl1));
                    _mm_store_ps(volt + stride_n + idx, v1);

                    __m128 v2 = _mm_load_ps(volt + 2*stride_n + idx);
                    __m128 vv2 = _mm_load_ps(vv + 2*stride_n + idx);
                    __m128 vi2 = _mm_load_ps(vi + 2*stride_n + idx);

                    __m128 c1_x = _mm_load_ps(curr + stride_n + idx);
                    __m128 c1_xm = _mm_load_ps(curr + stride_n + idx - shift_x);
                    __m128 c0_y = _mm_load_ps(curr + idx);
                    __m128 c0_ym = _mm_load_ps(curr + idx - shift_y);

                    __m128 curl2 = _mm_sub_ps(_mm_sub_ps(_mm_add_ps(c1_x, c0_ym), c1_xm), c0_y);
                    v2 = _mm_add_ps(_mm_mul_ps(v2, vv2), _mm_mul_ps(vi2, curl2));
                    _mm_store_ps(volt + 2*stride_n + idx, v2);
                }

                // Boundary
                for (unsigned int z = 0; z < 4 && z < nz; z += 4) {
                    size_t idx = base_xy + z;
                    __m128 v0 = _mm_load_ps(volt + idx);
                    __m128 v1 = _mm_load_ps(volt + stride_n + idx);
                    __m128 v2 = _mm_load_ps(volt + 2*stride_n + idx);
                    __m128 vv0 = _mm_load_ps(vv + idx);
                    __m128 vv1 = _mm_load_ps(vv + stride_n + idx);
                    __m128 vv2 = _mm_load_ps(vv + 2*stride_n + idx);
                    v0 = _mm_mul_ps(v0, vv0);
                    v1 = _mm_mul_ps(v1, vv1);
                    v2 = _mm_mul_ps(v2, vv2);
                    _mm_store_ps(volt + idx, v0);
                    _mm_store_ps(volt + stride_n + idx, v1);
                    _mm_store_ps(volt + 2*stride_n + idx, v2);
                }
            }
        }
    }

    void UpdateCurrents() {
        const size_t stride_n = (size_t)nx * ny * nz;
        const size_t stride_x = (size_t)ny * nz;
        const size_t stride_y = nz;

        for (unsigned int x = 0; x < nx - 1; ++x) {
            for (unsigned int y = 0; y < ny - 1; ++y) {
                size_t base_xy = x * stride_x + y * stride_y;

                for (unsigned int z = 0; z < nz - 4; z += 4) {
                    size_t idx = base_xy + z;

                    __m128 c0 = _mm_load_ps(curr + idx);
                    __m128 ii0 = _mm_load_ps(ii + idx);
                    __m128 iv0 = _mm_load_ps(iv + idx);

                    __m128 v2_y = _mm_load_ps(volt + 2*stride_n + idx);
                    __m128 v2_yp = _mm_load_ps(volt + 2*stride_n + idx + stride_y);
                    __m128 v1_z = _mm_load_ps(volt + stride_n + idx);
                    __m128 v1_zp = _mm_load_ps(volt + stride_n + idx + 4);

                    __m128 curl0 = _mm_sub_ps(_mm_sub_ps(_mm_add_ps(v2_y, v1_zp), v2_yp), v1_z);
                    c0 = _mm_add_ps(_mm_mul_ps(c0, ii0), _mm_mul_ps(iv0, curl0));
                    _mm_store_ps(curr + idx, c0);

                    __m128 c1 = _mm_load_ps(curr + stride_n + idx);
                    __m128 ii1 = _mm_load_ps(ii + stride_n + idx);
                    __m128 iv1 = _mm_load_ps(iv + stride_n + idx);

                    __m128 v0_z = _mm_load_ps(volt + idx);
                    __m128 v0_zp = _mm_load_ps(volt + idx + 4);
                    __m128 v2_x = _mm_load_ps(volt + 2*stride_n + idx);
                    __m128 v2_xp = _mm_load_ps(volt + 2*stride_n + idx + stride_x);

                    __m128 curl1 = _mm_sub_ps(_mm_sub_ps(_mm_add_ps(v0_z, v2_xp), v0_zp), v2_x);
                    c1 = _mm_add_ps(_mm_mul_ps(c1, ii1), _mm_mul_ps(iv1, curl1));
                    _mm_store_ps(curr + stride_n + idx, c1);

                    __m128 c2 = _mm_load_ps(curr + 2*stride_n + idx);
                    __m128 ii2 = _mm_load_ps(ii + 2*stride_n + idx);
                    __m128 iv2 = _mm_load_ps(iv + 2*stride_n + idx);

                    __m128 v1_x = _mm_load_ps(volt + stride_n + idx);
                    __m128 v1_xp = _mm_load_ps(volt + stride_n + idx + stride_x);
                    __m128 v0_y = _mm_load_ps(volt + idx);
                    __m128 v0_yp = _mm_load_ps(volt + idx + stride_y);

                    __m128 curl2 = _mm_sub_ps(_mm_sub_ps(_mm_add_ps(v1_x, v0_yp), v1_xp), v0_y);
                    c2 = _mm_add_ps(_mm_mul_ps(c2, ii2), _mm_mul_ps(iv2, curl2));
                    _mm_store_ps(curr + 2*stride_n + idx, c2);
                }
            }
        }
    }
};
#endif

//=============================================================================
// Benchmark runner
//=============================================================================
template<typename Engine>
double benchmark(const char* name, unsigned int nx, unsigned int ny, unsigned int nz,
                 unsigned int iterations, unsigned int warmup) {
    Engine engine(nx, ny, nz);
    Timer timer;

    for (unsigned int i = 0; i < warmup; ++i) {
        engine.UpdateVoltages();
        engine.UpdateCurrents();
    }

    timer.start();
    for (unsigned int i = 0; i < iterations; ++i) {
        engine.UpdateVoltages();
        engine.UpdateCurrents();
    }
    timer.stop();

    double per_iter_us = timer.elapsed_us() / iterations;
    double cells = (double)nx * ny * nz;
    double mcps = (cells * iterations) / timer.elapsed_us();

    cout << setw(25) << left << name
         << setw(15) << right << fixed << setprecision(1) << per_iter_us << " us/iter"
         << setw(15) << right << fixed << setprecision(2) << mcps << " MCPS"
         << endl;

    return per_iter_us;
}

int main(int argc, char* argv[]) {
#ifdef __SSE__
    _mm_setcsr(_mm_getcsr() | 0x8040);  // FTZ + DAZ
#endif

    unsigned int nx = 100, ny = 100, nz = 128;
    unsigned int iterations = 100, warmup = 10;

    if (argc >= 4) {
        nx = atoi(argv[1]);
        ny = atoi(argv[2]);
        nz = atoi(argv[3]);
    }
    if (argc >= 5) {
        iterations = atoi(argv[4]);
    }

    // Get Highway vector width
    const D d;
    const size_t lanes = hn::Lanes(d);

    // Round nz up to multiple of lanes for alignment
    nz = ((nz + lanes - 1) / lanes) * lanes;

    cout << "=== Highway SIMD Benchmark ===" << endl;
    cout << "Highway vector width: " << lanes << " floats" << endl;
#if HWY_TARGET == HWY_AVX3_ZEN4
    cout << "Target: AVX-512 (Zen4)" << endl;
#elif HWY_TARGET == HWY_AVX3_SPR
    cout << "Target: AVX-512 (Sapphire Rapids)" << endl;
#elif HWY_TARGET == HWY_AVX3
    cout << "Target: AVX-512" << endl;
#elif HWY_TARGET == HWY_AVX2
    cout << "Target: AVX2" << endl;
#elif HWY_TARGET == HWY_SSE4
    cout << "Target: SSE4" << endl;
#elif HWY_TARGET == HWY_SSE2
    cout << "Target: SSE2" << endl;
#else
    cout << "Target: Unknown" << endl;
#endif
    cout << "Grid size: " << nx << " x " << ny << " x " << nz << endl;
    cout << "Iterations: " << iterations << " (warmup: " << warmup << ")" << endl;
    cout << endl;

    cout << setw(25) << left << "Engine"
         << setw(18) << right << "Per Iter"
         << setw(18) << right << "Throughput"
         << endl;
    cout << string(61, '-') << endl;

#ifdef __SSE2__
    double sse_time = benchmark<Engine_SSE_Ref>("SSE (hand-written)", nx, ny, nz, iterations, warmup);
#endif

    double hwy_time = benchmark<Engine_Highway>("Highway (portable)", nx, ny, nz, iterations, warmup);

    cout << string(61, '-') << endl;

#ifdef __SSE2__
    cout << endl << "Highway speedup vs SSE: " << fixed << setprecision(2) << sse_time / hwy_time << "x" << endl;

    if (lanes > 4) {
        cout << "Expected speedup from wider vectors: " << fixed << setprecision(2) << (float)lanes / 4.0f << "x" << endl;
    }
#endif

    return 0;
}
