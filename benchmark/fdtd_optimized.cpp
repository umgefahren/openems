/*
 * FDTD Optimized Implementations
 * Tests additional optimization techniques:
 * - Software prefetching
 * - FMA instructions
 * - OpenMP parallelization
 * - Better memory layout
 */

#include <iostream>
#include <chrono>
#include <vector>
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

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;
using namespace std::chrono;

using FDTD_FLOAT = float;

union f4vector {
    __m128 v;
    float f[4];
};

#ifdef __AVX__
union f8vector {
    __m256 v;
    float f[8];
};
#endif

template<typename T, size_t Align = 64>
T* aligned_alloc_array(size_t n) {
    void* ptr = nullptr;
    if (posix_memalign(&ptr, Align, n * sizeof(T)) != 0) {
        throw std::bad_alloc();
    }
    memset(ptr, 0, n * sizeof(T));
    return static_cast<T*>(ptr);
}

template<typename T>
void aligned_free(T* ptr) {
    free(ptr);
}

class Timer {
public:
    void start() { m_start = high_resolution_clock::now(); }
    void stop() { m_end = high_resolution_clock::now(); }
    double elapsed_ms() const {
        return duration_cast<microseconds>(m_end - m_start).count() / 1000.0;
    }
    double elapsed_us() const {
        return duration_cast<microseconds>(m_end - m_start).count();
    }
private:
    time_point<high_resolution_clock> m_start, m_end;
};

//=============================================================================
// OPTIMIZATION 1: SSE with software prefetching
//=============================================================================
class Engine_SSE_Prefetch {
public:
    unsigned int numLines[3];
    unsigned int numVectors;
    f4vector* f4_volt;
    f4vector* f4_curr;
    f4vector* f4_vv;
    f4vector* f4_vi;
    f4vector* f4_ii;
    f4vector* f4_iv;

    static constexpr int PREFETCH_DISTANCE = 16;

    Engine_SSE_Prefetch(unsigned int nx, unsigned int ny, unsigned int nz) {
        numLines[0] = nx;
        numLines[1] = ny;
        numLines[2] = nz;
        numVectors = (nz + 3) / 4;

        size_t total = 3ULL * nx * ny * numVectors;
        f4_volt = aligned_alloc_array<f4vector>(total);
        f4_curr = aligned_alloc_array<f4vector>(total);
        f4_vv = aligned_alloc_array<f4vector>(total);
        f4_vi = aligned_alloc_array<f4vector>(total);
        f4_ii = aligned_alloc_array<f4vector>(total);
        f4_iv = aligned_alloc_array<f4vector>(total);

        for (size_t i = 0; i < total; i++) {
            for (int j = 0; j < 4; j++) {
                f4_volt[i].f[j] = 0.001f * ((i * 4 + j) % 1000);
                f4_curr[i].f[j] = 0.001f * (((i * 4 + j) + 500) % 1000);
                f4_vv[i].f[j] = 0.999f;
                f4_vi[i].f[j] = 0.001f;
                f4_ii[i].f[j] = 0.999f;
                f4_iv[i].f[j] = 0.001f;
            }
        }
    }

    ~Engine_SSE_Prefetch() {
        aligned_free(f4_volt);
        aligned_free(f4_curr);
        aligned_free(f4_vv);
        aligned_free(f4_vi);
        aligned_free(f4_ii);
        aligned_free(f4_iv);
    }

    inline size_t idx(unsigned int n, unsigned int x, unsigned int y, unsigned int z) const {
        return n * numLines[0] * numLines[1] * numVectors +
               x * numLines[1] * numVectors +
               y * numVectors + z;
    }

    void UpdateVoltages() {
#ifdef __SSE2__
        const int stride_y = numVectors;
        const int stride_x = numLines[1] * numVectors;
        const int stride_n = numLines[0] * numLines[1] * numVectors;

        for (unsigned int x = 0; x < numLines[0]; ++x) {
            int shift_x = (x > 0) ? stride_x : 0;
            for (unsigned int y = 0; y < numLines[1]; ++y) {
                int shift_y = (y > 0) ? stride_y : 0;

                // Prefetch next y-line
                if (y + 1 < numLines[1]) {
                    _mm_prefetch((char*)&f4_curr[(y + 1) * stride_y + x * stride_x], _MM_HINT_T0);
                    _mm_prefetch((char*)&f4_volt[(y + 1) * stride_y + x * stride_x], _MM_HINT_T0);
                }

                for (unsigned int z = 1; z < numVectors; ++z) {
                    // Prefetch ahead in z-direction
                    if (z + PREFETCH_DISTANCE < numVectors) {
                        _mm_prefetch((char*)&f4_curr[x * stride_x + y * stride_y + z + PREFETCH_DISTANCE], _MM_HINT_T0);
                        _mm_prefetch((char*)&f4_vv[x * stride_x + y * stride_y + z + PREFETCH_DISTANCE], _MM_HINT_T0);
                    }

                    size_t base = x * stride_x + y * stride_y + z;
                    size_t i0 = base;
                    size_t i1 = base + stride_n;
                    size_t i2 = base + 2 * stride_n;

                    // x-polarization
                    f4_volt[i0].v = _mm_mul_ps(f4_volt[i0].v, f4_vv[i0].v);
                    f4_volt[i0].v = _mm_add_ps(f4_volt[i0].v,
                        _mm_mul_ps(f4_vi[i0].v,
                            _mm_sub_ps(
                                _mm_sub_ps(
                                    _mm_add_ps(f4_curr[i2].v, f4_curr[i1 - 1].v),
                                    f4_curr[i2 - shift_y].v),
                                f4_curr[i1].v)));

                    // y-polarization
                    f4_volt[i1].v = _mm_mul_ps(f4_volt[i1].v, f4_vv[i1].v);
                    f4_volt[i1].v = _mm_add_ps(f4_volt[i1].v,
                        _mm_mul_ps(f4_vi[i1].v,
                            _mm_sub_ps(
                                _mm_sub_ps(
                                    _mm_add_ps(f4_curr[i0].v, f4_curr[i2 - shift_x].v),
                                    f4_curr[i0 - 1].v),
                                f4_curr[i2].v)));

                    // z-polarization
                    f4_volt[i2].v = _mm_mul_ps(f4_volt[i2].v, f4_vv[i2].v);
                    f4_volt[i2].v = _mm_add_ps(f4_volt[i2].v,
                        _mm_mul_ps(f4_vi[i2].v,
                            _mm_sub_ps(
                                _mm_sub_ps(
                                    _mm_add_ps(f4_curr[i1].v, f4_curr[i0 - shift_y].v),
                                    f4_curr[i1 - shift_x].v),
                                f4_curr[i0].v)));
                }

                // z=0 boundary simplified
                {
                    size_t base = x * stride_x + y * stride_y;
                    f4_volt[base].v = _mm_mul_ps(f4_volt[base].v, f4_vv[base].v);
                    f4_volt[base + stride_n].v = _mm_mul_ps(f4_volt[base + stride_n].v, f4_vv[base + stride_n].v);
                    f4_volt[base + 2*stride_n].v = _mm_mul_ps(f4_volt[base + 2*stride_n].v, f4_vv[base + 2*stride_n].v);
                }
            }
        }
#endif
    }

    void UpdateCurrents() {
#ifdef __SSE2__
        const int stride_y = numVectors;
        const int stride_x = numLines[1] * numVectors;
        const int stride_n = numLines[0] * numLines[1] * numVectors;

        for (unsigned int x = 0; x < numLines[0] - 1; ++x) {
            for (unsigned int y = 0; y < numLines[1] - 1; ++y) {
                // Prefetch next y-line
                if (y + 1 < numLines[1] - 1) {
                    _mm_prefetch((char*)&f4_volt[(y + 1) * stride_y + x * stride_x], _MM_HINT_T0);
                    _mm_prefetch((char*)&f4_curr[(y + 1) * stride_y + x * stride_x], _MM_HINT_T0);
                }

                for (unsigned int z = 0; z < numVectors - 1; ++z) {
                    // Prefetch ahead
                    if (z + PREFETCH_DISTANCE < numVectors - 1) {
                        _mm_prefetch((char*)&f4_volt[x * stride_x + y * stride_y + z + PREFETCH_DISTANCE], _MM_HINT_T0);
                        _mm_prefetch((char*)&f4_ii[x * stride_x + y * stride_y + z + PREFETCH_DISTANCE], _MM_HINT_T0);
                    }

                    size_t base = x * stride_x + y * stride_y + z;
                    size_t i0 = base;
                    size_t i1 = base + stride_n;
                    size_t i2 = base + 2 * stride_n;

                    // x-polarization
                    f4_curr[i0].v = _mm_mul_ps(f4_curr[i0].v, f4_ii[i0].v);
                    f4_curr[i0].v = _mm_add_ps(f4_curr[i0].v,
                        _mm_mul_ps(f4_iv[i0].v,
                            _mm_sub_ps(
                                _mm_sub_ps(
                                    _mm_add_ps(f4_volt[i2].v, f4_volt[i1 + 1].v),
                                    f4_volt[i2 + stride_y].v),
                                f4_volt[i1].v)));

                    // y-polarization
                    f4_curr[i1].v = _mm_mul_ps(f4_curr[i1].v, f4_ii[i1].v);
                    f4_curr[i1].v = _mm_add_ps(f4_curr[i1].v,
                        _mm_mul_ps(f4_iv[i1].v,
                            _mm_sub_ps(
                                _mm_sub_ps(
                                    _mm_add_ps(f4_volt[i0].v, f4_volt[i2 + stride_x].v),
                                    f4_volt[i0 + 1].v),
                                f4_volt[i2].v)));

                    // z-polarization
                    f4_curr[i2].v = _mm_mul_ps(f4_curr[i2].v, f4_ii[i2].v);
                    f4_curr[i2].v = _mm_add_ps(f4_curr[i2].v,
                        _mm_mul_ps(f4_iv[i2].v,
                            _mm_sub_ps(
                                _mm_sub_ps(
                                    _mm_add_ps(f4_volt[i1].v, f4_volt[i0 + stride_y].v),
                                    f4_volt[i1 + stride_x].v),
                                f4_volt[i0].v)));
                }

                // z = numVectors-1 boundary simplified
                {
                    size_t z = numVectors - 1;
                    size_t base = x * stride_x + y * stride_y + z;
                    f4_curr[base].v = _mm_mul_ps(f4_curr[base].v, f4_ii[base].v);
                    f4_curr[base + stride_n].v = _mm_mul_ps(f4_curr[base + stride_n].v, f4_ii[base + stride_n].v);
                    f4_curr[base + 2*stride_n].v = _mm_mul_ps(f4_curr[base + 2*stride_n].v, f4_ii[base + 2*stride_n].v);
                }
            }
        }
#endif
    }
};

//=============================================================================
// OPTIMIZATION 2: AVX with FMA (Fused Multiply-Add)
//=============================================================================
#if defined(__AVX__) && defined(__FMA__)
class Engine_AVX_FMA {
public:
    unsigned int numLines[3];
    unsigned int numVectors;
    f8vector* f8_volt;
    f8vector* f8_curr;
    f8vector* f8_vv;
    f8vector* f8_vi;
    f8vector* f8_ii;
    f8vector* f8_iv;

    Engine_AVX_FMA(unsigned int nx, unsigned int ny, unsigned int nz) {
        numLines[0] = nx;
        numLines[1] = ny;
        numLines[2] = nz;
        numVectors = (nz + 7) / 8;

        size_t total = 3ULL * nx * ny * numVectors;
        f8_volt = aligned_alloc_array<f8vector>(total);
        f8_curr = aligned_alloc_array<f8vector>(total);
        f8_vv = aligned_alloc_array<f8vector>(total);
        f8_vi = aligned_alloc_array<f8vector>(total);
        f8_ii = aligned_alloc_array<f8vector>(total);
        f8_iv = aligned_alloc_array<f8vector>(total);

        for (size_t i = 0; i < total; i++) {
            for (int j = 0; j < 8; j++) {
                f8_volt[i].f[j] = 0.001f * ((i * 8 + j) % 1000);
                f8_curr[i].f[j] = 0.001f * (((i * 8 + j) + 500) % 1000);
                f8_vv[i].f[j] = 0.999f;
                f8_vi[i].f[j] = 0.001f;
                f8_ii[i].f[j] = 0.999f;
                f8_iv[i].f[j] = 0.001f;
            }
        }
    }

    ~Engine_AVX_FMA() {
        aligned_free(f8_volt);
        aligned_free(f8_curr);
        aligned_free(f8_vv);
        aligned_free(f8_vi);
        aligned_free(f8_ii);
        aligned_free(f8_iv);
    }

    inline size_t idx(unsigned int n, unsigned int x, unsigned int y, unsigned int z) const {
        return n * numLines[0] * numLines[1] * numVectors +
               x * numLines[1] * numVectors +
               y * numVectors + z;
    }

    void UpdateVoltages() {
        const int stride_y = numVectors;
        const int stride_x = numLines[1] * numVectors;
        const int stride_n = numLines[0] * numLines[1] * numVectors;

        for (unsigned int x = 0; x < numLines[0]; ++x) {
            int shift_x = (x > 0) ? stride_x : 0;
            for (unsigned int y = 0; y < numLines[1]; ++y) {
                int shift_y = (y > 0) ? stride_y : 0;

                for (unsigned int z = 1; z < numVectors; ++z) {
                    size_t base = x * stride_x + y * stride_y + z;
                    size_t i0 = base;
                    size_t i1 = base + stride_n;
                    size_t i2 = base + 2 * stride_n;

                    // Using FMA: a = a * b + c * d  becomes fmadd(c, d, a * b)
                    __m256 curl_x = _mm256_sub_ps(
                        _mm256_sub_ps(
                            _mm256_add_ps(f8_curr[i2].v, f8_curr[i1 - 1].v),
                            f8_curr[i2 - shift_y].v),
                        f8_curr[i1].v);
                    f8_volt[i0].v = _mm256_fmadd_ps(f8_vi[i0].v, curl_x,
                        _mm256_mul_ps(f8_volt[i0].v, f8_vv[i0].v));

                    __m256 curl_y = _mm256_sub_ps(
                        _mm256_sub_ps(
                            _mm256_add_ps(f8_curr[i0].v, f8_curr[i2 - shift_x].v),
                            f8_curr[i0 - 1].v),
                        f8_curr[i2].v);
                    f8_volt[i1].v = _mm256_fmadd_ps(f8_vi[i1].v, curl_y,
                        _mm256_mul_ps(f8_volt[i1].v, f8_vv[i1].v));

                    __m256 curl_z = _mm256_sub_ps(
                        _mm256_sub_ps(
                            _mm256_add_ps(f8_curr[i1].v, f8_curr[i0 - shift_y].v),
                            f8_curr[i1 - shift_x].v),
                        f8_curr[i0].v);
                    f8_volt[i2].v = _mm256_fmadd_ps(f8_vi[i2].v, curl_z,
                        _mm256_mul_ps(f8_volt[i2].v, f8_vv[i2].v));
                }

                // Boundary at z=0
                {
                    size_t base = x * stride_x + y * stride_y;
                    f8_volt[base].v = _mm256_mul_ps(f8_volt[base].v, f8_vv[base].v);
                    f8_volt[base + stride_n].v = _mm256_mul_ps(f8_volt[base + stride_n].v, f8_vv[base + stride_n].v);
                    f8_volt[base + 2*stride_n].v = _mm256_mul_ps(f8_volt[base + 2*stride_n].v, f8_vv[base + 2*stride_n].v);
                }
            }
        }
    }

    void UpdateCurrents() {
        const int stride_y = numVectors;
        const int stride_x = numLines[1] * numVectors;
        const int stride_n = numLines[0] * numLines[1] * numVectors;

        for (unsigned int x = 0; x < numLines[0] - 1; ++x) {
            for (unsigned int y = 0; y < numLines[1] - 1; ++y) {
                for (unsigned int z = 0; z < numVectors - 1; ++z) {
                    size_t base = x * stride_x + y * stride_y + z;
                    size_t i0 = base;
                    size_t i1 = base + stride_n;
                    size_t i2 = base + 2 * stride_n;

                    __m256 curl_x = _mm256_sub_ps(
                        _mm256_sub_ps(
                            _mm256_add_ps(f8_volt[i2].v, f8_volt[i1 + 1].v),
                            f8_volt[i2 + stride_y].v),
                        f8_volt[i1].v);
                    f8_curr[i0].v = _mm256_fmadd_ps(f8_iv[i0].v, curl_x,
                        _mm256_mul_ps(f8_curr[i0].v, f8_ii[i0].v));

                    __m256 curl_y = _mm256_sub_ps(
                        _mm256_sub_ps(
                            _mm256_add_ps(f8_volt[i0].v, f8_volt[i2 + stride_x].v),
                            f8_volt[i0 + 1].v),
                        f8_volt[i2].v);
                    f8_curr[i1].v = _mm256_fmadd_ps(f8_iv[i1].v, curl_y,
                        _mm256_mul_ps(f8_curr[i1].v, f8_ii[i1].v));

                    __m256 curl_z = _mm256_sub_ps(
                        _mm256_sub_ps(
                            _mm256_add_ps(f8_volt[i1].v, f8_volt[i0 + stride_y].v),
                            f8_volt[i1 + stride_x].v),
                        f8_volt[i0].v);
                    f8_curr[i2].v = _mm256_fmadd_ps(f8_iv[i2].v, curl_z,
                        _mm256_mul_ps(f8_curr[i2].v, f8_ii[i2].v));
                }

                // z = numVectors-1 boundary
                {
                    size_t z = numVectors - 1;
                    size_t base = x * stride_x + y * stride_y + z;
                    f8_curr[base].v = _mm256_mul_ps(f8_curr[base].v, f8_ii[base].v);
                    f8_curr[base + stride_n].v = _mm256_mul_ps(f8_curr[base + stride_n].v, f8_ii[base + stride_n].v);
                    f8_curr[base + 2*stride_n].v = _mm256_mul_ps(f8_curr[base + 2*stride_n].v, f8_ii[base + 2*stride_n].v);
                }
            }
        }
    }
};
#endif

//=============================================================================
// OPTIMIZATION 3: OpenMP parallelization
//=============================================================================
#ifdef _OPENMP
class Engine_OpenMP {
public:
    unsigned int numLines[3];
    unsigned int numVectors;
    f4vector* f4_volt;
    f4vector* f4_curr;
    f4vector* f4_vv;
    f4vector* f4_vi;
    f4vector* f4_ii;
    f4vector* f4_iv;

    Engine_OpenMP(unsigned int nx, unsigned int ny, unsigned int nz) {
        numLines[0] = nx;
        numLines[1] = ny;
        numLines[2] = nz;
        numVectors = (nz + 3) / 4;

        size_t total = 3ULL * nx * ny * numVectors;
        f4_volt = aligned_alloc_array<f4vector>(total);
        f4_curr = aligned_alloc_array<f4vector>(total);
        f4_vv = aligned_alloc_array<f4vector>(total);
        f4_vi = aligned_alloc_array<f4vector>(total);
        f4_ii = aligned_alloc_array<f4vector>(total);
        f4_iv = aligned_alloc_array<f4vector>(total);

        for (size_t i = 0; i < total; i++) {
            for (int j = 0; j < 4; j++) {
                f4_volt[i].f[j] = 0.001f * ((i * 4 + j) % 1000);
                f4_curr[i].f[j] = 0.001f * (((i * 4 + j) + 500) % 1000);
                f4_vv[i].f[j] = 0.999f;
                f4_vi[i].f[j] = 0.001f;
                f4_ii[i].f[j] = 0.999f;
                f4_iv[i].f[j] = 0.001f;
            }
        }
    }

    ~Engine_OpenMP() {
        aligned_free(f4_volt);
        aligned_free(f4_curr);
        aligned_free(f4_vv);
        aligned_free(f4_vi);
        aligned_free(f4_ii);
        aligned_free(f4_iv);
    }

    inline size_t idx(unsigned int n, unsigned int x, unsigned int y, unsigned int z) const {
        return n * numLines[0] * numLines[1] * numVectors +
               x * numLines[1] * numVectors +
               y * numVectors + z;
    }

    void UpdateVoltages() {
#ifdef __SSE2__
        const int stride_y = numVectors;
        const int stride_x = numLines[1] * numVectors;
        const int stride_n = numLines[0] * numLines[1] * numVectors;

        #pragma omp parallel for schedule(static)
        for (unsigned int x = 0; x < numLines[0]; ++x) {
            int shift_x = (x > 0) ? stride_x : 0;
            for (unsigned int y = 0; y < numLines[1]; ++y) {
                int shift_y = (y > 0) ? stride_y : 0;

                for (unsigned int z = 1; z < numVectors; ++z) {
                    size_t base = x * stride_x + y * stride_y + z;
                    size_t i0 = base;
                    size_t i1 = base + stride_n;
                    size_t i2 = base + 2 * stride_n;

                    f4_volt[i0].v = _mm_mul_ps(f4_volt[i0].v, f4_vv[i0].v);
                    f4_volt[i0].v = _mm_add_ps(f4_volt[i0].v,
                        _mm_mul_ps(f4_vi[i0].v,
                            _mm_sub_ps(
                                _mm_sub_ps(
                                    _mm_add_ps(f4_curr[i2].v, f4_curr[i1 - 1].v),
                                    f4_curr[i2 - shift_y].v),
                                f4_curr[i1].v)));

                    f4_volt[i1].v = _mm_mul_ps(f4_volt[i1].v, f4_vv[i1].v);
                    f4_volt[i1].v = _mm_add_ps(f4_volt[i1].v,
                        _mm_mul_ps(f4_vi[i1].v,
                            _mm_sub_ps(
                                _mm_sub_ps(
                                    _mm_add_ps(f4_curr[i0].v, f4_curr[i2 - shift_x].v),
                                    f4_curr[i0 - 1].v),
                                f4_curr[i2].v)));

                    f4_volt[i2].v = _mm_mul_ps(f4_volt[i2].v, f4_vv[i2].v);
                    f4_volt[i2].v = _mm_add_ps(f4_volt[i2].v,
                        _mm_mul_ps(f4_vi[i2].v,
                            _mm_sub_ps(
                                _mm_sub_ps(
                                    _mm_add_ps(f4_curr[i1].v, f4_curr[i0 - shift_y].v),
                                    f4_curr[i1 - shift_x].v),
                                f4_curr[i0].v)));
                }

                // z=0 boundary
                {
                    size_t base = x * stride_x + y * stride_y;
                    f4_volt[base].v = _mm_mul_ps(f4_volt[base].v, f4_vv[base].v);
                    f4_volt[base + stride_n].v = _mm_mul_ps(f4_volt[base + stride_n].v, f4_vv[base + stride_n].v);
                    f4_volt[base + 2*stride_n].v = _mm_mul_ps(f4_volt[base + 2*stride_n].v, f4_vv[base + 2*stride_n].v);
                }
            }
        }
#endif
    }

    void UpdateCurrents() {
#ifdef __SSE2__
        const int stride_y = numVectors;
        const int stride_x = numLines[1] * numVectors;
        const int stride_n = numLines[0] * numLines[1] * numVectors;

        #pragma omp parallel for schedule(static)
        for (unsigned int x = 0; x < numLines[0] - 1; ++x) {
            for (unsigned int y = 0; y < numLines[1] - 1; ++y) {
                for (unsigned int z = 0; z < numVectors - 1; ++z) {
                    size_t base = x * stride_x + y * stride_y + z;
                    size_t i0 = base;
                    size_t i1 = base + stride_n;
                    size_t i2 = base + 2 * stride_n;

                    f4_curr[i0].v = _mm_mul_ps(f4_curr[i0].v, f4_ii[i0].v);
                    f4_curr[i0].v = _mm_add_ps(f4_curr[i0].v,
                        _mm_mul_ps(f4_iv[i0].v,
                            _mm_sub_ps(
                                _mm_sub_ps(
                                    _mm_add_ps(f4_volt[i2].v, f4_volt[i1 + 1].v),
                                    f4_volt[i2 + stride_y].v),
                                f4_volt[i1].v)));

                    f4_curr[i1].v = _mm_mul_ps(f4_curr[i1].v, f4_ii[i1].v);
                    f4_curr[i1].v = _mm_add_ps(f4_curr[i1].v,
                        _mm_mul_ps(f4_iv[i1].v,
                            _mm_sub_ps(
                                _mm_sub_ps(
                                    _mm_add_ps(f4_volt[i0].v, f4_volt[i2 + stride_x].v),
                                    f4_volt[i0 + 1].v),
                                f4_volt[i2].v)));

                    f4_curr[i2].v = _mm_mul_ps(f4_curr[i2].v, f4_ii[i2].v);
                    f4_curr[i2].v = _mm_add_ps(f4_curr[i2].v,
                        _mm_mul_ps(f4_iv[i2].v,
                            _mm_sub_ps(
                                _mm_sub_ps(
                                    _mm_add_ps(f4_volt[i1].v, f4_volt[i0 + stride_y].v),
                                    f4_volt[i1 + stride_x].v),
                                f4_volt[i0].v)));
                }

                // z = numVectors-1 boundary
                {
                    size_t z = numVectors - 1;
                    size_t base = x * stride_x + y * stride_y + z;
                    f4_curr[base].v = _mm_mul_ps(f4_curr[base].v, f4_ii[base].v);
                    f4_curr[base + stride_n].v = _mm_mul_ps(f4_curr[base + stride_n].v, f4_ii[base + stride_n].v);
                    f4_curr[base + 2*stride_n].v = _mm_mul_ps(f4_curr[base + 2*stride_n].v, f4_ii[base + 2*stride_n].v);
                }
            }
        }
#endif
    }
};
#endif

//=============================================================================
// OPTIMIZATION 4: Interleaved component layout (AoS instead of SoA)
// Each cell stores [Ex, Ey, Ez] contiguously
//=============================================================================
class Engine_AoS {
public:
    unsigned int numLines[3];
    unsigned int numVectors;

    // AoS layout: [Ex, Ey, Ez, pad] for each cell
    struct CellData {
        __m128 v;  // x, y, z components + padding
    };

    CellData* volt;
    CellData* curr;
    CellData* vv;
    CellData* vi;
    CellData* ii;
    CellData* iv;

    Engine_AoS(unsigned int nx, unsigned int ny, unsigned int nz) {
        numLines[0] = nx;
        numLines[1] = ny;
        numLines[2] = nz;

        size_t total = nx * ny * nz;
        volt = aligned_alloc_array<CellData>(total);
        curr = aligned_alloc_array<CellData>(total);
        vv = aligned_alloc_array<CellData>(total);
        vi = aligned_alloc_array<CellData>(total);
        ii = aligned_alloc_array<CellData>(total);
        iv = aligned_alloc_array<CellData>(total);

        for (size_t i = 0; i < total; i++) {
            float* vf = (float*)&volt[i].v;
            float* cf = (float*)&curr[i].v;
            float* vvf = (float*)&vv[i].v;
            float* vif = (float*)&vi[i].v;
            float* iif = (float*)&ii[i].v;
            float* ivf = (float*)&iv[i].v;
            for (int j = 0; j < 4; j++) {
                vf[j] = 0.001f * ((i * 4 + j) % 1000);
                cf[j] = 0.001f * (((i * 4 + j) + 500) % 1000);
                vvf[j] = 0.999f;
                vif[j] = 0.001f;
                iif[j] = 0.999f;
                ivf[j] = 0.001f;
            }
        }
    }

    ~Engine_AoS() {
        aligned_free(volt);
        aligned_free(curr);
        aligned_free(vv);
        aligned_free(vi);
        aligned_free(ii);
        aligned_free(iv);
    }

    inline size_t idx(unsigned int x, unsigned int y, unsigned int z) const {
        return x * numLines[1] * numLines[2] + y * numLines[2] + z;
    }

    void UpdateVoltages() {
#ifdef __SSE2__
        for (unsigned int x = 0; x < numLines[0]; ++x) {
            bool shift_x = x > 0;
            for (unsigned int y = 0; y < numLines[1]; ++y) {
                bool shift_y = y > 0;
                for (unsigned int z = 0; z < numLines[2]; ++z) {
                    bool shift_z = z > 0;

                    size_t i = idx(x, y, z);
                    size_t i_xm = idx(x - shift_x, y, z);
                    size_t i_ym = idx(x, y - shift_y, z);
                    size_t i_zm = idx(x, y, z - shift_z);

                    // This layout is not optimal for FDTD due to curl dependencies
                    // But demonstrates AoS approach
                    volt[i].v = _mm_mul_ps(volt[i].v, vv[i].v);
                    // Simplified - actual curl computation needs reordering
                }
            }
        }
#endif
    }

    void UpdateCurrents() {
#ifdef __SSE2__
        for (unsigned int x = 0; x < numLines[0] - 1; ++x) {
            for (unsigned int y = 0; y < numLines[1] - 1; ++y) {
                for (unsigned int z = 0; z < numLines[2] - 1; ++z) {
                    size_t i = idx(x, y, z);
                    curr[i].v = _mm_mul_ps(curr[i].v, ii[i].v);
                }
            }
        }
#endif
    }
};

//=============================================================================
// Benchmark runner
//=============================================================================
template<typename Engine>
double benchmark_engine(const char* name, unsigned int nx, unsigned int ny, unsigned int nz,
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

    double total_ms = timer.elapsed_ms();
    double per_iter_us = timer.elapsed_us() / iterations;
    double cells = (double)nx * ny * nz;
    double mcps = (cells * iterations) / (timer.elapsed_us());

    cout << setw(25) << left << name
         << setw(12) << right << fixed << setprecision(2) << total_ms << " ms"
         << setw(12) << right << fixed << setprecision(1) << per_iter_us << " us/iter"
         << setw(12) << right << fixed << setprecision(2) << mcps << " MCPS"
         << endl;

    return per_iter_us;
}

int main(int argc, char* argv[]) {
#ifdef __SSE__
    _mm_setcsr(_mm_getcsr() | 0x8040);
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

    cout << "=== Optimized FDTD Benchmark ===" << endl;
    cout << "Grid size: " << nx << " x " << ny << " x " << nz << endl;
    cout << "Total cells: " << (nx * ny * nz) << endl;
    cout << "Iterations: " << iterations << " (warmup: " << warmup << ")" << endl;
    cout << endl;

#ifdef _OPENMP
    cout << "OpenMP threads: " << omp_get_max_threads() << endl;
    cout << endl;
#endif

    cout << setw(25) << left << "Engine"
         << setw(15) << right << "Total"
         << setw(15) << right << "Per Iter"
         << setw(15) << right << "Throughput"
         << endl;
    cout << string(70, '-') << endl;

#ifdef __SSE2__
    double prefetch_time = benchmark_engine<Engine_SSE_Prefetch>("SSE + prefetch", nx, ny, nz, iterations, warmup);
#endif

#if defined(__AVX__) && defined(__FMA__)
    double avx_fma_time = benchmark_engine<Engine_AVX_FMA>("AVX + FMA", nx, ny, nz, iterations, warmup);
#endif

#ifdef _OPENMP
    double openmp_time = benchmark_engine<Engine_OpenMP>("SSE + OpenMP", nx, ny, nz, iterations, warmup);
#endif

    double aos_time = benchmark_engine<Engine_AoS>("AoS layout", nx, ny, nz, iterations, warmup);

    cout << string(70, '-') << endl;

    return 0;
}
