/*
 * FDTD Critical Path Micro-Benchmark
 * Tests different implementations of the voltage/current update loops
 * to identify optimization opportunities.
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

// Flush denormals to zero for performance
#ifdef __SSE__
#include <xmmintrin.h>
#endif

using namespace std;
using namespace std::chrono;

// Grid dimensions for benchmarking
constexpr unsigned int DEFAULT_NX = 100;
constexpr unsigned int DEFAULT_NY = 100;
constexpr unsigned int DEFAULT_NZ = 128;  // Multiple of 4 for SIMD
constexpr unsigned int NUM_ITERATIONS = 100;
constexpr unsigned int NUM_WARMUP = 10;

// FDTD_FLOAT matches openEMS
using FDTD_FLOAT = float;

// f4vector for SSE (matches openEMS)
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

// Memory-aligned allocator
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

// Timer class
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
// BENCHMARK 1: Basic scalar implementation (reference)
//=============================================================================
class Engine_Basic {
public:
    unsigned int numLines[3];
    FDTD_FLOAT* volt;  // 3 components * NX * NY * NZ
    FDTD_FLOAT* curr;
    FDTD_FLOAT* vv;    // operator coefficients
    FDTD_FLOAT* vi;
    FDTD_FLOAT* ii;
    FDTD_FLOAT* iv;

    Engine_Basic(unsigned int nx, unsigned int ny, unsigned int nz) {
        numLines[0] = nx;
        numLines[1] = ny;
        numLines[2] = nz;

        size_t total = 3ULL * nx * ny * nz;
        volt = aligned_alloc_array<FDTD_FLOAT>(total);
        curr = aligned_alloc_array<FDTD_FLOAT>(total);
        vv = aligned_alloc_array<FDTD_FLOAT>(total);
        vi = aligned_alloc_array<FDTD_FLOAT>(total);
        ii = aligned_alloc_array<FDTD_FLOAT>(total);
        iv = aligned_alloc_array<FDTD_FLOAT>(total);

        // Initialize with non-zero values
        for (size_t i = 0; i < total; i++) {
            volt[i] = 0.001f * (i % 1000);
            curr[i] = 0.001f * ((i + 500) % 1000);
            vv[i] = 0.999f;
            vi[i] = 0.001f;
            ii[i] = 0.999f;
            iv[i] = 0.001f;
        }
    }

    ~Engine_Basic() {
        aligned_free(volt);
        aligned_free(curr);
        aligned_free(vv);
        aligned_free(vi);
        aligned_free(ii);
        aligned_free(iv);
    }

    // N-I-J-K memory layout (component first)
    inline size_t idx(unsigned int n, unsigned int x, unsigned int y, unsigned int z) const {
        return n * numLines[0] * numLines[1] * numLines[2] +
               x * numLines[1] * numLines[2] +
               y * numLines[2] + z;
    }

    void UpdateVoltages() {
        for (unsigned int x = 0; x < numLines[0]; ++x) {
            bool shift_x = x > 0;
            for (unsigned int y = 0; y < numLines[1]; ++y) {
                bool shift_y = y > 0;
                for (unsigned int z = 0; z < numLines[2]; ++z) {
                    bool shift_z = z > 0;

                    // x-component
                    size_t i0 = idx(0, x, y, z);
                    volt[i0] *= vv[i0];
                    volt[i0] += vi[i0] * (
                        curr[idx(2, x, y, z)] -
                        curr[idx(2, x, y - shift_y, z)] -
                        curr[idx(1, x, y, z)] +
                        curr[idx(1, x, y, z - shift_z)]
                    );

                    // y-component
                    size_t i1 = idx(1, x, y, z);
                    volt[i1] *= vv[i1];
                    volt[i1] += vi[i1] * (
                        curr[idx(0, x, y, z)] -
                        curr[idx(0, x, y, z - shift_z)] -
                        curr[idx(2, x, y, z)] +
                        curr[idx(2, x - shift_x, y, z)]
                    );

                    // z-component
                    size_t i2 = idx(2, x, y, z);
                    volt[i2] *= vv[i2];
                    volt[i2] += vi[i2] * (
                        curr[idx(1, x, y, z)] -
                        curr[idx(1, x - shift_x, y, z)] -
                        curr[idx(0, x, y, z)] +
                        curr[idx(0, x, y - shift_y, z)]
                    );
                }
            }
        }
    }

    void UpdateCurrents() {
        for (unsigned int x = 0; x < numLines[0] - 1; ++x) {
            for (unsigned int y = 0; y < numLines[1] - 1; ++y) {
                for (unsigned int z = 0; z < numLines[2] - 1; ++z) {
                    // x-component
                    size_t i0 = idx(0, x, y, z);
                    curr[i0] *= ii[i0];
                    curr[i0] += iv[i0] * (
                        volt[idx(2, x, y, z)] -
                        volt[idx(2, x, y + 1, z)] -
                        volt[idx(1, x, y, z)] +
                        volt[idx(1, x, y, z + 1)]
                    );

                    // y-component
                    size_t i1 = idx(1, x, y, z);
                    curr[i1] *= ii[i1];
                    curr[i1] += iv[i1] * (
                        volt[idx(0, x, y, z)] -
                        volt[idx(0, x, y, z + 1)] -
                        volt[idx(2, x, y, z)] +
                        volt[idx(2, x + 1, y, z)]
                    );

                    // z-component
                    size_t i2 = idx(2, x, y, z);
                    curr[i2] *= ii[i2];
                    curr[i2] += iv[i2] * (
                        volt[idx(1, x, y, z)] -
                        volt[idx(1, x + 1, y, z)] -
                        volt[idx(0, x, y, z)] +
                        volt[idx(0, x, y + 1, z)]
                    );
                }
            }
        }
    }
};

//=============================================================================
// BENCHMARK 2: I-J-K-N memory layout (component interleaved - like ArrayENG)
//=============================================================================
class Engine_IJKN {
public:
    unsigned int numLines[3];
    FDTD_FLOAT* volt;
    FDTD_FLOAT* curr;
    FDTD_FLOAT* vv;
    FDTD_FLOAT* vi;
    FDTD_FLOAT* ii;
    FDTD_FLOAT* iv;

    Engine_IJKN(unsigned int nx, unsigned int ny, unsigned int nz) {
        numLines[0] = nx;
        numLines[1] = ny;
        numLines[2] = nz;

        size_t total = 3ULL * nx * ny * nz;
        volt = aligned_alloc_array<FDTD_FLOAT>(total);
        curr = aligned_alloc_array<FDTD_FLOAT>(total);
        vv = aligned_alloc_array<FDTD_FLOAT>(total);
        vi = aligned_alloc_array<FDTD_FLOAT>(total);
        ii = aligned_alloc_array<FDTD_FLOAT>(total);
        iv = aligned_alloc_array<FDTD_FLOAT>(total);

        for (size_t i = 0; i < total; i++) {
            volt[i] = 0.001f * (i % 1000);
            curr[i] = 0.001f * ((i + 500) % 1000);
            vv[i] = 0.999f;
            vi[i] = 0.001f;
            ii[i] = 0.999f;
            iv[i] = 0.001f;
        }
    }

    ~Engine_IJKN() {
        aligned_free(volt);
        aligned_free(curr);
        aligned_free(vv);
        aligned_free(vi);
        aligned_free(ii);
        aligned_free(iv);
    }

    // I-J-K-N memory layout (component last - interleaved)
    inline size_t idx(unsigned int n, unsigned int x, unsigned int y, unsigned int z) const {
        return x * numLines[1] * numLines[2] * 3 +
               y * numLines[2] * 3 +
               z * 3 + n;
    }

    void UpdateVoltages() {
        for (unsigned int x = 0; x < numLines[0]; ++x) {
            bool shift_x = x > 0;
            for (unsigned int y = 0; y < numLines[1]; ++y) {
                bool shift_y = y > 0;
                for (unsigned int z = 0; z < numLines[2]; ++z) {
                    bool shift_z = z > 0;

                    size_t i0 = idx(0, x, y, z);
                    volt[i0] *= vv[i0];
                    volt[i0] += vi[i0] * (
                        curr[idx(2, x, y, z)] -
                        curr[idx(2, x, y - shift_y, z)] -
                        curr[idx(1, x, y, z)] +
                        curr[idx(1, x, y, z - shift_z)]
                    );

                    size_t i1 = idx(1, x, y, z);
                    volt[i1] *= vv[i1];
                    volt[i1] += vi[i1] * (
                        curr[idx(0, x, y, z)] -
                        curr[idx(0, x, y, z - shift_z)] -
                        curr[idx(2, x, y, z)] +
                        curr[idx(2, x - shift_x, y, z)]
                    );

                    size_t i2 = idx(2, x, y, z);
                    volt[i2] *= vv[i2];
                    volt[i2] += vi[i2] * (
                        curr[idx(1, x, y, z)] -
                        curr[idx(1, x - shift_x, y, z)] -
                        curr[idx(0, x, y, z)] +
                        curr[idx(0, x, y - shift_y, z)]
                    );
                }
            }
        }
    }

    void UpdateCurrents() {
        for (unsigned int x = 0; x < numLines[0] - 1; ++x) {
            for (unsigned int y = 0; y < numLines[1] - 1; ++y) {
                for (unsigned int z = 0; z < numLines[2] - 1; ++z) {
                    size_t i0 = idx(0, x, y, z);
                    curr[i0] *= ii[i0];
                    curr[i0] += iv[i0] * (
                        volt[idx(2, x, y, z)] -
                        volt[idx(2, x, y + 1, z)] -
                        volt[idx(1, x, y, z)] +
                        volt[idx(1, x, y, z + 1)]
                    );

                    size_t i1 = idx(1, x, y, z);
                    curr[i1] *= ii[i1];
                    curr[i1] += iv[i1] * (
                        volt[idx(0, x, y, z)] -
                        volt[idx(0, x, y, z + 1)] -
                        volt[idx(2, x, y, z)] +
                        volt[idx(2, x + 1, y, z)]
                    );

                    size_t i2 = idx(2, x, y, z);
                    curr[i2] *= ii[i2];
                    curr[i2] += iv[i2] * (
                        volt[idx(1, x, y, z)] -
                        volt[idx(1, x + 1, y, z)] -
                        volt[idx(0, x, y, z)] +
                        volt[idx(0, x, y + 1, z)]
                    );
                }
            }
        }
    }
};

//=============================================================================
// BENCHMARK 3: SSE vectorized (similar to engine_sse.cpp)
//=============================================================================
class Engine_SSE {
public:
    unsigned int numLines[3];
    unsigned int numVectors;
    f4vector* f4_volt;
    f4vector* f4_curr;
    f4vector* f4_vv;
    f4vector* f4_vi;
    f4vector* f4_ii;
    f4vector* f4_iv;

    Engine_SSE(unsigned int nx, unsigned int ny, unsigned int nz) {
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

    ~Engine_SSE() {
        aligned_free(f4_volt);
        aligned_free(f4_curr);
        aligned_free(f4_vv);
        aligned_free(f4_vi);
        aligned_free(f4_ii);
        aligned_free(f4_iv);
    }

    // N-I-J-K layout with z-dimension vectorized
    inline size_t idx(unsigned int n, unsigned int x, unsigned int y, unsigned int z) const {
        return n * numLines[0] * numLines[1] * numVectors +
               x * numLines[1] * numVectors +
               y * numVectors + z;
    }

    void UpdateVoltages() {
#ifdef __SSE2__
        for (unsigned int x = 0; x < numLines[0]; ++x) {
            int shift_x = (x > 0) ? (numLines[1] * numVectors) : 0;
            for (unsigned int y = 0; y < numLines[1]; ++y) {
                int shift_y = (y > 0) ? numVectors : 0;

                // Main loop for z >= 1
                for (unsigned int z = 1; z < numVectors; ++z) {
                    size_t i0 = idx(0, x, y, z);
                    size_t i1 = idx(1, x, y, z);
                    size_t i2 = idx(2, x, y, z);

                    // x-polarization
                    f4_volt[i0].v = _mm_mul_ps(f4_volt[i0].v, f4_vv[i0].v);
                    f4_volt[i0].v = _mm_add_ps(f4_volt[i0].v,
                        _mm_mul_ps(f4_vi[i0].v,
                            _mm_sub_ps(
                                _mm_sub_ps(
                                    _mm_add_ps(f4_curr[idx(2, x, y, z)].v,
                                               f4_curr[idx(1, x, y, z - 1)].v),
                                    f4_curr[idx(2, x, y, z) - shift_y].v),
                                f4_curr[idx(1, x, y, z)].v)));

                    // y-polarization
                    f4_volt[i1].v = _mm_mul_ps(f4_volt[i1].v, f4_vv[i1].v);
                    f4_volt[i1].v = _mm_add_ps(f4_volt[i1].v,
                        _mm_mul_ps(f4_vi[i1].v,
                            _mm_sub_ps(
                                _mm_sub_ps(
                                    _mm_add_ps(f4_curr[idx(0, x, y, z)].v,
                                               f4_curr[idx(2, x, y, z) - shift_x].v),
                                    f4_curr[idx(0, x, y, z - 1)].v),
                                f4_curr[idx(2, x, y, z)].v)));

                    // z-polarization
                    f4_volt[i2].v = _mm_mul_ps(f4_volt[i2].v, f4_vv[i2].v);
                    f4_volt[i2].v = _mm_add_ps(f4_volt[i2].v,
                        _mm_mul_ps(f4_vi[i2].v,
                            _mm_sub_ps(
                                _mm_sub_ps(
                                    _mm_add_ps(f4_curr[idx(1, x, y, z)].v,
                                               f4_curr[idx(0, x, y, z) - shift_y].v),
                                    f4_curr[idx(1, x, y, z) - shift_x].v),
                                f4_curr[idx(0, x, y, z)].v)));
                }

                // Handle z = 0 boundary (shift from end of array)
                {
                    unsigned int z = 0;
                    size_t i0 = idx(0, x, y, 0);
                    size_t i1 = idx(1, x, y, 0);
                    size_t i2 = idx(2, x, y, 0);

                    f4vector temp;

                    // x-polarization with boundary handling
                    temp.v = (__m128)_mm_slli_si128((__m128i)f4_curr[idx(1, x, y, numVectors - 1)].v, 4);
                    f4_volt[i0].v = _mm_mul_ps(f4_volt[i0].v, f4_vv[i0].v);
                    f4_volt[i0].v = _mm_add_ps(f4_volt[i0].v,
                        _mm_mul_ps(f4_vi[i0].v,
                            _mm_sub_ps(
                                _mm_sub_ps(
                                    _mm_add_ps(f4_curr[idx(2, x, y, 0)].v, temp.v),
                                    f4_curr[idx(2, x, y, 0) - shift_y].v),
                                f4_curr[idx(1, x, y, 0)].v)));

                    temp.v = (__m128)_mm_slli_si128((__m128i)f4_curr[idx(0, x, y, numVectors - 1)].v, 4);
                    f4_volt[i1].v = _mm_mul_ps(f4_volt[i1].v, f4_vv[i1].v);
                    f4_volt[i1].v = _mm_add_ps(f4_volt[i1].v,
                        _mm_mul_ps(f4_vi[i1].v,
                            _mm_sub_ps(
                                _mm_sub_ps(
                                    _mm_add_ps(f4_curr[idx(0, x, y, 0)].v,
                                               f4_curr[idx(2, x, y, 0) - shift_x].v),
                                    temp.v),
                                f4_curr[idx(2, x, y, 0)].v)));

                    f4_volt[i2].v = _mm_mul_ps(f4_volt[i2].v, f4_vv[i2].v);
                    f4_volt[i2].v = _mm_add_ps(f4_volt[i2].v,
                        _mm_mul_ps(f4_vi[i2].v,
                            _mm_sub_ps(
                                _mm_sub_ps(
                                    _mm_add_ps(f4_curr[idx(1, x, y, 0)].v,
                                               f4_curr[idx(0, x, y, 0) - shift_y].v),
                                    f4_curr[idx(1, x, y, 0) - shift_x].v),
                                f4_curr[idx(0, x, y, 0)].v)));
                }
            }
        }
#endif
    }

    void UpdateCurrents() {
#ifdef __SSE2__
        for (unsigned int x = 0; x < numLines[0] - 1; ++x) {
            int shift_x = numLines[1] * numVectors;
            for (unsigned int y = 0; y < numLines[1] - 1; ++y) {
                int shift_y = numVectors;

                for (unsigned int z = 0; z < numVectors - 1; ++z) {
                    size_t i0 = idx(0, x, y, z);
                    size_t i1 = idx(1, x, y, z);
                    size_t i2 = idx(2, x, y, z);

                    // x-polarization
                    f4_curr[i0].v = _mm_mul_ps(f4_curr[i0].v, f4_ii[i0].v);
                    f4_curr[i0].v = _mm_add_ps(f4_curr[i0].v,
                        _mm_mul_ps(f4_iv[i0].v,
                            _mm_sub_ps(
                                _mm_sub_ps(
                                    _mm_add_ps(f4_volt[idx(2, x, y, z)].v,
                                               f4_volt[idx(1, x, y, z + 1)].v),
                                    f4_volt[idx(2, x, y, z) + shift_y].v),
                                f4_volt[idx(1, x, y, z)].v)));

                    // y-polarization
                    f4_curr[i1].v = _mm_mul_ps(f4_curr[i1].v, f4_ii[i1].v);
                    f4_curr[i1].v = _mm_add_ps(f4_curr[i1].v,
                        _mm_mul_ps(f4_iv[i1].v,
                            _mm_sub_ps(
                                _mm_sub_ps(
                                    _mm_add_ps(f4_volt[idx(0, x, y, z)].v,
                                               f4_volt[idx(2, x, y, z) + shift_x].v),
                                    f4_volt[idx(0, x, y, z + 1)].v),
                                f4_volt[idx(2, x, y, z)].v)));

                    // z-polarization
                    f4_curr[i2].v = _mm_mul_ps(f4_curr[i2].v, f4_ii[i2].v);
                    f4_curr[i2].v = _mm_add_ps(f4_curr[i2].v,
                        _mm_mul_ps(f4_iv[i2].v,
                            _mm_sub_ps(
                                _mm_sub_ps(
                                    _mm_add_ps(f4_volt[idx(1, x, y, z)].v,
                                               f4_volt[idx(0, x, y, z) + shift_y].v),
                                    f4_volt[idx(1, x, y, z) + shift_x].v),
                                f4_volt[idx(0, x, y, z)].v)));
                }

                // Handle z = numVectors - 1 boundary
                {
                    unsigned int z = numVectors - 1;
                    size_t i0 = idx(0, x, y, z);
                    size_t i1 = idx(1, x, y, z);
                    size_t i2 = idx(2, x, y, z);

                    f4vector temp;

                    temp.v = (__m128)_mm_srli_si128((__m128i)f4_volt[idx(1, x, y, 0)].v, 4);
                    f4_curr[i0].v = _mm_mul_ps(f4_curr[i0].v, f4_ii[i0].v);
                    f4_curr[i0].v = _mm_add_ps(f4_curr[i0].v,
                        _mm_mul_ps(f4_iv[i0].v,
                            _mm_sub_ps(
                                _mm_sub_ps(
                                    _mm_add_ps(f4_volt[idx(2, x, y, z)].v, temp.v),
                                    f4_volt[idx(2, x, y, z) + shift_y].v),
                                f4_volt[idx(1, x, y, z)].v)));

                    temp.v = (__m128)_mm_srli_si128((__m128i)f4_volt[idx(0, x, y, 0)].v, 4);
                    f4_curr[i1].v = _mm_mul_ps(f4_curr[i1].v, f4_ii[i1].v);
                    f4_curr[i1].v = _mm_add_ps(f4_curr[i1].v,
                        _mm_mul_ps(f4_iv[i1].v,
                            _mm_sub_ps(
                                _mm_sub_ps(
                                    _mm_add_ps(f4_volt[idx(0, x, y, z)].v,
                                               f4_volt[idx(2, x, y, z) + shift_x].v),
                                    temp.v),
                                f4_volt[idx(2, x, y, z)].v)));

                    f4_curr[i2].v = _mm_mul_ps(f4_curr[i2].v, f4_ii[i2].v);
                    f4_curr[i2].v = _mm_add_ps(f4_curr[i2].v,
                        _mm_mul_ps(f4_iv[i2].v,
                            _mm_sub_ps(
                                _mm_sub_ps(
                                    _mm_add_ps(f4_volt[idx(1, x, y, z)].v,
                                               f4_volt[idx(0, x, y, z) + shift_y].v),
                                    f4_volt[idx(1, x, y, z) + shift_x].v),
                                f4_volt[idx(0, x, y, z)].v)));
                }
            }
        }
#endif
    }
};

#ifdef __AVX__
//=============================================================================
// BENCHMARK 4: AVX vectorized (8 floats at a time)
//=============================================================================
class Engine_AVX {
public:
    unsigned int numLines[3];
    unsigned int numVectors;
    f8vector* f8_volt;
    f8vector* f8_curr;
    f8vector* f8_vv;
    f8vector* f8_vi;
    f8vector* f8_ii;
    f8vector* f8_iv;

    Engine_AVX(unsigned int nx, unsigned int ny, unsigned int nz) {
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

    ~Engine_AVX() {
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
        for (unsigned int x = 0; x < numLines[0]; ++x) {
            int shift_x = (x > 0) ? (numLines[1] * numVectors) : 0;
            for (unsigned int y = 0; y < numLines[1]; ++y) {
                int shift_y = (y > 0) ? numVectors : 0;

                for (unsigned int z = 1; z < numVectors; ++z) {
                    size_t i0 = idx(0, x, y, z);
                    size_t i1 = idx(1, x, y, z);
                    size_t i2 = idx(2, x, y, z);

                    // x-polarization
                    f8_volt[i0].v = _mm256_mul_ps(f8_volt[i0].v, f8_vv[i0].v);
                    f8_volt[i0].v = _mm256_add_ps(f8_volt[i0].v,
                        _mm256_mul_ps(f8_vi[i0].v,
                            _mm256_sub_ps(
                                _mm256_sub_ps(
                                    _mm256_add_ps(f8_curr[idx(2, x, y, z)].v,
                                                  f8_curr[idx(1, x, y, z - 1)].v),
                                    f8_curr[idx(2, x, y, z) - shift_y].v),
                                f8_curr[idx(1, x, y, z)].v)));

                    // y-polarization
                    f8_volt[i1].v = _mm256_mul_ps(f8_volt[i1].v, f8_vv[i1].v);
                    f8_volt[i1].v = _mm256_add_ps(f8_volt[i1].v,
                        _mm256_mul_ps(f8_vi[i1].v,
                            _mm256_sub_ps(
                                _mm256_sub_ps(
                                    _mm256_add_ps(f8_curr[idx(0, x, y, z)].v,
                                                  f8_curr[idx(2, x, y, z) - shift_x].v),
                                    f8_curr[idx(0, x, y, z - 1)].v),
                                f8_curr[idx(2, x, y, z)].v)));

                    // z-polarization
                    f8_volt[i2].v = _mm256_mul_ps(f8_volt[i2].v, f8_vv[i2].v);
                    f8_volt[i2].v = _mm256_add_ps(f8_volt[i2].v,
                        _mm256_mul_ps(f8_vi[i2].v,
                            _mm256_sub_ps(
                                _mm256_sub_ps(
                                    _mm256_add_ps(f8_curr[idx(1, x, y, z)].v,
                                                  f8_curr[idx(0, x, y, z) - shift_y].v),
                                    f8_curr[idx(1, x, y, z) - shift_x].v),
                                f8_curr[idx(0, x, y, z)].v)));
                }

                // Boundary at z=0 - simplified for benchmark
                {
                    size_t i0 = idx(0, x, y, 0);
                    size_t i1 = idx(1, x, y, 0);
                    size_t i2 = idx(2, x, y, 0);

                    f8_volt[i0].v = _mm256_mul_ps(f8_volt[i0].v, f8_vv[i0].v);
                    f8_volt[i1].v = _mm256_mul_ps(f8_volt[i1].v, f8_vv[i1].v);
                    f8_volt[i2].v = _mm256_mul_ps(f8_volt[i2].v, f8_vv[i2].v);
                }
            }
        }
    }

    void UpdateCurrents() {
        for (unsigned int x = 0; x < numLines[0] - 1; ++x) {
            int shift_x = numLines[1] * numVectors;
            for (unsigned int y = 0; y < numLines[1] - 1; ++y) {
                int shift_y = numVectors;

                for (unsigned int z = 0; z < numVectors - 1; ++z) {
                    size_t i0 = idx(0, x, y, z);
                    size_t i1 = idx(1, x, y, z);
                    size_t i2 = idx(2, x, y, z);

                    // x-polarization
                    f8_curr[i0].v = _mm256_mul_ps(f8_curr[i0].v, f8_ii[i0].v);
                    f8_curr[i0].v = _mm256_add_ps(f8_curr[i0].v,
                        _mm256_mul_ps(f8_iv[i0].v,
                            _mm256_sub_ps(
                                _mm256_sub_ps(
                                    _mm256_add_ps(f8_volt[idx(2, x, y, z)].v,
                                                  f8_volt[idx(1, x, y, z + 1)].v),
                                    f8_volt[idx(2, x, y, z) + shift_y].v),
                                f8_volt[idx(1, x, y, z)].v)));

                    // y-polarization
                    f8_curr[i1].v = _mm256_mul_ps(f8_curr[i1].v, f8_ii[i1].v);
                    f8_curr[i1].v = _mm256_add_ps(f8_curr[i1].v,
                        _mm256_mul_ps(f8_iv[i1].v,
                            _mm256_sub_ps(
                                _mm256_sub_ps(
                                    _mm256_add_ps(f8_volt[idx(0, x, y, z)].v,
                                                  f8_volt[idx(2, x, y, z) + shift_x].v),
                                    f8_volt[idx(0, x, y, z + 1)].v),
                                f8_volt[idx(2, x, y, z)].v)));

                    // z-polarization
                    f8_curr[i2].v = _mm256_mul_ps(f8_curr[i2].v, f8_ii[i2].v);
                    f8_curr[i2].v = _mm256_add_ps(f8_curr[i2].v,
                        _mm256_mul_ps(f8_iv[i2].v,
                            _mm256_sub_ps(
                                _mm256_sub_ps(
                                    _mm256_add_ps(f8_volt[idx(1, x, y, z)].v,
                                                  f8_volt[idx(0, x, y, z) + shift_y].v),
                                    f8_volt[idx(1, x, y, z) + shift_x].v),
                                f8_volt[idx(0, x, y, z)].v)));
                }

                // Boundary at z = numVectors - 1 - simplified
                {
                    size_t z = numVectors - 1;
                    size_t i0 = idx(0, x, y, z);
                    size_t i1 = idx(1, x, y, z);
                    size_t i2 = idx(2, x, y, z);

                    f8_curr[i0].v = _mm256_mul_ps(f8_curr[i0].v, f8_ii[i0].v);
                    f8_curr[i1].v = _mm256_mul_ps(f8_curr[i1].v, f8_ii[i1].v);
                    f8_curr[i2].v = _mm256_mul_ps(f8_curr[i2].v, f8_ii[i2].v);
                }
            }
        }
    }
};
#endif

//=============================================================================
// BENCHMARK 5: Loop tiling for better cache utilization
//=============================================================================
class Engine_Tiled {
public:
    unsigned int numLines[3];
    unsigned int numVectors;
    f4vector* f4_volt;
    f4vector* f4_curr;
    f4vector* f4_vv;
    f4vector* f4_vi;
    f4vector* f4_ii;
    f4vector* f4_iv;

    static constexpr unsigned int TILE_X = 8;
    static constexpr unsigned int TILE_Y = 8;

    Engine_Tiled(unsigned int nx, unsigned int ny, unsigned int nz) {
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

    ~Engine_Tiled() {
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
        // Tiled loop for better cache utilization
        for (unsigned int tx = 0; tx < numLines[0]; tx += TILE_X) {
            unsigned int x_end = min(tx + TILE_X, numLines[0]);
            for (unsigned int ty = 0; ty < numLines[1]; ty += TILE_Y) {
                unsigned int y_end = min(ty + TILE_Y, numLines[1]);

                for (unsigned int x = tx; x < x_end; ++x) {
                    int shift_x = (x > 0) ? (numLines[1] * numVectors) : 0;
                    for (unsigned int y = ty; y < y_end; ++y) {
                        int shift_y = (y > 0) ? numVectors : 0;

                        for (unsigned int z = 1; z < numVectors; ++z) {
                            size_t i0 = idx(0, x, y, z);
                            size_t i1 = idx(1, x, y, z);
                            size_t i2 = idx(2, x, y, z);

                            f4_volt[i0].v = _mm_mul_ps(f4_volt[i0].v, f4_vv[i0].v);
                            f4_volt[i0].v = _mm_add_ps(f4_volt[i0].v,
                                _mm_mul_ps(f4_vi[i0].v,
                                    _mm_sub_ps(
                                        _mm_sub_ps(
                                            _mm_add_ps(f4_curr[idx(2, x, y, z)].v,
                                                       f4_curr[idx(1, x, y, z - 1)].v),
                                            f4_curr[idx(2, x, y, z) - shift_y].v),
                                        f4_curr[idx(1, x, y, z)].v)));

                            f4_volt[i1].v = _mm_mul_ps(f4_volt[i1].v, f4_vv[i1].v);
                            f4_volt[i1].v = _mm_add_ps(f4_volt[i1].v,
                                _mm_mul_ps(f4_vi[i1].v,
                                    _mm_sub_ps(
                                        _mm_sub_ps(
                                            _mm_add_ps(f4_curr[idx(0, x, y, z)].v,
                                                       f4_curr[idx(2, x, y, z) - shift_x].v),
                                            f4_curr[idx(0, x, y, z - 1)].v),
                                        f4_curr[idx(2, x, y, z)].v)));

                            f4_volt[i2].v = _mm_mul_ps(f4_volt[i2].v, f4_vv[i2].v);
                            f4_volt[i2].v = _mm_add_ps(f4_volt[i2].v,
                                _mm_mul_ps(f4_vi[i2].v,
                                    _mm_sub_ps(
                                        _mm_sub_ps(
                                            _mm_add_ps(f4_curr[idx(1, x, y, z)].v,
                                                       f4_curr[idx(0, x, y, z) - shift_y].v),
                                            f4_curr[idx(1, x, y, z) - shift_x].v),
                                        f4_curr[idx(0, x, y, z)].v)));
                        }

                        // z=0 boundary
                        {
                            size_t i0 = idx(0, x, y, 0);
                            size_t i1 = idx(1, x, y, 0);
                            size_t i2 = idx(2, x, y, 0);

                            f4vector temp;
                            temp.v = (__m128)_mm_slli_si128((__m128i)f4_curr[idx(1, x, y, numVectors - 1)].v, 4);
                            f4_volt[i0].v = _mm_mul_ps(f4_volt[i0].v, f4_vv[i0].v);
                            f4_volt[i0].v = _mm_add_ps(f4_volt[i0].v,
                                _mm_mul_ps(f4_vi[i0].v,
                                    _mm_sub_ps(
                                        _mm_sub_ps(
                                            _mm_add_ps(f4_curr[idx(2, x, y, 0)].v, temp.v),
                                            f4_curr[idx(2, x, y, 0) - shift_y].v),
                                        f4_curr[idx(1, x, y, 0)].v)));

                            temp.v = (__m128)_mm_slli_si128((__m128i)f4_curr[idx(0, x, y, numVectors - 1)].v, 4);
                            f4_volt[i1].v = _mm_mul_ps(f4_volt[i1].v, f4_vv[i1].v);
                            f4_volt[i1].v = _mm_add_ps(f4_volt[i1].v,
                                _mm_mul_ps(f4_vi[i1].v,
                                    _mm_sub_ps(
                                        _mm_sub_ps(
                                            _mm_add_ps(f4_curr[idx(0, x, y, 0)].v,
                                                       f4_curr[idx(2, x, y, 0) - shift_x].v),
                                            temp.v),
                                        f4_curr[idx(2, x, y, 0)].v)));

                            f4_volt[i2].v = _mm_mul_ps(f4_volt[i2].v, f4_vv[i2].v);
                            f4_volt[i2].v = _mm_add_ps(f4_volt[i2].v,
                                _mm_mul_ps(f4_vi[i2].v,
                                    _mm_sub_ps(
                                        _mm_sub_ps(
                                            _mm_add_ps(f4_curr[idx(1, x, y, 0)].v,
                                                       f4_curr[idx(0, x, y, 0) - shift_y].v),
                                            f4_curr[idx(1, x, y, 0) - shift_x].v),
                                        f4_curr[idx(0, x, y, 0)].v)));
                        }
                    }
                }
            }
        }
#endif
    }

    void UpdateCurrents() {
#ifdef __SSE2__
        for (unsigned int tx = 0; tx < numLines[0] - 1; tx += TILE_X) {
            unsigned int x_end = min(tx + TILE_X, numLines[0] - 1);
            for (unsigned int ty = 0; ty < numLines[1] - 1; ty += TILE_Y) {
                unsigned int y_end = min(ty + TILE_Y, numLines[1] - 1);

                for (unsigned int x = tx; x < x_end; ++x) {
                    int shift_x = numLines[1] * numVectors;
                    for (unsigned int y = ty; y < y_end; ++y) {
                        int shift_y = numVectors;

                        for (unsigned int z = 0; z < numVectors - 1; ++z) {
                            size_t i0 = idx(0, x, y, z);
                            size_t i1 = idx(1, x, y, z);
                            size_t i2 = idx(2, x, y, z);

                            f4_curr[i0].v = _mm_mul_ps(f4_curr[i0].v, f4_ii[i0].v);
                            f4_curr[i0].v = _mm_add_ps(f4_curr[i0].v,
                                _mm_mul_ps(f4_iv[i0].v,
                                    _mm_sub_ps(
                                        _mm_sub_ps(
                                            _mm_add_ps(f4_volt[idx(2, x, y, z)].v,
                                                       f4_volt[idx(1, x, y, z + 1)].v),
                                            f4_volt[idx(2, x, y, z) + shift_y].v),
                                        f4_volt[idx(1, x, y, z)].v)));

                            f4_curr[i1].v = _mm_mul_ps(f4_curr[i1].v, f4_ii[i1].v);
                            f4_curr[i1].v = _mm_add_ps(f4_curr[i1].v,
                                _mm_mul_ps(f4_iv[i1].v,
                                    _mm_sub_ps(
                                        _mm_sub_ps(
                                            _mm_add_ps(f4_volt[idx(0, x, y, z)].v,
                                                       f4_volt[idx(2, x, y, z) + shift_x].v),
                                            f4_volt[idx(0, x, y, z + 1)].v),
                                        f4_volt[idx(2, x, y, z)].v)));

                            f4_curr[i2].v = _mm_mul_ps(f4_curr[i2].v, f4_ii[i2].v);
                            f4_curr[i2].v = _mm_add_ps(f4_curr[i2].v,
                                _mm_mul_ps(f4_iv[i2].v,
                                    _mm_sub_ps(
                                        _mm_sub_ps(
                                            _mm_add_ps(f4_volt[idx(1, x, y, z)].v,
                                                       f4_volt[idx(0, x, y, z) + shift_y].v),
                                            f4_volt[idx(1, x, y, z) + shift_x].v),
                                        f4_volt[idx(0, x, y, z)].v)));
                        }

                        // z = numVectors - 1 boundary
                        {
                            unsigned int z = numVectors - 1;
                            size_t i0 = idx(0, x, y, z);
                            size_t i1 = idx(1, x, y, z);
                            size_t i2 = idx(2, x, y, z);

                            f4vector temp;
                            temp.v = (__m128)_mm_srli_si128((__m128i)f4_volt[idx(1, x, y, 0)].v, 4);
                            f4_curr[i0].v = _mm_mul_ps(f4_curr[i0].v, f4_ii[i0].v);
                            f4_curr[i0].v = _mm_add_ps(f4_curr[i0].v,
                                _mm_mul_ps(f4_iv[i0].v,
                                    _mm_sub_ps(
                                        _mm_sub_ps(
                                            _mm_add_ps(f4_volt[idx(2, x, y, z)].v, temp.v),
                                            f4_volt[idx(2, x, y, z) + shift_y].v),
                                        f4_volt[idx(1, x, y, z)].v)));

                            temp.v = (__m128)_mm_srli_si128((__m128i)f4_volt[idx(0, x, y, 0)].v, 4);
                            f4_curr[i1].v = _mm_mul_ps(f4_curr[i1].v, f4_ii[i1].v);
                            f4_curr[i1].v = _mm_add_ps(f4_curr[i1].v,
                                _mm_mul_ps(f4_iv[i1].v,
                                    _mm_sub_ps(
                                        _mm_sub_ps(
                                            _mm_add_ps(f4_volt[idx(0, x, y, z)].v,
                                                       f4_volt[idx(2, x, y, z) + shift_x].v),
                                            temp.v),
                                        f4_volt[idx(2, x, y, z)].v)));

                            f4_curr[i2].v = _mm_mul_ps(f4_curr[i2].v, f4_ii[i2].v);
                            f4_curr[i2].v = _mm_add_ps(f4_curr[i2].v,
                                _mm_mul_ps(f4_iv[i2].v,
                                    _mm_sub_ps(
                                        _mm_sub_ps(
                                            _mm_add_ps(f4_volt[idx(1, x, y, z)].v,
                                                       f4_volt[idx(0, x, y, z) + shift_y].v),
                                            f4_volt[idx(1, x, y, z) + shift_x].v),
                                        f4_volt[idx(0, x, y, z)].v)));
                        }
                    }
                }
            }
        }
#endif
    }
};

//=============================================================================
// BENCHMARK 6: Loop fusion (update voltage and current in single pass)
//=============================================================================
class Engine_Fused {
public:
    unsigned int numLines[3];
    unsigned int numVectors;
    f4vector* f4_volt;
    f4vector* f4_curr;
    f4vector* f4_vv;
    f4vector* f4_vi;
    f4vector* f4_ii;
    f4vector* f4_iv;

    Engine_Fused(unsigned int nx, unsigned int ny, unsigned int nz) {
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

    ~Engine_Fused() {
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

    // Note: True loop fusion would require careful dependency analysis
    // This implementation just demonstrates the concept
    void UpdateVoltages() {
#ifdef __SSE2__
        for (unsigned int x = 0; x < numLines[0]; ++x) {
            int shift_x = (x > 0) ? (numLines[1] * numVectors) : 0;
            for (unsigned int y = 0; y < numLines[1]; ++y) {
                int shift_y = (y > 0) ? numVectors : 0;

                for (unsigned int z = 1; z < numVectors; ++z) {
                    size_t i0 = idx(0, x, y, z);
                    size_t i1 = idx(1, x, y, z);
                    size_t i2 = idx(2, x, y, z);

                    f4_volt[i0].v = _mm_mul_ps(f4_volt[i0].v, f4_vv[i0].v);
                    f4_volt[i0].v = _mm_add_ps(f4_volt[i0].v,
                        _mm_mul_ps(f4_vi[i0].v,
                            _mm_sub_ps(
                                _mm_sub_ps(
                                    _mm_add_ps(f4_curr[idx(2, x, y, z)].v,
                                               f4_curr[idx(1, x, y, z - 1)].v),
                                    f4_curr[idx(2, x, y, z) - shift_y].v),
                                f4_curr[idx(1, x, y, z)].v)));

                    f4_volt[i1].v = _mm_mul_ps(f4_volt[i1].v, f4_vv[i1].v);
                    f4_volt[i1].v = _mm_add_ps(f4_volt[i1].v,
                        _mm_mul_ps(f4_vi[i1].v,
                            _mm_sub_ps(
                                _mm_sub_ps(
                                    _mm_add_ps(f4_curr[idx(0, x, y, z)].v,
                                               f4_curr[idx(2, x, y, z) - shift_x].v),
                                    f4_curr[idx(0, x, y, z - 1)].v),
                                f4_curr[idx(2, x, y, z)].v)));

                    f4_volt[i2].v = _mm_mul_ps(f4_volt[i2].v, f4_vv[i2].v);
                    f4_volt[i2].v = _mm_add_ps(f4_volt[i2].v,
                        _mm_mul_ps(f4_vi[i2].v,
                            _mm_sub_ps(
                                _mm_sub_ps(
                                    _mm_add_ps(f4_curr[idx(1, x, y, z)].v,
                                               f4_curr[idx(0, x, y, z) - shift_y].v),
                                    f4_curr[idx(1, x, y, z) - shift_x].v),
                                f4_curr[idx(0, x, y, z)].v)));
                }

                // z=0 boundary handling simplified for clarity
                {
                    size_t i0 = idx(0, x, y, 0);
                    size_t i1 = idx(1, x, y, 0);
                    size_t i2 = idx(2, x, y, 0);
                    f4_volt[i0].v = _mm_mul_ps(f4_volt[i0].v, f4_vv[i0].v);
                    f4_volt[i1].v = _mm_mul_ps(f4_volt[i1].v, f4_vv[i1].v);
                    f4_volt[i2].v = _mm_mul_ps(f4_volt[i2].v, f4_vv[i2].v);
                }
            }
        }
#endif
    }

    void UpdateCurrents() {
#ifdef __SSE2__
        for (unsigned int x = 0; x < numLines[0] - 1; ++x) {
            int shift_x = numLines[1] * numVectors;
            for (unsigned int y = 0; y < numLines[1] - 1; ++y) {
                int shift_y = numVectors;

                for (unsigned int z = 0; z < numVectors - 1; ++z) {
                    size_t i0 = idx(0, x, y, z);
                    size_t i1 = idx(1, x, y, z);
                    size_t i2 = idx(2, x, y, z);

                    f4_curr[i0].v = _mm_mul_ps(f4_curr[i0].v, f4_ii[i0].v);
                    f4_curr[i0].v = _mm_add_ps(f4_curr[i0].v,
                        _mm_mul_ps(f4_iv[i0].v,
                            _mm_sub_ps(
                                _mm_sub_ps(
                                    _mm_add_ps(f4_volt[idx(2, x, y, z)].v,
                                               f4_volt[idx(1, x, y, z + 1)].v),
                                    f4_volt[idx(2, x, y, z) + shift_y].v),
                                f4_volt[idx(1, x, y, z)].v)));

                    f4_curr[i1].v = _mm_mul_ps(f4_curr[i1].v, f4_ii[i1].v);
                    f4_curr[i1].v = _mm_add_ps(f4_curr[i1].v,
                        _mm_mul_ps(f4_iv[i1].v,
                            _mm_sub_ps(
                                _mm_sub_ps(
                                    _mm_add_ps(f4_volt[idx(0, x, y, z)].v,
                                               f4_volt[idx(2, x, y, z) + shift_x].v),
                                    f4_volt[idx(0, x, y, z + 1)].v),
                                f4_volt[idx(2, x, y, z)].v)));

                    f4_curr[i2].v = _mm_mul_ps(f4_curr[i2].v, f4_ii[i2].v);
                    f4_curr[i2].v = _mm_add_ps(f4_curr[i2].v,
                        _mm_mul_ps(f4_iv[i2].v,
                            _mm_sub_ps(
                                _mm_sub_ps(
                                    _mm_add_ps(f4_volt[idx(1, x, y, z)].v,
                                               f4_volt[idx(0, x, y, z) + shift_y].v),
                                    f4_volt[idx(1, x, y, z) + shift_x].v),
                                f4_volt[idx(0, x, y, z)].v)));
                }

                // z = numVectors - 1 boundary simplified
                {
                    unsigned int z = numVectors - 1;
                    size_t i0 = idx(0, x, y, z);
                    size_t i1 = idx(1, x, y, z);
                    size_t i2 = idx(2, x, y, z);
                    f4_curr[i0].v = _mm_mul_ps(f4_curr[i0].v, f4_ii[i0].v);
                    f4_curr[i1].v = _mm_mul_ps(f4_curr[i1].v, f4_ii[i1].v);
                    f4_curr[i2].v = _mm_mul_ps(f4_curr[i2].v, f4_ii[i2].v);
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

    // Warmup
    for (unsigned int i = 0; i < warmup; ++i) {
        engine.UpdateVoltages();
        engine.UpdateCurrents();
    }

    // Benchmark
    timer.start();
    for (unsigned int i = 0; i < iterations; ++i) {
        engine.UpdateVoltages();
        engine.UpdateCurrents();
    }
    timer.stop();

    double total_ms = timer.elapsed_ms();
    double per_iter_us = timer.elapsed_us() / iterations;

    // Calculate MCPS (million cells per second)
    double cells = (double)nx * ny * nz;
    double mcps = (cells * iterations) / (timer.elapsed_us());

    cout << setw(25) << left << name
         << setw(12) << right << fixed << setprecision(2) << total_ms << " ms"
         << setw(12) << right << fixed << setprecision(1) << per_iter_us << " us/iter"
         << setw(12) << right << fixed << setprecision(2) << mcps << " MCPS"
         << endl;

    return per_iter_us;
}

void print_system_info() {
    cout << "=== System Information ===" << endl;

#ifdef __SSE2__
    cout << "SSE2: enabled" << endl;
#else
    cout << "SSE2: disabled" << endl;
#endif

#ifdef __AVX__
    cout << "AVX: enabled" << endl;
#else
    cout << "AVX: disabled" << endl;
#endif

#ifdef __AVX2__
    cout << "AVX2: enabled" << endl;
#else
    cout << "AVX2: disabled" << endl;
#endif

#ifdef __AVX512F__
    cout << "AVX512: enabled" << endl;
#else
    cout << "AVX512: disabled" << endl;
#endif

    cout << endl;
}

int main(int argc, char* argv[]) {
    // Disable denormals for performance
#ifdef __SSE__
    _mm_setcsr(_mm_getcsr() | 0x8040);  // FTZ + DAZ
#endif

    unsigned int nx = DEFAULT_NX;
    unsigned int ny = DEFAULT_NY;
    unsigned int nz = DEFAULT_NZ;
    unsigned int iterations = NUM_ITERATIONS;
    unsigned int warmup = NUM_WARMUP;

    // Parse command line arguments
    if (argc >= 4) {
        nx = atoi(argv[1]);
        ny = atoi(argv[2]);
        nz = atoi(argv[3]);
    }
    if (argc >= 5) {
        iterations = atoi(argv[4]);
    }

    print_system_info();

    cout << "=== FDTD Benchmark ===" << endl;
    cout << "Grid size: " << nx << " x " << ny << " x " << nz << endl;
    cout << "Total cells: " << (nx * ny * nz) << endl;
    cout << "Iterations: " << iterations << " (warmup: " << warmup << ")" << endl;
    cout << "Memory per field array: " << (3ULL * nx * ny * nz * sizeof(float) / 1024 / 1024) << " MB" << endl;
    cout << endl;

    cout << setw(25) << left << "Engine"
         << setw(15) << right << "Total"
         << setw(15) << right << "Per Iter"
         << setw(15) << right << "Throughput"
         << endl;
    cout << string(70, '-') << endl;

    double basic_time = benchmark_engine<Engine_Basic>("Basic (scalar)", nx, ny, nz, iterations, warmup);
    double ijkn_time = benchmark_engine<Engine_IJKN>("IJKN layout", nx, ny, nz, iterations, warmup);

#ifdef __SSE2__
    double sse_time = benchmark_engine<Engine_SSE>("SSE vectorized", nx, ny, nz, iterations, warmup);
    double tiled_time = benchmark_engine<Engine_Tiled>("SSE + tiling", nx, ny, nz, iterations, warmup);
    double fused_time = benchmark_engine<Engine_Fused>("SSE simplified", nx, ny, nz, iterations, warmup);
#endif

#ifdef __AVX__
    double avx_time = benchmark_engine<Engine_AVX>("AVX vectorized", nx, ny, nz, iterations, warmup);
#endif

    cout << string(70, '-') << endl;
    cout << endl;

    // Print speedup summary
    cout << "=== Speedup vs Basic ===" << endl;
    cout << "IJKN layout:    " << fixed << setprecision(2) << basic_time / ijkn_time << "x" << endl;
#ifdef __SSE2__
    cout << "SSE:            " << fixed << setprecision(2) << basic_time / sse_time << "x" << endl;
    cout << "SSE + tiling:   " << fixed << setprecision(2) << basic_time / tiled_time << "x" << endl;
    cout << "SSE simplified: " << fixed << setprecision(2) << basic_time / fused_time << "x" << endl;
#endif
#ifdef __AVX__
    cout << "AVX:            " << fixed << setprecision(2) << basic_time / avx_time << "x" << endl;
#endif

    return 0;
}
