/*
 * FDTD Engine Benchmark
 * Measures performance of SSE and basic FDTD kernels
 */

#include <iostream>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdlib>

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

// Grid dimensions
constexpr unsigned int NX = 100;  // X dimension
constexpr unsigned int NY = 100;  // Y dimension
constexpr unsigned int NZ = 100;  // Z dimension
constexpr unsigned int NUM_VECTORS = (NZ + 3) / 4;  // Vectorized Z dimension
constexpr unsigned int TIMESTEPS = 1000;

// 4D array for SSE: [polarization][x][y][z_vector]
f4vector**** volt_sse = nullptr;
f4vector**** curr_sse = nullptr;
f4vector**** vv_sse = nullptr;
f4vector**** vi_sse = nullptr;
f4vector**** ii_sse = nullptr;
f4vector**** iv_sse = nullptr;

// 4D array for basic: [polarization][x][y][z]
float**** volt_basic = nullptr;
float**** curr_basic = nullptr;
float**** vv_basic = nullptr;
float**** vi_basic = nullptr;
float**** ii_basic = nullptr;
float**** iv_basic = nullptr;

// Allocate 4D SSE array
f4vector**** allocate_sse_array(unsigned int nx, unsigned int ny, unsigned int nz_vec) {
    f4vector**** arr = new f4vector***[3];
    for (int n = 0; n < 3; n++) {
        arr[n] = new f4vector**[nx];
        for (unsigned int x = 0; x < nx; x++) {
            arr[n][x] = new f4vector*[ny];
            for (unsigned int y = 0; y < ny; y++) {
                arr[n][x][y] = new f4vector[nz_vec];
                for (unsigned int z = 0; z < nz_vec; z++) {
                    for (int i = 0; i < 4; i++) {
                        arr[n][x][y][z].f[i] = 0.0f;
                    }
                }
            }
        }
    }
    return arr;
}

// Allocate 4D basic array
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

// Free 4D SSE array
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

// Free 4D basic array
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

// Initialize arrays with test values
void init_sse_arrays() {
    volt_sse = allocate_sse_array(NX, NY, NUM_VECTORS);
    curr_sse = allocate_sse_array(NX, NY, NUM_VECTORS);
    vv_sse = allocate_sse_array(NX, NY, NUM_VECTORS);
    vi_sse = allocate_sse_array(NX, NY, NUM_VECTORS);
    ii_sse = allocate_sse_array(NX, NY, NUM_VECTORS);
    iv_sse = allocate_sse_array(NX, NY, NUM_VECTORS);

    // Initialize coefficients
    for (int n = 0; n < 3; n++) {
        for (unsigned int x = 0; x < NX; x++) {
            for (unsigned int y = 0; y < NY; y++) {
                for (unsigned int z = 0; z < NUM_VECTORS; z++) {
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

    // Initialize a source point
    unsigned int cx = NX/2, cy = NY/2, cz = NUM_VECTORS/2;
    volt_sse[0][cx][cy][cz].f[0] = 1.0f;
}

void init_basic_arrays() {
    volt_basic = allocate_basic_array(NX, NY, NZ);
    curr_basic = allocate_basic_array(NX, NY, NZ);
    vv_basic = allocate_basic_array(NX, NY, NZ);
    vi_basic = allocate_basic_array(NX, NY, NZ);
    ii_basic = allocate_basic_array(NX, NY, NZ);
    iv_basic = allocate_basic_array(NX, NY, NZ);

    // Initialize coefficients
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

    // Initialize a source point
    unsigned int cx = NX/2, cy = NY/2, cz = NZ/2;
    volt_basic[0][cx][cy][cz] = 1.0f;
}

// SSE UpdateVoltages (simplified from openEMS)
void UpdateVoltages_SSE() {
    for (unsigned int x = 1; x < NX-1; x++) {
        for (unsigned int y = 1; y < NY-1; y++) {
            for (unsigned int z = 1; z < NUM_VECTORS; z++) {
                // x-polarization
                volt_sse[0][x][y][z].v *= vv_sse[0][x][y][z].v;
                volt_sse[0][x][y][z].v += vi_sse[0][x][y][z].v * (
                    curr_sse[2][x][y][z].v - curr_sse[2][x][y-1][z].v -
                    curr_sse[1][x][y][z].v + curr_sse[1][x][y][z-1].v
                );

                // y-polarization
                volt_sse[1][x][y][z].v *= vv_sse[1][x][y][z].v;
                volt_sse[1][x][y][z].v += vi_sse[1][x][y][z].v * (
                    curr_sse[0][x][y][z].v - curr_sse[0][x][y][z-1].v -
                    curr_sse[2][x][y][z].v + curr_sse[2][x-1][y][z].v
                );

                // z-polarization
                volt_sse[2][x][y][z].v *= vv_sse[2][x][y][z].v;
                volt_sse[2][x][y][z].v += vi_sse[2][x][y][z].v * (
                    curr_sse[1][x][y][z].v - curr_sse[1][x-1][y][z].v -
                    curr_sse[0][x][y][z].v + curr_sse[0][x][y-1][z].v
                );
            }
        }
    }
}

// SSE UpdateCurrents (simplified from openEMS)
void UpdateCurrents_SSE() {
    for (unsigned int x = 1; x < NX-1; x++) {
        for (unsigned int y = 1; y < NY-2; y++) {
            for (unsigned int z = 0; z < NUM_VECTORS-1; z++) {
                // x-polarization
                curr_sse[0][x][y][z].v *= ii_sse[0][x][y][z].v;
                curr_sse[0][x][y][z].v += iv_sse[0][x][y][z].v * (
                    volt_sse[2][x][y][z].v - volt_sse[2][x][y+1][z].v -
                    volt_sse[1][x][y][z].v + volt_sse[1][x][y][z+1].v
                );

                // y-polarization
                curr_sse[1][x][y][z].v *= ii_sse[1][x][y][z].v;
                curr_sse[1][x][y][z].v += iv_sse[1][x][y][z].v * (
                    volt_sse[0][x][y][z].v - volt_sse[0][x][y][z+1].v -
                    volt_sse[2][x][y][z].v + volt_sse[2][x+1][y][z].v
                );

                // z-polarization
                curr_sse[2][x][y][z].v *= ii_sse[2][x][y][z].v;
                curr_sse[2][x][y][z].v += iv_sse[2][x][y][z].v * (
                    volt_sse[1][x][y][z].v - volt_sse[1][x+1][y][z].v -
                    volt_sse[0][x][y][z].v + volt_sse[0][x][y+1][z].v
                );
            }
        }
    }
}

// Basic UpdateVoltages (scalar)
void UpdateVoltages_Basic() {
    for (unsigned int x = 1; x < NX-1; x++) {
        for (unsigned int y = 1; y < NY-1; y++) {
            for (unsigned int z = 1; z < NZ-1; z++) {
                // x-polarization
                volt_basic[0][x][y][z] *= vv_basic[0][x][y][z];
                volt_basic[0][x][y][z] += vi_basic[0][x][y][z] * (
                    curr_basic[2][x][y][z] - curr_basic[2][x][y-1][z] -
                    curr_basic[1][x][y][z] + curr_basic[1][x][y][z-1]
                );

                // y-polarization
                volt_basic[1][x][y][z] *= vv_basic[1][x][y][z];
                volt_basic[1][x][y][z] += vi_basic[1][x][y][z] * (
                    curr_basic[0][x][y][z] - curr_basic[0][x][y][z-1] -
                    curr_basic[2][x][y][z] + curr_basic[2][x-1][y][z]
                );

                // z-polarization
                volt_basic[2][x][y][z] *= vv_basic[2][x][y][z];
                volt_basic[2][x][y][z] += vi_basic[2][x][y][z] * (
                    curr_basic[1][x][y][z] - curr_basic[1][x-1][y][z] -
                    curr_basic[0][x][y][z] + curr_basic[0][x][y-1][z]
                );
            }
        }
    }
}

// Basic UpdateCurrents (scalar)
void UpdateCurrents_Basic() {
    for (unsigned int x = 1; x < NX-1; x++) {
        for (unsigned int y = 1; y < NY-2; y++) {
            for (unsigned int z = 0; z < NZ-1; z++) {
                // x-polarization
                curr_basic[0][x][y][z] *= ii_basic[0][x][y][z];
                curr_basic[0][x][y][z] += iv_basic[0][x][y][z] * (
                    volt_basic[2][x][y][z] - volt_basic[2][x][y+1][z] -
                    volt_basic[1][x][y][z] + volt_basic[1][x][y][z+1]
                );

                // y-polarization
                curr_basic[1][x][y][z] *= ii_basic[1][x][y][z];
                curr_basic[1][x][y][z] += iv_basic[1][x][y][z] * (
                    volt_basic[0][x][y][z] - volt_basic[0][x][y][z+1] -
                    volt_basic[2][x][y][z] + volt_basic[2][x+1][y][z]
                );

                // z-polarization
                curr_basic[2][x][y][z] *= ii_basic[2][x][y][z];
                curr_basic[2][x][y][z] += iv_basic[2][x][y][z] * (
                    volt_basic[1][x][y][z] - volt_basic[1][x+1][y][z] -
                    volt_basic[0][x][y][z] + volt_basic[0][x][y+1][z]
                );
            }
        }
    }
}

int main(int argc, char* argv[]) {
    std::cout << "====== FDTD Engine Benchmark ======" << std::endl;
    std::cout << "Grid size: " << NX << " x " << NY << " x " << NZ << std::endl;
    std::cout << "Timesteps: " << TIMESTEPS << std::endl;
    std::cout << "Total cells: " << (unsigned long long)NX * NY * NZ << std::endl;
    std::cout << std::endl;

    // Benchmark SSE Engine
    std::cout << "Initializing SSE arrays..." << std::endl;
    init_sse_arrays();

    std::cout << "Running SSE benchmark..." << std::endl;
    auto start_sse = std::chrono::high_resolution_clock::now();

    for (unsigned int ts = 0; ts < TIMESTEPS; ts++) {
        UpdateVoltages_SSE();
        UpdateCurrents_SSE();
    }

    auto end_sse = std::chrono::high_resolution_clock::now();
    auto duration_sse = std::chrono::duration_cast<std::chrono::milliseconds>(end_sse - start_sse).count();

    double cells_per_sec_sse = ((double)NX * NY * NZ * TIMESTEPS) / (duration_sse / 1000.0);
    double mc_per_sec_sse = cells_per_sec_sse / 1e6;

    std::cout << "SSE Engine:" << std::endl;
    std::cout << "  Time: " << duration_sse << " ms" << std::endl;
    std::cout << "  Speed: " << mc_per_sec_sse << " MC/s" << std::endl;
    std::cout << std::endl;

    // Clean up SSE arrays
    free_sse_array(volt_sse, NX, NY);
    free_sse_array(curr_sse, NX, NY);
    free_sse_array(vv_sse, NX, NY);
    free_sse_array(vi_sse, NX, NY);
    free_sse_array(ii_sse, NX, NY);
    free_sse_array(iv_sse, NX, NY);

    // Benchmark Basic Engine
    std::cout << "Initializing Basic arrays..." << std::endl;
    init_basic_arrays();

    std::cout << "Running Basic benchmark..." << std::endl;
    auto start_basic = std::chrono::high_resolution_clock::now();

    for (unsigned int ts = 0; ts < TIMESTEPS; ts++) {
        UpdateVoltages_Basic();
        UpdateCurrents_Basic();
    }

    auto end_basic = std::chrono::high_resolution_clock::now();
    auto duration_basic = std::chrono::duration_cast<std::chrono::milliseconds>(end_basic - start_basic).count();

    double cells_per_sec_basic = ((double)NX * NY * NZ * TIMESTEPS) / (duration_basic / 1000.0);
    double mc_per_sec_basic = cells_per_sec_basic / 1e6;

    std::cout << "Basic Engine:" << std::endl;
    std::cout << "  Time: " << duration_basic << " ms" << std::endl;
    std::cout << "  Speed: " << mc_per_sec_basic << " MC/s" << std::endl;
    std::cout << std::endl;

    // Clean up Basic arrays
    free_basic_array(volt_basic, NX, NY);
    free_basic_array(curr_basic, NX, NY);
    free_basic_array(vv_basic, NX, NY);
    free_basic_array(vi_basic, NX, NY);
    free_basic_array(ii_basic, NX, NY);
    free_basic_array(iv_basic, NX, NY);

    // Summary
    std::cout << "====== Summary ======" << std::endl;
    std::cout << "SSE speedup over Basic: " << mc_per_sec_sse / mc_per_sec_basic << "x" << std::endl;

    return 0;
}
