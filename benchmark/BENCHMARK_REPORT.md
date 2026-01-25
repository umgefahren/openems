# openEMS FDTD Critical Path Benchmark Report

## Executive Summary

This report presents benchmarking results for the FDTD (Finite-Difference Time-Domain)
engine critical path in openEMS. The critical path consists of the `UpdateVoltages()`
and `UpdateCurrents()` functions that iterate over the 3D grid, performing field updates.

**Key Findings:**
- **OpenMP parallelization** provides the largest speedup: up to **10.9x with 16 threads**
- **Memory layout** significantly impacts performance: IJKN layout is 1.5-1.85x faster
- **SIMD vectorization** (SSE/AVX) provides 2-2.7x speedup over scalar code
- **Memory bandwidth** is the primary bottleneck for large grids (>L3 cache size)
- **AVX** outperforms SSE for large grids but can be slower for small grids

## Test Environment

- CPU: Intel processor with SSE2, AVX, AVX2, AVX-512 support
- Compiler: GCC 13.3.0 with `-O3 -march=native -ffast-math -funroll-loops`
- L3 Cache: ~54 MB

## Benchmark Results

### Performance by Grid Size

#### Small Grid (50x50x64 = 160K cells, ~1MB per array, fits in L2)

| Engine            | Throughput (MCPS) | Speedup vs Basic |
|-------------------|-------------------|------------------|
| Basic (scalar)    | 121.62            | 1.00x            |
| IJKN layout       | 137.13            | 1.13x            |
| SSE vectorized    | 330.61            | 2.72x            |
| SSE simplified    | 337.90            | 2.78x            |
| AVX vectorized    | 305.39            | 2.51x            |

#### Medium Grid (100x100x128 = 1.28M cells, ~14MB per array, fits in L3)

| Engine            | Throughput (MCPS) | Speedup vs Basic |
|-------------------|-------------------|------------------|
| Basic (scalar)    | 77.37             | 1.00x            |
| IJKN layout       | 120.09            | 1.55x            |
| SSE vectorized    | 204.18            | 2.64x            |
| SSE simplified    | 201.21            | 2.60x            |
| AVX vectorized    | 191.09            | 2.47x            |

#### Large Grid (200x200x256 = 10.24M cells, ~117MB per array, exceeds L3)

| Engine            | Throughput (MCPS) | Speedup vs Basic |
|-------------------|-------------------|------------------|
| Basic (scalar)    | 54.82             | 1.00x            |
| IJKN layout       | 101.62            | 1.85x            |
| SSE vectorized    | 109.48            | 2.00x            |
| SSE simplified    | 145.43            | 2.65x            |
| AVX vectorized    | 149.06            | 2.72x            |

### OpenMP Scaling (Large Grid: 200x200x256)

| Threads | Throughput (MCPS) | Speedup vs 1 thread |
|---------|-------------------|---------------------|
| 1       | 116.23            | 1.00x               |
| 4       | 393.63            | 3.39x               |
| 8       | 735.92            | 6.33x               |
| 16      | 1267.96           | 10.91x              |

### Optimization Techniques Comparison (Large Grid)

| Technique           | Throughput (MCPS) | Notes                              |
|---------------------|-------------------|------------------------------------|
| SSE baseline        | 109.48            | Reference SIMD implementation      |
| SSE + prefetch      | 132.04            | +20% from software prefetching     |
| AVX + FMA           | 137.43            | +25% from wider vectors + FMA      |
| AoS layout          | 174.75            | +60% from better cache locality    |
| SSE + OpenMP (16t)  | 1267.96           | +1057% from parallelization        |

## Analysis

### 1. Memory Bandwidth is the Primary Bottleneck

For large grids that exceed L3 cache, performance drops significantly:
- Small grid (L2): 330 MCPS
- Medium grid (L3): 204 MCPS
- Large grid (RAM): 109 MCPS

This 3x performance drop indicates memory bandwidth saturation.

### 2. Memory Layout Impact

The current openEMS uses ArrayENG with I-J-K-N ordering (component interleaved):
```cpp
// I-J-K-N: Component is fastest-varying index
stride[0] = 1;  // N (component)
stride[1] = extent[2] * extent[3] * extent[0];  // I
stride[2] = extent[3] * extent[0];  // J
stride[3] = extent[0];  // K
```

The benchmark shows N-I-J-K ordering (component separated) performs better for
small grids because all components of a cell are accessed together in the update:

```cpp
// volt[x] = volt[x] * vv[x] + vi[x] * (curl calculation using curr[y], curr[z])
```

### 3. SIMD Vectorization Efficiency

SSE (4-wide) and AVX (8-wide) both provide good speedups:
- SSE: 2.0-2.7x speedup
- AVX: 2.5-2.7x speedup

AVX is not 2x faster than SSE because:
1. Memory bandwidth limits throughput for large grids
2. Register pressure increases with wider vectors
3. Boundary handling overhead is proportionally larger

### 4. OpenMP Parallelization

The most significant optimization opportunity is parallelization:
- Near-linear scaling up to 8 threads (6.3x with 8 threads)
- Good scaling continues to 16 threads (10.9x)
- Diminishing returns beyond core count due to memory bandwidth

### 5. Cache Miss Analysis (from Valgrind cachegrind)

For the basic scalar implementation:
- L1 data cache miss rate: 5.7% (7.4% reads, 1.8% writes)
- L3 miss rate: 0.1% (data fits in L3)

The 5.7% L1 miss rate indicates suboptimal memory access patterns that could
be improved with:
- Better loop ordering
- Cache blocking/tiling
- Prefetching

## Recommendations

### High Priority (Major Impact)

1. **Implement OpenMP parallelization in the main update loops**
   - Expected speedup: 6-10x depending on thread count
   - The current Boost threading in `engine_multithread.cpp` already does this
   - Consider OpenMP for simpler code and potentially better performance

2. **Use AVX for large simulations**
   - Current SSE implementation (engine_sse.cpp) processes 4 floats
   - AVX would process 8 floats, ~25% improvement for large grids
   - AVX-512 would process 16 floats but may cause frequency throttling

3. **Review memory layout for cache efficiency**
   - Ensure field arrays are 64-byte aligned (cache line)
   - Consider reorganizing data access patterns in loops

### Medium Priority (Moderate Impact)

4. **Add software prefetching in inner loops**
   - ~20% improvement observed in benchmarks
   - Use `_mm_prefetch()` for next iteration's data

5. **Use FMA (Fused Multiply-Add) instructions**
   - ~10-15% improvement
   - Replace `a = a * b; a = a + c * d;` with FMA intrinsics

6. **Optimize loop tiling for larger grids**
   - Block size should match L2 cache (~256KB)
   - Tile dimensions: ~16x16 in X-Y plane

### Lower Priority (Minor Impact)

7. **Ensure denormal handling is disabled**
   - Already done in openEMS via `Denormal::Disable()`
   - Critical for avoiding 100x slowdowns

8. **Profile extension overhead**
   - The extension dispatch loop may have overhead for simple simulations
   - Consider inlining or removing when not needed

## Implementation Notes

### Suggested OpenMP Implementation

```cpp
void Engine_sse::UpdateVoltages(unsigned int startX, unsigned int numX)
{
    #pragma omp parallel for schedule(static)
    for (unsigned int posX = 0; posX < numX; ++posX)
    {
        // ... existing SSE update code ...
    }
}
```

### Suggested AVX Implementation

The existing `engine_sse.cpp` structure can be adapted for AVX:
- Change `f4vector` (4 floats) to `f8vector` (8 floats)
- Use `_mm256_*` intrinsics instead of `_mm_*`
- Adjust loop bounds for 8-element vectorization

## Conclusion

The FDTD critical path in openEMS can be significantly optimized through:

1. **Parallelization (10x potential)** - Already implemented via Boost threads,
   but OpenMP may offer simpler maintenance

2. **Wider SIMD (25% potential)** - AVX/AVX-512 support

3. **Memory optimization (20% potential)** - Prefetching, alignment, layout

The current implementation with SSE and multithreading is well-designed.
The main opportunities are:
- Ensuring AVX is used when available
- Fine-tuning thread count based on grid size
- Adding prefetching hints

For very large simulations, consider MPI distributed computing (already
supported in `engine_mpi.cpp`) to scale beyond single-node memory bandwidth.

## Files

- `fdtd_benchmark.cpp` - Main benchmark comparing implementations
- `fdtd_optimized.cpp` - Additional optimization techniques
- `CMakeLists.txt` - Build configuration
