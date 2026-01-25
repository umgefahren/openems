# OpenEMS Performance Analysis Report

## Executive Summary

This report documents a comprehensive performance analysis of OpenEMS, an electromagnetic simulation software using the Finite Difference Time Domain (FDTD) method. The analysis includes profiling, benchmarking, and identification of optimization opportunities.

## System Information

- **CPU Features**: SSE, SSE2, SSE4.1, SSE4.2, AVX, AVX2, AVX-512
- **Compiler**: GCC 13.3.0
- **Build Type**: RelWithDebInfo

## Benchmark Results

Benchmark configuration: 80x80x80 grid (531,441 cells), 2000 timesteps

| Engine Type | Performance (MCells/s) | Speedup vs Basic |
|-------------|------------------------|------------------|
| basic | 51.05 | 1.0x |
| sse | 148.03 | 2.9x |
| sse-compressed | 240.59 | 4.7x |
| multithreaded | 247.02 | 4.8x |

### Key Findings

1. **SSE vectorization provides 2.9x speedup** over the basic scalar implementation
2. **Operator compression provides additional 1.6x speedup** (148 -> 240 MCells/s) through reduced memory bandwidth
3. **Multithreading provides marginal improvement** (1.03x) for this problem size due to synchronization overhead

## Profiling Results

### Basic Engine (Callgrind, 10.5B instructions)

| Function | % of Total | Purpose |
|----------|------------|---------|
| Engine::UpdateVoltages (engine.cpp) | 18.95% | Voltage field update |
| Engine::UpdateVoltages (array_nijk.h) | 19.96% | Array access overhead |
| Engine::UpdateCurrents (engine.cpp) | 17.13% | Current field update |
| Engine::UpdateCurrents (array_nijk.h) | 17.01% | Array access overhead |
| AdrOp::GetPos | 4.27% | Coordinate calculations |
| Operator setup functions | ~26% | One-time setup cost |

**Observation**: FDTD update loops consume ~72% of total runtime.

### SSE Engine (Callgrind, 5.1B instructions - 51% fewer)

| Function | % of Total | Purpose |
|----------|------------|---------|
| Engine_sse::UpdateVoltages | 21.6% | Vectorized voltage update |
| Engine_sse::UpdateCurrents | 22.3% | Vectorized current update |
| AdrOp::GetPos | 8.84% | Coordinate calculations (setup) |
| Operator functions | ~26% | One-time setup cost |

**Observation**: SSE vectorization halves the instruction count while maintaining similar loop structure.

## Architecture Analysis

### Current SIMD Implementation

The current SSE implementation:
1. Uses 128-bit vectors (`f4vector` = 4 floats)
2. Vectorizes along the Z-axis (innermost loop)
3. Handles boundary conditions with special cases
4. Uses denormal flush-to-zero for numerical stability

### Compression Strategy

The "compressed" operator optimization:
1. Creates a lookup table of unique coefficient combinations
2. Stores only an index per cell instead of full coefficients
3. Reduces memory bandwidth requirements significantly
4. Works because most cells in typical simulations share material properties

## Optimization Opportunities

### High ROI (Recommended)

1. **AVX2/AVX-512 Vectorization**
   - Current SSE: 128-bit (4 floats)
   - AVX2: 256-bit (8 floats) - potential 2x speedup
   - AVX-512: 512-bit (16 floats) - potential 4x speedup
   - **Recommendation**: Use Google Highway for portable SIMD
   - **Expected Benefit**: 1.5-2x improvement over SSE-compressed

2. **Cache Blocking / Tiling**
   - Current: Processes entire X-Y planes before moving Z
   - Improvement: Process in 3D tiles that fit in L2 cache
   - **Expected Benefit**: 10-30% improvement for large problems

3. **Improved Threading**
   - Current: Barrier synchronization every timestep
   - Improvement: Use lock-free or batched synchronization
   - Consider std::execution parallel algorithms (C++17)
   - **Expected Benefit**: Better scaling for >4 threads

### Medium ROI

4. **Memory Layout Optimization**
   - Current: SoA for components, AoS within vectors
   - Consider: Full SoA with vectorized Z-dimension
   - **Trade-off**: May complicate boundary handling

5. **Prefetching**
   - Add software prefetch hints for next iteration's data
   - **Expected Benefit**: 5-15% for memory-bound cases

### Low ROI (Nice to Have)

6. **Operator Setup Optimization**
   - AdrOp::GetPos takes 8.84% in SSE profile
   - Only affects simulation startup, not runtime
   - Consider only for very short simulations

## Implementation Recommendations

### Using Google Highway

Google Highway provides portable SIMD that automatically selects the best available instruction set. Example integration:

```cpp
#include <hwy/highway.h>

HWY_BEFORE_NAMESPACE();
namespace openems {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

void UpdateVoltagesKernel(/*...*/) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);  // 4 for SSE, 8 for AVX2, 16 for AVX-512

    // Process N floats at a time
    for (size_t i = 0; i < size; i += N) {
        auto v = hn::Load(d, data + i);
        // ... operations ...
        hn::Store(v, d, data + i);
    }
}

}  // namespace HWY_NAMESPACE
}  // namespace openems
HWY_AFTER_NAMESPACE();
```

### Using C++17 Parallel Algorithms

```cpp
#include <execution>
#include <algorithm>

// Replace manual threading with parallel algorithms
std::for_each(std::execution::par_unseq,
    indices.begin(), indices.end(),
    [&](size_t idx) {
        UpdateCell(idx);
    });
```

## Build System Changes Required

To integrate Highway:

```cmake
# In CMakeLists.txt
find_package(hwy REQUIRED)
target_link_libraries(openEMS PRIVATE hwy::hwy)
set(CMAKE_CXX_STANDARD 17)
```

## Files Modified/Created During Analysis

### Created (Benchmark Infrastructure)
- `benchmark/benchmark_sim.py` - Python benchmark script
- `benchmark/run_profiling.py` - XML generation for profiling

### Build Changes
- Tagged repository with v0.0.36 for CMake version detection

## Conclusion

The OpenEMS FDTD engine is already well-optimized with SSE vectorization and operator compression. The main opportunities for further improvement are:

1. **AVX2/AVX-512 via Highway** (1.5-2x potential) - Highest ROI
2. **Cache blocking** (10-30% potential) - Medium effort
3. **Threading improvements** (better scaling) - For large problems

The current "sse-compressed" engine represents an excellent baseline at 240 MCells/s, with the combination of SSE vectorization and memory bandwidth optimization providing a 4.7x speedup over the basic implementation.

---
*Generated by performance analysis session, January 2026*
