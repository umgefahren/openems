/*
 * FDTD Kernel Implementation using Google Highway
 *
 * Highway provides portable SIMD that automatically dispatches to the best
 * available instruction set at runtime (SSE4, AVX2, AVX-512, ARM NEON, SVE, etc.)
 *
 * Key benefits over the current f4vector approach:
 * 1. Automatically uses widest available SIMD (up to AVX-512 = 16 floats)
 * 2. Runtime dispatch - same binary works on all CPUs
 * 3. Portable to ARM (Apple M1/M2, AWS Graviton, etc.)
 * 4. Better compiler optimization hints
 * 5. Active development and optimization
 */

#ifndef FDTD_HIGHWAY_H
#define FDTD_HIGHWAY_H

#include <cstdint>
#include <cstddef>

// Include Highway headers
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "fdtd_highway.h"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"
#include "hwy/aligned_allocator.h"

HWY_BEFORE_NAMESPACE();
namespace openems {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

// FDTD Grid structure using Highway-aligned arrays
struct FDTDGrid {
    uint32_t numX, numY, numZ;
    uint32_t numVectors;  // Number of SIMD vectors in Z dimension
    size_t vector_lanes;  // Number of floats per SIMD vector

    // Field arrays (3 polarizations each)
    // Layout: [polarization][x][y][z_vector] where z_vector packs multiple z values
    float* volt[3];
    float* curr[3];

    // Coefficient arrays
    float* vv[3];
    float* vi[3];
    float* ii[3];
    float* iv[3];

    FDTDGrid(uint32_t nx, uint32_t ny, uint32_t nz) {
        const hn::ScalableTag<float> d;
        vector_lanes = hn::Lanes(d);
        numX = nx;
        numY = ny;
        numZ = nz;
        numVectors = (nz + vector_lanes - 1) / vector_lanes;

        size_t array_size = (size_t)nx * ny * numVectors * vector_lanes;

        for (int n = 0; n < 3; ++n) {
            volt[n] = hwy::AllocateAligned<float>(array_size);
            curr[n] = hwy::AllocateAligned<float>(array_size);
            vv[n] = hwy::AllocateAligned<float>(array_size);
            vi[n] = hwy::AllocateAligned<float>(array_size);
            ii[n] = hwy::AllocateAligned<float>(array_size);
            iv[n] = hwy::AllocateAligned<float>(array_size);

            // Initialize to zero
            for (size_t i = 0; i < array_size; ++i) {
                volt[n][i] = 0.0f;
                curr[n][i] = 0.0f;
                vv[n][i] = 1.0f;  // Default coefficient
                vi[n][i] = 0.1f;
                ii[n][i] = 1.0f;
                iv[n][i] = 0.1f;
            }
        }
    }

    ~FDTDGrid() {
        for (int n = 0; n < 3; ++n) {
            hwy::FreeAlignedBytes(volt[n], nullptr, 0);
            hwy::FreeAlignedBytes(curr[n], nullptr, 0);
            hwy::FreeAlignedBytes(vv[n], nullptr, 0);
            hwy::FreeAlignedBytes(vi[n], nullptr, 0);
            hwy::FreeAlignedBytes(ii[n], nullptr, 0);
            hwy::FreeAlignedBytes(iv[n], nullptr, 0);
        }
    }

    // Linear index for accessing arrays
    HWY_INLINE size_t idx(uint32_t x, uint32_t y, uint32_t z_vec) const {
        return ((size_t)x * numY + y) * numVectors * vector_lanes + z_vec * vector_lanes;
    }
};

// UpdateVoltages kernel using Highway
HWY_ATTR void UpdateVoltagesHwy(FDTDGrid& grid, uint32_t startX, uint32_t numX) {
    const hn::ScalableTag<float> d;
    const size_t lanes = hn::Lanes(d);

    for (uint32_t x = startX; x < startX + numX; ++x) {
        const bool shift_x = (x > 0);
        const uint32_t x_prev = x - shift_x;

        for (uint32_t y = 0; y < grid.numY; ++y) {
            const bool shift_y = (y > 0);
            const uint32_t y_prev = y - shift_y;

            // Process all z-vectors (each contains 'lanes' floats)
            for (uint32_t zv = 1; zv < grid.numVectors; ++zv) {
                const size_t idx_xyz = grid.idx(x, y, zv);
                const size_t idx_xyz_z1 = grid.idx(x, y, zv - 1);
                const size_t idx_xyz_y1 = grid.idx(x, y_prev, zv);
                const size_t idx_xyz_x1 = grid.idx(x_prev, y, zv);

                // X-polarization: E_x update
                // dH_z/dy - dH_y/dz
                auto volt_x = hn::Load(d, grid.volt[0] + idx_xyz);
                auto vv_x = hn::Load(d, grid.vv[0] + idx_xyz);
                auto vi_x = hn::Load(d, grid.vi[0] + idx_xyz);

                auto curr_z_xy = hn::Load(d, grid.curr[2] + idx_xyz);
                auto curr_z_xy_y1 = hn::Load(d, grid.curr[2] + idx_xyz_y1);
                auto curr_y_xy = hn::Load(d, grid.curr[1] + idx_xyz);
                auto curr_y_xy_z1 = hn::Load(d, grid.curr[1] + idx_xyz_z1);

                auto curl_h_x = hn::Sub(curr_z_xy, curr_z_xy_y1);
                curl_h_x = hn::Sub(curl_h_x, curr_y_xy);
                curl_h_x = hn::Add(curl_h_x, curr_y_xy_z1);

                volt_x = hn::Mul(volt_x, vv_x);
                volt_x = hn::MulAdd(vi_x, curl_h_x, volt_x);
                hn::Store(volt_x, d, grid.volt[0] + idx_xyz);

                // Y-polarization: E_y update
                // dH_x/dz - dH_z/dx
                auto volt_y = hn::Load(d, grid.volt[1] + idx_xyz);
                auto vv_y = hn::Load(d, grid.vv[1] + idx_xyz);
                auto vi_y = hn::Load(d, grid.vi[1] + idx_xyz);

                auto curr_x_xy = hn::Load(d, grid.curr[0] + idx_xyz);
                auto curr_x_xy_z1 = hn::Load(d, grid.curr[0] + idx_xyz_z1);
                curr_z_xy = hn::Load(d, grid.curr[2] + idx_xyz);
                auto curr_z_x1y = hn::Load(d, grid.curr[2] + idx_xyz_x1);

                auto curl_h_y = hn::Sub(curr_x_xy, curr_x_xy_z1);
                curl_h_y = hn::Sub(curl_h_y, curr_z_xy);
                curl_h_y = hn::Add(curl_h_y, curr_z_x1y);

                volt_y = hn::Mul(volt_y, vv_y);
                volt_y = hn::MulAdd(vi_y, curl_h_y, volt_y);
                hn::Store(volt_y, d, grid.volt[1] + idx_xyz);

                // Z-polarization: E_z update
                // dH_y/dx - dH_x/dy
                auto volt_z = hn::Load(d, grid.volt[2] + idx_xyz);
                auto vv_z = hn::Load(d, grid.vv[2] + idx_xyz);
                auto vi_z = hn::Load(d, grid.vi[2] + idx_xyz);

                curr_y_xy = hn::Load(d, grid.curr[1] + idx_xyz);
                auto curr_y_x1y = hn::Load(d, grid.curr[1] + idx_xyz_x1);
                curr_x_xy = hn::Load(d, grid.curr[0] + idx_xyz);
                auto curr_x_xy1 = hn::Load(d, grid.curr[0] + idx_xyz_y1);

                auto curl_h_z = hn::Sub(curr_y_xy, curr_y_x1y);
                curl_h_z = hn::Sub(curl_h_z, curr_x_xy);
                curl_h_z = hn::Add(curl_h_z, curr_x_xy1);

                volt_z = hn::Mul(volt_z, vv_z);
                volt_z = hn::MulAdd(vi_z, curl_h_z, volt_z);
                hn::Store(volt_z, d, grid.volt[2] + idx_xyz);
            }

            // Handle z=0 boundary (requires special handling for z-1 access)
            // This would need a shifted load or scalar handling
            // For simplicity, we skip z=0 here (boundary condition)
        }
    }
}

// UpdateCurrents kernel using Highway
HWY_ATTR void UpdateCurrentsHwy(FDTDGrid& grid, uint32_t startX, uint32_t numX) {
    const hn::ScalableTag<float> d;
    const size_t lanes = hn::Lanes(d);

    for (uint32_t x = startX; x < startX + numX - 1; ++x) {
        for (uint32_t y = 0; y < grid.numY - 1; ++y) {
            for (uint32_t zv = 0; zv < grid.numVectors - 1; ++zv) {
                const size_t idx_xyz = grid.idx(x, y, zv);
                const size_t idx_xyz_z1 = grid.idx(x, y, zv + 1);
                const size_t idx_xyz_y1 = grid.idx(x, y + 1, zv);
                const size_t idx_xyz_x1 = grid.idx(x + 1, y, zv);

                // X-polarization: H_x update
                // dE_z/dy - dE_y/dz
                auto curr_x = hn::Load(d, grid.curr[0] + idx_xyz);
                auto ii_x = hn::Load(d, grid.ii[0] + idx_xyz);
                auto iv_x = hn::Load(d, grid.iv[0] + idx_xyz);

                auto volt_z_xy = hn::Load(d, grid.volt[2] + idx_xyz);
                auto volt_z_xy1 = hn::Load(d, grid.volt[2] + idx_xyz_y1);
                auto volt_y_xy = hn::Load(d, grid.volt[1] + idx_xyz);
                auto volt_y_xyz1 = hn::Load(d, grid.volt[1] + idx_xyz_z1);

                auto curl_e_x = hn::Sub(volt_z_xy, volt_z_xy1);
                curl_e_x = hn::Sub(curl_e_x, volt_y_xy);
                curl_e_x = hn::Add(curl_e_x, volt_y_xyz1);

                curr_x = hn::Mul(curr_x, ii_x);
                curr_x = hn::MulAdd(iv_x, curl_e_x, curr_x);
                hn::Store(curr_x, d, grid.curr[0] + idx_xyz);

                // Y-polarization: H_y update
                // dE_x/dz - dE_z/dx
                auto curr_y = hn::Load(d, grid.curr[1] + idx_xyz);
                auto ii_y = hn::Load(d, grid.ii[1] + idx_xyz);
                auto iv_y = hn::Load(d, grid.iv[1] + idx_xyz);

                auto volt_x_xy = hn::Load(d, grid.volt[0] + idx_xyz);
                auto volt_x_xyz1 = hn::Load(d, grid.volt[0] + idx_xyz_z1);
                volt_z_xy = hn::Load(d, grid.volt[2] + idx_xyz);
                auto volt_z_x1y = hn::Load(d, grid.volt[2] + idx_xyz_x1);

                auto curl_e_y = hn::Sub(volt_x_xy, volt_x_xyz1);
                curl_e_y = hn::Sub(curl_e_y, volt_z_xy);
                curl_e_y = hn::Add(curl_e_y, volt_z_x1y);

                curr_y = hn::Mul(curr_y, ii_y);
                curr_y = hn::MulAdd(iv_y, curl_e_y, curr_y);
                hn::Store(curr_y, d, grid.curr[1] + idx_xyz);

                // Z-polarization: H_z update
                // dE_y/dx - dE_x/dy
                auto curr_z = hn::Load(d, grid.curr[2] + idx_xyz);
                auto ii_z = hn::Load(d, grid.ii[2] + idx_xyz);
                auto iv_z = hn::Load(d, grid.iv[2] + idx_xyz);

                volt_y_xy = hn::Load(d, grid.volt[1] + idx_xyz);
                auto volt_y_x1y = hn::Load(d, grid.volt[1] + idx_xyz_x1);
                volt_x_xy = hn::Load(d, grid.volt[0] + idx_xyz);
                auto volt_x_xy1 = hn::Load(d, grid.volt[0] + idx_xyz_y1);

                auto curl_e_z = hn::Sub(volt_y_xy, volt_y_x1y);
                curl_e_z = hn::Sub(curl_e_z, volt_x_xy);
                curl_e_z = hn::Add(curl_e_z, volt_x_xy1);

                curr_z = hn::Mul(curr_z, ii_z);
                curr_z = hn::MulAdd(iv_z, curl_e_z, curr_z);
                hn::Store(curr_z, d, grid.curr[2] + idx_xyz);
            }
        }
    }
}

// Full timestep iteration
HWY_ATTR void IterateTimestepHwy(FDTDGrid& grid) {
    UpdateVoltagesHwy(grid, 0, grid.numX);
    UpdateCurrentsHwy(grid, 0, grid.numX);
}

}  // namespace HWY_NAMESPACE
}  // namespace openems
HWY_AFTER_NAMESPACE();

// Dispatch functions that automatically select the best implementation
#if HWY_ONCE
namespace openems {

HWY_EXPORT(UpdateVoltagesHwy);
HWY_EXPORT(UpdateCurrentsHwy);
HWY_EXPORT(IterateTimestepHwy);

void UpdateVoltages(FDTDGrid& grid, uint32_t startX, uint32_t numX) {
    HWY_DYNAMIC_DISPATCH(UpdateVoltagesHwy)(grid, startX, numX);
}

void UpdateCurrents(FDTDGrid& grid, uint32_t startX, uint32_t numX) {
    HWY_DYNAMIC_DISPATCH(UpdateCurrentsHwy)(grid, startX, numX);
}

void IterateTimestep(FDTDGrid& grid) {
    HWY_DYNAMIC_DISPATCH(IterateTimestepHwy)(grid);
}

// Get information about the selected SIMD target
const char* GetSimdTargetName() {
    return hwy::SupportedTargets();
}

}  // namespace openems
#endif  // HWY_ONCE

#endif  // FDTD_HIGHWAY_H
