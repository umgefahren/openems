/*
*	Copyright (C) 2025 OpenEMS Contributors
*
*	This program is free software: you can redistribute it and/or modify
*	it under the terms of the GNU General Public License as published by
*	the Free Software Foundation, either version 3 of the License, or
*	(at your option) any later version.
*
*	This program is distributed in the hope that it will be useful,
*	but WITHOUT ANY WARRANTY; without even the implied warranty of
*	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*	GNU General Public License for more details.
*
*	You should have received a copy of the GNU General Public License
*	along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "engine_hwy_wide.h"
#include "operator_hwy_wide.h"

#include <iostream>
#include <cstring>

// Highway SIMD library
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "FDTD/engine_hwy_wide.cpp"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

HWY_BEFORE_NAMESPACE();
namespace openEMS {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

// Use ScalableTag for maximum SIMD width (16 floats on AVX-512, 8 on AVX2, 4 on SSE)
using D = hn::ScalableTag<float>;

/**
 * @brief True wide SIMD voltage update
 *
 * Processes N floats per iteration where N is the native SIMD width:
 * - AVX-512: 16 floats (512 bits)
 * - AVX2: 8 floats (256 bits)
 * - SSE: 4 floats (128 bits)
 */
HWY_ATTR void UpdateVoltagesWide(
    float* __restrict volt_x,
    float* __restrict volt_y,
    float* __restrict volt_z,
    const float* __restrict curr_x,
    const float* __restrict curr_y,
    const float* __restrict curr_z,
    const float* __restrict vv_x,
    const float* __restrict vv_y,
    const float* __restrict vv_z,
    const float* __restrict vi_x,
    const float* __restrict vi_y,
    const float* __restrict vi_z,
    size_t stride_x,
    size_t stride_y,
    unsigned int nx,
    unsigned int ny,
    unsigned int nz,
    unsigned int paddedNz,
    unsigned int startX,
    unsigned int numX)
{
    const D d;
    const size_t N = hn::Lanes(d);  // SIMD width (16 for AVX-512)

    for (unsigned int px = 0; px < numX; ++px) {
        const unsigned int posX = startX + px;
        const size_t x_off = posX * stride_x;
        const size_t x_off_m1 = (posX > 0) ? (posX - 1) * stride_x : x_off;

        for (unsigned int posY = 0; posY < ny; ++posY) {
            const size_t xy_off = x_off + posY * stride_y;
            const size_t xy_off_ym1 = (posY > 0) ? (x_off + (posY - 1) * stride_y) : xy_off;
            const size_t xy_off_xm1 = x_off_m1 + posY * stride_y;

            // Process z-dimension in chunks of N (SIMD width)
            // Skip z=0 for now, handle boundary separately
            unsigned int posZ = N;
            for (; posZ + N <= paddedNz && posZ < nz; posZ += N) {
                const size_t idx = xy_off + posZ;
                const size_t idx_zm1 = xy_off + posZ - 1;
                const size_t idx_ym1 = xy_off_ym1 + posZ;
                const size_t idx_xm1 = xy_off_xm1 + posZ;

                // Load voltage values
                auto vx = hn::Load(d, volt_x + idx);
                auto vy = hn::Load(d, volt_y + idx);
                auto vz = hn::Load(d, volt_z + idx);

                // Load operator coefficients (contiguous access - very fast!)
                auto vv_x_val = hn::Load(d, vv_x + idx);
                auto vv_y_val = hn::Load(d, vv_y + idx);
                auto vv_z_val = hn::Load(d, vv_z + idx);
                auto vi_x_val = hn::Load(d, vi_x + idx);
                auto vi_y_val = hn::Load(d, vi_y + idx);
                auto vi_z_val = hn::Load(d, vi_z + idx);

                // Load current field values for curl calculation
                auto cx = hn::Load(d, curr_x + idx);
                auto cy = hn::Load(d, curr_y + idx);
                auto cz = hn::Load(d, curr_z + idx);

                // x-polarization: curl = Hz(y) - Hz(y-1) - Hy(z) + Hy(z-1)
                auto curl_x = hn::Sub(cz, hn::Load(d, curr_z + idx_ym1));
                curl_x = hn::Sub(curl_x, cy);
                curl_x = hn::Add(curl_x, hn::Load(d, curr_y + idx_zm1));
                vx = hn::MulAdd(vi_x_val, curl_x, hn::Mul(vx, vv_x_val));
                hn::Store(vx, d, volt_x + idx);

                // y-polarization: curl = Hx(z) - Hx(z-1) - Hz(x) + Hz(x-1)
                auto curl_y = hn::Sub(cx, hn::Load(d, curr_x + idx_zm1));
                curl_y = hn::Sub(curl_y, cz);
                curl_y = hn::Add(curl_y, hn::Load(d, curr_z + idx_xm1));
                vy = hn::MulAdd(vi_y_val, curl_y, hn::Mul(vy, vv_y_val));
                hn::Store(vy, d, volt_y + idx);

                // z-polarization: curl = Hy(x) - Hy(x-1) - Hx(y) + Hx(y-1)
                auto curl_z = hn::Sub(cy, hn::Load(d, curr_y + idx_xm1));
                curl_z = hn::Sub(curl_z, cx);
                curl_z = hn::Add(curl_z, hn::Load(d, curr_x + idx_ym1));
                vz = hn::MulAdd(vi_z_val, curl_z, hn::Mul(vz, vv_z_val));
                hn::Store(vz, d, volt_z + idx);
            }

            // Handle remaining elements (z=1 to z=N-1 and tail)
            for (unsigned int z = 1; z < nz && z < posZ; ++z) {
                const size_t idx = xy_off + z;
                const size_t idx_zm1 = xy_off + z - 1;
                const size_t idx_ym1 = xy_off_ym1 + z;
                const size_t idx_xm1 = xy_off_xm1 + z;

                // x-polarization
                float curl_x = curr_z[idx] - curr_z[idx_ym1] - curr_y[idx] + curr_y[idx_zm1];
                volt_x[idx] = volt_x[idx] * vv_x[idx] + curl_x * vi_x[idx];

                // y-polarization
                float curl_y = curr_x[idx] - curr_x[idx_zm1] - curr_z[idx] + curr_z[idx_xm1];
                volt_y[idx] = volt_y[idx] * vv_y[idx] + curl_y * vi_y[idx];

                // z-polarization
                float curl_z = curr_y[idx] - curr_y[idx_xm1] - curr_x[idx] + curr_x[idx_ym1];
                volt_z[idx] = volt_z[idx] * vv_z[idx] + curl_z * vi_z[idx];
            }

            // Handle z=0 boundary (needs wrap-around from z=nz-1)
            {
                const size_t idx = xy_off;
                const size_t idx_zm1 = xy_off + nz - 1;  // wrap around
                const size_t idx_ym1 = xy_off_ym1;
                const size_t idx_xm1 = xy_off_xm1;

                // x-polarization
                float curl_x = curr_z[idx] - curr_z[idx_ym1] - curr_y[idx] + curr_y[idx_zm1];
                volt_x[idx] = volt_x[idx] * vv_x[idx] + curl_x * vi_x[idx];

                // y-polarization
                float curl_y = curr_x[idx] - curr_x[idx_zm1] - curr_z[idx] + curr_z[idx_xm1];
                volt_y[idx] = volt_y[idx] * vv_y[idx] + curl_y * vi_y[idx];

                // z-polarization
                float curl_z = curr_y[idx] - curr_y[idx_xm1] - curr_x[idx] + curr_x[idx_ym1];
                volt_z[idx] = volt_z[idx] * vv_z[idx] + curl_z * vi_z[idx];
            }
        }
    }
}

/**
 * @brief True wide SIMD current update
 */
HWY_ATTR void UpdateCurrentsWide(
    float* __restrict curr_x,
    float* __restrict curr_y,
    float* __restrict curr_z,
    const float* __restrict volt_x,
    const float* __restrict volt_y,
    const float* __restrict volt_z,
    const float* __restrict ii_x,
    const float* __restrict ii_y,
    const float* __restrict ii_z,
    const float* __restrict iv_x,
    const float* __restrict iv_y,
    const float* __restrict iv_z,
    size_t stride_x,
    size_t stride_y,
    unsigned int nx,
    unsigned int ny,
    unsigned int nz,
    unsigned int paddedNz,
    unsigned int startX,
    unsigned int numX)
{
    const D d;
    const size_t N = hn::Lanes(d);

    for (unsigned int px = 0; px < numX; ++px) {
        const unsigned int posX = startX + px;
        const size_t x_off = posX * stride_x;
        const size_t x_off_p1 = (posX + 1 < nx) ? (posX + 1) * stride_x : x_off;

        for (unsigned int posY = 0; posY < ny - 1; ++posY) {
            const size_t xy_off = x_off + posY * stride_y;
            const size_t xy_off_yp1 = x_off + (posY + 1) * stride_y;
            const size_t xy_off_xp1 = x_off_p1 + posY * stride_y;

            // Process z-dimension in chunks of N
            unsigned int posZ = 0;
            for (; posZ + N <= paddedNz && posZ + N <= nz - 1; posZ += N) {
                const size_t idx = xy_off + posZ;
                const size_t idx_zp1 = xy_off + posZ + 1;
                const size_t idx_yp1 = xy_off_yp1 + posZ;
                const size_t idx_xp1 = xy_off_xp1 + posZ;

                // Load current values
                auto cx = hn::Load(d, curr_x + idx);
                auto cy = hn::Load(d, curr_y + idx);
                auto cz = hn::Load(d, curr_z + idx);

                // Load operator coefficients
                auto ii_x_val = hn::Load(d, ii_x + idx);
                auto ii_y_val = hn::Load(d, ii_y + idx);
                auto ii_z_val = hn::Load(d, ii_z + idx);
                auto iv_x_val = hn::Load(d, iv_x + idx);
                auto iv_y_val = hn::Load(d, iv_y + idx);
                auto iv_z_val = hn::Load(d, iv_z + idx);

                // Load voltage field values
                auto vx = hn::Load(d, volt_x + idx);
                auto vy = hn::Load(d, volt_y + idx);
                auto vz = hn::Load(d, volt_z + idx);

                // x-polarization: curl = Ez(y) - Ez(y+1) - Ey(z) + Ey(z+1)
                auto curl_x = hn::Sub(vz, hn::Load(d, volt_z + idx_yp1));
                curl_x = hn::Sub(curl_x, vy);
                curl_x = hn::Add(curl_x, hn::Load(d, volt_y + idx_zp1));
                cx = hn::MulAdd(iv_x_val, curl_x, hn::Mul(cx, ii_x_val));
                hn::Store(cx, d, curr_x + idx);

                // y-polarization: curl = Ex(z) - Ex(z+1) - Ez(x) + Ez(x+1)
                auto curl_y = hn::Sub(vx, hn::Load(d, volt_x + idx_zp1));
                curl_y = hn::Sub(curl_y, vz);
                curl_y = hn::Add(curl_y, hn::Load(d, volt_z + idx_xp1));
                cy = hn::MulAdd(iv_y_val, curl_y, hn::Mul(cy, ii_y_val));
                hn::Store(cy, d, curr_y + idx);

                // z-polarization: curl = Ey(x) - Ey(x+1) - Ex(y) + Ex(y+1)
                auto curl_z = hn::Sub(vy, hn::Load(d, volt_y + idx_xp1));
                curl_z = hn::Sub(curl_z, vx);
                curl_z = hn::Add(curl_z, hn::Load(d, volt_x + idx_yp1));
                cz = hn::MulAdd(iv_z_val, curl_z, hn::Mul(cz, ii_z_val));
                hn::Store(cz, d, curr_z + idx);
            }

            // Handle remaining elements
            for (; posZ < nz - 1; ++posZ) {
                const size_t idx = xy_off + posZ;
                const size_t idx_zp1 = xy_off + posZ + 1;
                const size_t idx_yp1 = xy_off_yp1 + posZ;
                const size_t idx_xp1 = xy_off_xp1 + posZ;

                // x-polarization
                float curl_x = volt_z[idx] - volt_z[idx_yp1] - volt_y[idx] + volt_y[idx_zp1];
                curr_x[idx] = curr_x[idx] * ii_x[idx] + curl_x * iv_x[idx];

                // y-polarization
                float curl_y = volt_x[idx] - volt_x[idx_zp1] - volt_z[idx] + volt_z[idx_xp1];
                curr_y[idx] = curr_y[idx] * ii_y[idx] + curl_y * iv_y[idx];

                // z-polarization
                float curl_z = volt_y[idx] - volt_y[idx_xp1] - volt_x[idx] + volt_x[idx_yp1];
                curr_z[idx] = curr_z[idx] * ii_z[idx] + curl_z * iv_z[idx];
            }

            // Handle z=nz-1 boundary (needs wrap-around)
            {
                const size_t idx = xy_off + nz - 1;
                const size_t idx_zp1 = xy_off;  // wrap around
                const size_t idx_yp1 = xy_off_yp1 + nz - 1;
                const size_t idx_xp1 = xy_off_xp1 + nz - 1;

                // x-polarization
                float curl_x = volt_z[idx] - volt_z[idx_yp1] - volt_y[idx] + volt_y[idx_zp1];
                curr_x[idx] = curr_x[idx] * ii_x[idx] + curl_x * iv_x[idx];

                // y-polarization
                float curl_y = volt_x[idx] - volt_x[idx_zp1] - volt_z[idx] + volt_z[idx_xp1];
                curr_y[idx] = curr_y[idx] * ii_y[idx] + curl_y * iv_y[idx];

                // z-polarization
                float curl_z = volt_y[idx] - volt_y[idx_xp1] - volt_x[idx] + volt_x[idx_yp1];
                curr_z[idx] = curr_z[idx] * ii_z[idx] + curl_z * iv_z[idx];
            }
        }
    }
}

}  // namespace HWY_NAMESPACE
}  // namespace openEMS
HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace openEMS {

// Function pointer types
using UpdateVoltagesWidePtr = void (*)(
    float*, float*, float*, const float*, const float*, const float*,
    const float*, const float*, const float*, const float*, const float*, const float*,
    size_t, size_t, unsigned int, unsigned int, unsigned int, unsigned int,
    unsigned int, unsigned int);

using UpdateCurrentsWidePtr = void (*)(
    float*, float*, float*, const float*, const float*, const float*,
    const float*, const float*, const float*, const float*, const float*, const float*,
    size_t, size_t, unsigned int, unsigned int, unsigned int, unsigned int,
    unsigned int, unsigned int);

HWY_EXPORT(UpdateVoltagesWide);
HWY_EXPORT(UpdateCurrentsWide);

static UpdateVoltagesWidePtr g_UpdateVoltagesWidePtr = nullptr;
static UpdateCurrentsWidePtr g_UpdateCurrentsWidePtr = nullptr;

void InitWideFunctionPointers() {
    if (!g_UpdateVoltagesWidePtr) {
        g_UpdateVoltagesWidePtr = HWY_DYNAMIC_POINTER(UpdateVoltagesWide);
        g_UpdateCurrentsWidePtr = HWY_DYNAMIC_POINTER(UpdateCurrentsWide);
    }
}

}  // namespace openEMS

using std::cout;
using std::endl;

Engine_Hwy_Wide* Engine_Hwy_Wide::New(const Operator_Hwy_Wide* op, unsigned int numThreads)
{
    // Get SIMD info
    const auto target = hwy::SupportedTargets();
    const char* simd_name = "Unknown";
    unsigned int simd_width = 4;

    if (target & HWY_AVX3_DL) { simd_name = "AVX-512"; simd_width = 16; }
    else if (target & HWY_AVX3) { simd_name = "AVX-512"; simd_width = 16; }
    else if (target & HWY_AVX2) { simd_name = "AVX2"; simd_width = 8; }
    else if (target & HWY_SSE4) { simd_name = "SSE4"; simd_width = 4; }
    else if (target & HWY_SSE2) { simd_name = "SSE2"; simd_width = 4; }
    else if (target & HWY_NEON) { simd_name = "NEON"; simd_width = 4; }

    cout << "Create FDTD engine (Highway Wide SIMD - " << simd_name
         << " @ " << simd_width << " floats/op + multi-threading)" << endl;

    openEMS::InitWideFunctionPointers();

    Engine_Hwy_Wide* e = new Engine_Hwy_Wide(op);
    e->setNumThreads(numThreads);
    e->Init();
    return e;
}

Engine_Hwy_Wide::Engine_Hwy_Wide(const Operator_Hwy_Wide* op) : Engine_Multithread(op)
{
    m_Op_Wide = op;
    m_type = SSE;  // For compatibility with extensions

    for (int n = 0; n < 3; ++n) {
        m_volt[n] = nullptr;
        m_curr[n] = nullptr;
    }

    m_nx = m_ny = m_nz = 0;
    m_paddedNz = 0;
    m_stride_x = m_stride_y = 0;
    m_totalSize = 0;
}

Engine_Hwy_Wide::~Engine_Hwy_Wide()
{
    Reset();
}

void Engine_Hwy_Wide::Init()
{
    Engine_Multithread::Init();

    // Get dimensions from operator
    m_nx = m_Op_Wide->GetNumberOfLines(0, true);
    m_ny = m_Op_Wide->GetNumberOfLines(1, true);
    m_nz = m_Op_Wide->GetNumberOfLines(2, true);
    m_paddedNz = m_Op_Wide->GetPaddedNz();
    m_stride_x = m_Op_Wide->GetStrideX();
    m_stride_y = m_Op_Wide->GetStrideY();
    m_totalSize = static_cast<size_t>(m_nx) * m_stride_x;

    // Allocate field arrays
    for (int n = 0; n < 3; ++n) {
        m_volt[n] = AlignedAlloc<float>(m_totalSize);
        m_curr[n] = AlignedAlloc<float>(m_totalSize);
        std::memset(m_volt[n], 0, m_totalSize * sizeof(float));
        std::memset(m_curr[n], 0, m_totalSize * sizeof(float));
    }

    cout << "  Field array memory: "
         << (m_totalSize * sizeof(float) * 6 / (1024*1024)) << " MB" << endl;
}

void Engine_Hwy_Wide::Reset()
{
    for (int n = 0; n < 3; ++n) {
        if (m_volt[n]) { std::free(m_volt[n]); m_volt[n] = nullptr; }
        if (m_curr[n]) { std::free(m_curr[n]); m_curr[n] = nullptr; }
    }
    Engine_Multithread::Reset();
}

void Engine_Hwy_Wide::UpdateVoltages(unsigned int startX, unsigned int numX)
{
    openEMS::g_UpdateVoltagesWidePtr(
        m_volt[0], m_volt[1], m_volt[2],
        m_curr[0], m_curr[1], m_curr[2],
        m_Op_Wide->m_vv[0], m_Op_Wide->m_vv[1], m_Op_Wide->m_vv[2],
        m_Op_Wide->m_vi[0], m_Op_Wide->m_vi[1], m_Op_Wide->m_vi[2],
        m_stride_x, m_stride_y,
        m_nx, m_ny, m_nz, m_paddedNz,
        startX, numX);
}

void Engine_Hwy_Wide::UpdateCurrents(unsigned int startX, unsigned int numX)
{
    openEMS::g_UpdateCurrentsWidePtr(
        m_curr[0], m_curr[1], m_curr[2],
        m_volt[0], m_volt[1], m_volt[2],
        m_Op_Wide->m_ii[0], m_Op_Wide->m_ii[1], m_Op_Wide->m_ii[2],
        m_Op_Wide->m_iv[0], m_Op_Wide->m_iv[1], m_Op_Wide->m_iv[2],
        m_stride_x, m_stride_y,
        m_nx, m_ny, m_nz, m_paddedNz,
        startX, numX);
}

// Field accessors
FDTD_FLOAT Engine_Hwy_Wide::GetVolt(unsigned int n, unsigned int x, unsigned int y, unsigned int z) const
{
    return m_volt[n][LinearIndex(x, y, z)];
}

FDTD_FLOAT Engine_Hwy_Wide::GetVolt(unsigned int n, const unsigned int pos[3]) const
{
    return m_volt[n][LinearIndex(pos[0], pos[1], pos[2])];
}

FDTD_FLOAT Engine_Hwy_Wide::GetCurr(unsigned int n, unsigned int x, unsigned int y, unsigned int z) const
{
    return m_curr[n][LinearIndex(x, y, z)];
}

FDTD_FLOAT Engine_Hwy_Wide::GetCurr(unsigned int n, const unsigned int pos[3]) const
{
    return m_curr[n][LinearIndex(pos[0], pos[1], pos[2])];
}

void Engine_Hwy_Wide::SetVolt(unsigned int n, unsigned int x, unsigned int y, unsigned int z, FDTD_FLOAT value)
{
    m_volt[n][LinearIndex(x, y, z)] = value;
}

void Engine_Hwy_Wide::SetVolt(unsigned int n, const unsigned int pos[3], FDTD_FLOAT value)
{
    m_volt[n][LinearIndex(pos[0], pos[1], pos[2])] = value;
}

void Engine_Hwy_Wide::SetCurr(unsigned int n, unsigned int x, unsigned int y, unsigned int z, FDTD_FLOAT value)
{
    m_curr[n][LinearIndex(x, y, z)] = value;
}

void Engine_Hwy_Wide::SetCurr(unsigned int n, const unsigned int pos[3], FDTD_FLOAT value)
{
    m_curr[n][LinearIndex(pos[0], pos[1], pos[2])] = value;
}

#endif  // HWY_ONCE
