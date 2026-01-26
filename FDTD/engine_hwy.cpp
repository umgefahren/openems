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

#include "engine_hwy.h"
#include "operator_hwy.h"

#include <iostream>

// Highway SIMD library
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "FDTD/engine_hwy.cpp"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

HWY_BEFORE_NAMESPACE();
namespace openEMS {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

// Use FixedTag<float, 4> for 128-bit operations (matches f4vector)
using F32x4 = hn::FixedTag<float, 4>;

/**
 * @brief Optimized voltage update with register reuse
 *
 * Key optimizations:
 * - Load current field values once and reuse across polarizations
 * - Pre-compute all strides and base pointers
 * - Cache operator data pointers
 */
HWY_ATTR void UpdateVoltagesHwy(
    f4vector* __restrict f4_volt,
    f4vector* __restrict f4_curr,
    const Operator_SSE_Compressed* Op,
    const ArrayLib::ArrayENG<f4vector>* f4_volt_ptr,
    const ArrayLib::ArrayENG<f4vector>* f4_curr_ptr,
    unsigned int numLines1,
    unsigned int numVectors,
    unsigned int startX,
    unsigned int numX)
{
    const F32x4 d;

    // Pre-compute all strides once
    const int v_stride_n = f4_volt_ptr->stride(0);
    const int v_stride_x = f4_volt_ptr->stride(1);
    const int v_stride_y = f4_volt_ptr->stride(2);
    const int v_stride_z = f4_volt_ptr->stride(3);

    const int i_stride_n = f4_curr_ptr->stride(0);
    const int i_stride_x = f4_curr_ptr->stride(1);
    const int i_stride_y = f4_curr_ptr->stride(2);
    const int i_stride_z = f4_curr_ptr->stride(3);

    const unsigned int idx_stride_x = Op->m_Op_index.stride(0);
    const unsigned int idx_stride_y = Op->m_Op_index.stride(1);
    const unsigned int* op_idx_data = Op->m_Op_index.data();

    const f4vector* vv0 = Op->f4_vv_Compressed[0].data();
    const f4vector* vv1 = Op->f4_vv_Compressed[1].data();
    const f4vector* vv2 = Op->f4_vv_Compressed[2].data();
    const f4vector* vi0 = Op->f4_vi_Compressed[0].data();
    const f4vector* vi1 = Op->f4_vi_Compressed[1].data();
    const f4vector* vi2 = Op->f4_vi_Compressed[2].data();

    for (unsigned int px = 0; px < numX; ++px) {
        const unsigned int posX = startX + px;
        const int i_shift_x = (posX > 0) ? i_stride_x : 0;

        const int volt_base_x = posX * v_stride_x;
        const int curr_base_x = posX * i_stride_x;
        const unsigned int idx_base_x = posX * idx_stride_x;

        for (unsigned int posY = 0; posY < numLines1; ++posY) {
            const int i_shift_y = (posY > 0) ? i_stride_y : 0;

            const int volt_base_xy = volt_base_x + posY * v_stride_y;
            const int curr_base_xy = curr_base_x + posY * i_stride_y;
            const unsigned int idx_base_xy = idx_base_x + posY * idx_stride_y;

            // Main loop - process z from 1 to numVectors-1
            for (unsigned int posZ = 1; posZ < numVectors; ++posZ) {
                const unsigned int index = op_idx_data[idx_base_xy + posZ];
                const int v_off = volt_base_xy + posZ * v_stride_z;
                const int i_off = curr_base_xy + posZ * i_stride_z;

                // Load current field values ONCE and reuse across polarizations
                const auto curr_hx = hn::Load(d, f4_curr[i_off].f);
                const auto curr_hy = hn::Load(d, f4_curr[i_off + i_stride_n].f);
                const auto curr_hz = hn::Load(d, f4_curr[i_off + 2*i_stride_n].f);

                // x-polarization: curl = Hz(y) - Hz(y-1) - Hy(z) + Hy(z-1)
                auto volt_x = hn::Load(d, f4_volt[v_off].f);
                auto vv_x = hn::Load(d, vv0[index].f);
                auto vi_x = hn::Load(d, vi0[index].f);

                auto curl_x = hn::Sub(curr_hz, hn::Load(d, f4_curr[i_off + 2*i_stride_n - i_shift_y].f));
                curl_x = hn::Sub(curl_x, curr_hy);
                curl_x = hn::Add(curl_x, hn::Load(d, f4_curr[i_off + i_stride_n - i_stride_z].f));

                volt_x = hn::MulAdd(vi_x, curl_x, hn::Mul(volt_x, vv_x));
                hn::Store(volt_x, d, f4_volt[v_off].f);

                // y-polarization: curl = Hx(z) - Hx(z-1) - Hz(x) + Hz(x-1)
                const int v_off_y = v_off + v_stride_n;
                auto volt_y = hn::Load(d, f4_volt[v_off_y].f);
                auto vv_y = hn::Load(d, vv1[index].f);
                auto vi_y = hn::Load(d, vi1[index].f);

                auto curl_y = hn::Sub(curr_hx, hn::Load(d, f4_curr[i_off - i_stride_z].f));
                curl_y = hn::Sub(curl_y, curr_hz);
                curl_y = hn::Add(curl_y, hn::Load(d, f4_curr[i_off + 2*i_stride_n - i_shift_x].f));

                volt_y = hn::MulAdd(vi_y, curl_y, hn::Mul(volt_y, vv_y));
                hn::Store(volt_y, d, f4_volt[v_off_y].f);

                // z-polarization: curl = Hy(x) - Hy(x-1) - Hx(y) + Hx(y-1)
                const int v_off_z = v_off_y + v_stride_n;
                auto volt_z = hn::Load(d, f4_volt[v_off_z].f);
                auto vv_z = hn::Load(d, vv2[index].f);
                auto vi_z = hn::Load(d, vi2[index].f);

                auto curl_z = hn::Sub(curr_hy, hn::Load(d, f4_curr[i_off + i_stride_n - i_shift_x].f));
                curl_z = hn::Sub(curl_z, curr_hx);
                curl_z = hn::Add(curl_z, hn::Load(d, f4_curr[i_off - i_shift_y].f));

                volt_z = hn::MulAdd(vi_z, curl_z, hn::Mul(volt_z, vv_z));
                hn::Store(volt_z, d, f4_volt[v_off_z].f);
            }

            // Handle z=0 boundary (wrap-around for periodic BC)
            const unsigned int index = op_idx_data[idx_base_xy];
            const int v_off = volt_base_xy;
            const int i_off_start = curr_base_xy;
            const int i_off_end = curr_base_xy + (numVectors - 1) * i_stride_z;

            alignas(16) float temp_arr[4];

            // x-polarization at z=0
            temp_arr[0] = 0;
            temp_arr[1] = f4_curr[i_off_end + i_stride_n].f[0];
            temp_arr[2] = f4_curr[i_off_end + i_stride_n].f[1];
            temp_arr[3] = f4_curr[i_off_end + i_stride_n].f[2];

            auto volt = hn::Load(d, f4_volt[v_off].f);
            auto vv = hn::Load(d, vv0[index].f);
            auto vi = hn::Load(d, vi0[index].f);
            auto curl = hn::Sub(
                hn::Load(d, f4_curr[i_off_start + 2*i_stride_n].f),
                hn::Load(d, f4_curr[i_off_start + 2*i_stride_n - i_shift_y].f)
            );
            curl = hn::Sub(curl, hn::Load(d, f4_curr[i_off_start + i_stride_n].f));
            curl = hn::Add(curl, hn::Load(d, temp_arr));
            volt = hn::MulAdd(vi, curl, hn::Mul(volt, vv));
            hn::Store(volt, d, f4_volt[v_off].f);

            // y-polarization at z=0
            temp_arr[0] = 0;
            temp_arr[1] = f4_curr[i_off_end].f[0];
            temp_arr[2] = f4_curr[i_off_end].f[1];
            temp_arr[3] = f4_curr[i_off_end].f[2];

            const int v_off_y0 = v_off + v_stride_n;
            volt = hn::Load(d, f4_volt[v_off_y0].f);
            vv = hn::Load(d, vv1[index].f);
            vi = hn::Load(d, vi1[index].f);
            curl = hn::Sub(hn::Load(d, f4_curr[i_off_start].f), hn::Load(d, temp_arr));
            curl = hn::Sub(curl, hn::Load(d, f4_curr[i_off_start + 2*i_stride_n].f));
            curl = hn::Add(curl, hn::Load(d, f4_curr[i_off_start + 2*i_stride_n - i_shift_x].f));
            volt = hn::MulAdd(vi, curl, hn::Mul(volt, vv));
            hn::Store(volt, d, f4_volt[v_off_y0].f);

            // z-polarization at z=0
            const int v_off_z0 = v_off_y0 + v_stride_n;
            volt = hn::Load(d, f4_volt[v_off_z0].f);
            vv = hn::Load(d, vv2[index].f);
            vi = hn::Load(d, vi2[index].f);
            curl = hn::Sub(
                hn::Load(d, f4_curr[i_off_start + i_stride_n].f),
                hn::Load(d, f4_curr[i_off_start + i_stride_n - i_shift_x].f)
            );
            curl = hn::Sub(curl, hn::Load(d, f4_curr[i_off_start].f));
            curl = hn::Add(curl, hn::Load(d, f4_curr[i_off_start - i_shift_y].f));
            volt = hn::MulAdd(vi, curl, hn::Mul(volt, vv));
            hn::Store(volt, d, f4_volt[v_off_z0].f);
        }
    }
}

/**
 * @brief Optimized current update with register reuse
 */
HWY_ATTR void UpdateCurrentsHwy(
    f4vector* __restrict f4_curr,
    f4vector* __restrict f4_volt,
    const Operator_SSE_Compressed* Op,
    const ArrayLib::ArrayENG<f4vector>* f4_volt_ptr,
    const ArrayLib::ArrayENG<f4vector>* f4_curr_ptr,
    unsigned int numLines1,
    unsigned int numVectors,
    unsigned int startX,
    unsigned int numX)
{
    const F32x4 d;

    const int v_stride_n = f4_volt_ptr->stride(0);
    const int v_stride_x = f4_volt_ptr->stride(1);
    const int v_stride_y = f4_volt_ptr->stride(2);
    const int v_stride_z = f4_volt_ptr->stride(3);

    const int i_stride_n = f4_curr_ptr->stride(0);
    const int i_stride_x = f4_curr_ptr->stride(1);
    const int i_stride_y = f4_curr_ptr->stride(2);
    const int i_stride_z = f4_curr_ptr->stride(3);

    const unsigned int idx_stride_x = Op->m_Op_index.stride(0);
    const unsigned int idx_stride_y = Op->m_Op_index.stride(1);
    const unsigned int* op_idx_data = Op->m_Op_index.data();

    const f4vector* ii0 = Op->f4_ii_Compressed[0].data();
    const f4vector* ii1 = Op->f4_ii_Compressed[1].data();
    const f4vector* ii2 = Op->f4_ii_Compressed[2].data();
    const f4vector* iv0 = Op->f4_iv_Compressed[0].data();
    const f4vector* iv1 = Op->f4_iv_Compressed[1].data();
    const f4vector* iv2 = Op->f4_iv_Compressed[2].data();

    for (unsigned int px = 0; px < numX; ++px) {
        const unsigned int posX = startX + px;

        const int volt_base_x = posX * v_stride_x;
        const int curr_base_x = posX * i_stride_x;
        const unsigned int idx_base_x = posX * idx_stride_x;

        for (unsigned int posY = 0; posY < numLines1 - 1; ++posY) {
            const int volt_base_xy = volt_base_x + posY * v_stride_y;
            const int curr_base_xy = curr_base_x + posY * i_stride_y;
            const unsigned int idx_base_xy = idx_base_x + posY * idx_stride_y;

            for (unsigned int posZ = 0; posZ < numVectors - 1; ++posZ) {
                const unsigned int index = op_idx_data[idx_base_xy + posZ];
                const int v_off = volt_base_xy + posZ * v_stride_z;
                const int i_off = curr_base_xy + posZ * i_stride_z;

                // Load voltage field values ONCE and reuse
                const auto volt_ex = hn::Load(d, f4_volt[v_off].f);
                const auto volt_ey = hn::Load(d, f4_volt[v_off + v_stride_n].f);
                const auto volt_ez = hn::Load(d, f4_volt[v_off + 2*v_stride_n].f);

                // x-pol: curl = Ez(y) - Ez(y+1) - Ey(z) + Ey(z+1)
                auto curr_x = hn::Load(d, f4_curr[i_off].f);
                auto ii_x = hn::Load(d, ii0[index].f);
                auto iv_x = hn::Load(d, iv0[index].f);

                auto curl_x = hn::Sub(volt_ez, hn::Load(d, f4_volt[v_off + 2*v_stride_n + v_stride_y].f));
                curl_x = hn::Sub(curl_x, volt_ey);
                curl_x = hn::Add(curl_x, hn::Load(d, f4_volt[v_off + v_stride_n + v_stride_z].f));

                curr_x = hn::MulAdd(iv_x, curl_x, hn::Mul(curr_x, ii_x));
                hn::Store(curr_x, d, f4_curr[i_off].f);

                // y-pol: curl = Ex(z) - Ex(z+1) - Ez(x) + Ez(x+1)
                const int i_off_y = i_off + i_stride_n;
                auto curr_y = hn::Load(d, f4_curr[i_off_y].f);
                auto ii_y = hn::Load(d, ii1[index].f);
                auto iv_y = hn::Load(d, iv1[index].f);

                auto curl_y = hn::Sub(volt_ex, hn::Load(d, f4_volt[v_off + v_stride_z].f));
                curl_y = hn::Sub(curl_y, volt_ez);
                curl_y = hn::Add(curl_y, hn::Load(d, f4_volt[v_off + 2*v_stride_n + v_stride_x].f));

                curr_y = hn::MulAdd(iv_y, curl_y, hn::Mul(curr_y, ii_y));
                hn::Store(curr_y, d, f4_curr[i_off_y].f);

                // z-pol: curl = Ey(x) - Ey(x+1) - Ex(y) + Ex(y+1)
                const int i_off_z = i_off_y + i_stride_n;
                auto curr_z = hn::Load(d, f4_curr[i_off_z].f);
                auto ii_z = hn::Load(d, ii2[index].f);
                auto iv_z = hn::Load(d, iv2[index].f);

                auto curl_z = hn::Sub(volt_ey, hn::Load(d, f4_volt[v_off + v_stride_n + v_stride_x].f));
                curl_z = hn::Sub(curl_z, volt_ex);
                curl_z = hn::Add(curl_z, hn::Load(d, f4_volt[v_off + v_stride_y].f));

                curr_z = hn::MulAdd(iv_z, curl_z, hn::Mul(curr_z, ii_z));
                hn::Store(curr_z, d, f4_curr[i_off_z].f);
            }

            // Handle z=numVectors-1 boundary
            const unsigned int index = op_idx_data[idx_base_xy + numVectors - 1];
            const int v_off_start = volt_base_xy;
            const int v_off_end = volt_base_xy + (numVectors - 1) * v_stride_z;
            const int i_off = curr_base_xy + (numVectors - 1) * i_stride_z;

            alignas(16) float temp_arr[4];

            // x-pol at z=numVectors-1
            temp_arr[0] = f4_volt[v_off_start + v_stride_n].f[1];
            temp_arr[1] = f4_volt[v_off_start + v_stride_n].f[2];
            temp_arr[2] = f4_volt[v_off_start + v_stride_n].f[3];
            temp_arr[3] = 0;

            auto curr = hn::Load(d, f4_curr[i_off].f);
            auto ii = hn::Load(d, ii0[index].f);
            auto iv = hn::Load(d, iv0[index].f);
            auto curl = hn::Sub(
                hn::Load(d, f4_volt[v_off_end + 2*v_stride_n].f),
                hn::Load(d, f4_volt[v_off_end + 2*v_stride_n + v_stride_y].f)
            );
            curl = hn::Sub(curl, hn::Load(d, f4_volt[v_off_end + v_stride_n].f));
            curl = hn::Add(curl, hn::Load(d, temp_arr));
            curr = hn::MulAdd(iv, curl, hn::Mul(curr, ii));
            hn::Store(curr, d, f4_curr[i_off].f);

            // y-pol at z=numVectors-1
            temp_arr[0] = f4_volt[v_off_start].f[1];
            temp_arr[1] = f4_volt[v_off_start].f[2];
            temp_arr[2] = f4_volt[v_off_start].f[3];
            temp_arr[3] = 0;

            const int i_off_y = i_off + i_stride_n;
            curr = hn::Load(d, f4_curr[i_off_y].f);
            ii = hn::Load(d, ii1[index].f);
            iv = hn::Load(d, iv1[index].f);
            curl = hn::Sub(hn::Load(d, f4_volt[v_off_end].f), hn::Load(d, temp_arr));
            curl = hn::Sub(curl, hn::Load(d, f4_volt[v_off_end + 2*v_stride_n].f));
            curl = hn::Add(curl, hn::Load(d, f4_volt[v_off_end + 2*v_stride_n + v_stride_x].f));
            curr = hn::MulAdd(iv, curl, hn::Mul(curr, ii));
            hn::Store(curr, d, f4_curr[i_off_y].f);

            // z-pol at z=numVectors-1
            const int i_off_z = i_off_y + i_stride_n;
            curr = hn::Load(d, f4_curr[i_off_z].f);
            ii = hn::Load(d, ii2[index].f);
            iv = hn::Load(d, iv2[index].f);
            curl = hn::Sub(
                hn::Load(d, f4_volt[v_off_end + v_stride_n].f),
                hn::Load(d, f4_volt[v_off_end + v_stride_n + v_stride_x].f)
            );
            curl = hn::Sub(curl, hn::Load(d, f4_volt[v_off_end].f));
            curl = hn::Add(curl, hn::Load(d, f4_volt[v_off_end + v_stride_y].f));
            curr = hn::MulAdd(iv, curl, hn::Mul(curr, ii));
            hn::Store(curr, d, f4_curr[i_off_z].f);
        }
    }
}

}  // namespace HWY_NAMESPACE
}  // namespace openEMS
HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace openEMS {

using UpdateVoltagesPtr = void (*)(f4vector*, f4vector*, const Operator_SSE_Compressed*,
    const ArrayLib::ArrayENG<f4vector>*, const ArrayLib::ArrayENG<f4vector>*,
    unsigned int, unsigned int, unsigned int, unsigned int);

using UpdateCurrentsPtr = void (*)(f4vector*, f4vector*, const Operator_SSE_Compressed*,
    const ArrayLib::ArrayENG<f4vector>*, const ArrayLib::ArrayENG<f4vector>*,
    unsigned int, unsigned int, unsigned int, unsigned int);

HWY_EXPORT(UpdateVoltagesHwy);
HWY_EXPORT(UpdateCurrentsHwy);

static UpdateVoltagesPtr g_UpdateVoltagesPtr = nullptr;
static UpdateCurrentsPtr g_UpdateCurrentsPtr = nullptr;

void InitFunctionPointers() {
    if (!g_UpdateVoltagesPtr) {
        g_UpdateVoltagesPtr = HWY_DYNAMIC_POINTER(UpdateVoltagesHwy);
        g_UpdateCurrentsPtr = HWY_DYNAMIC_POINTER(UpdateCurrentsHwy);
    }
}

void CallUpdateVoltagesHwy(
    f4vector* f4_volt, f4vector* f4_curr,
    const Operator_SSE_Compressed* Op,
    const ArrayLib::ArrayENG<f4vector>* f4_volt_ptr,
    const ArrayLib::ArrayENG<f4vector>* f4_curr_ptr,
    unsigned int numLines1, unsigned int numVectors,
    unsigned int startX, unsigned int numX)
{
    if (HWY_UNLIKELY(!g_UpdateVoltagesPtr)) {
        InitFunctionPointers();
    }
    g_UpdateVoltagesPtr(f4_volt, f4_curr, Op, f4_volt_ptr, f4_curr_ptr,
                        numLines1, numVectors, startX, numX);
}

void CallUpdateCurrentsHwy(
    f4vector* f4_curr, f4vector* f4_volt,
    const Operator_SSE_Compressed* Op,
    const ArrayLib::ArrayENG<f4vector>* f4_volt_ptr,
    const ArrayLib::ArrayENG<f4vector>* f4_curr_ptr,
    unsigned int numLines1, unsigned int numVectors,
    unsigned int startX, unsigned int numX)
{
    if (HWY_UNLIKELY(!g_UpdateCurrentsPtr)) {
        InitFunctionPointers();
    }
    g_UpdateCurrentsPtr(f4_curr, f4_volt, Op, f4_volt_ptr, f4_curr_ptr,
                        numLines1, numVectors, startX, numX);
}
}  // namespace openEMS

using std::cout;
using std::endl;

Engine_Hwy* Engine_Hwy::New(const Operator_Hwy* op, unsigned int numThreads)
{
    const auto target = hwy::SupportedTargets();
    const char* simd_name = "Unknown";
    if (target & HWY_AVX3_DL) simd_name = "AVX-512";
    else if (target & HWY_AVX3) simd_name = "AVX-512";
    else if (target & HWY_AVX2) simd_name = "AVX2";
    else if (target & HWY_SSE4) simd_name = "SSE4";
    else if (target & HWY_SSE2) simd_name = "SSE2";
    else if (target & HWY_NEON) simd_name = "NEON";

    cout << "Create FDTD engine (Highway SIMD - " << simd_name << " + multi-threading)" << endl;

    openEMS::InitFunctionPointers();

    Engine_Hwy* e = new Engine_Hwy(op);
    e->setNumThreads(numThreads);
    e->Init();
    return e;
}

Engine_Hwy::Engine_Hwy(const Operator_Hwy* op) : Engine_Multithread(op)
{
    m_Op_Hwy = op;
    m_type = SSE;
}

Engine_Hwy::~Engine_Hwy()
{
}

void Engine_Hwy::UpdateVoltages(unsigned int startX, unsigned int numX)
{
    openEMS::CallUpdateVoltagesHwy(
        f4_volt_ptr->data(), f4_curr_ptr->data(), Op,
        f4_volt_ptr, f4_curr_ptr, numLines[1], numVectors, startX, numX);
}

void Engine_Hwy::UpdateCurrents(unsigned int startX, unsigned int numX)
{
    openEMS::CallUpdateCurrentsHwy(
        f4_curr_ptr->data(), f4_volt_ptr->data(), Op,
        f4_volt_ptr, f4_curr_ptr, numLines[1], numVectors, startX, numX);
}

#endif  // HWY_ONCE
