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

// Highway SIMD library - use FixedTag<float, 4> to match f4vector
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "FDTD/engine_hwy.cpp"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

HWY_BEFORE_NAMESPACE();
namespace openEMS {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

// Use FixedTag<float, 4> to match f4vector size (4 floats = 128 bits)
using F32x4 = hn::FixedTag<float, 4>;

/**
 * @brief Update voltages using Highway SIMD with FixedTag<float, 4>
 */
void UpdateVoltagesHwy(
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

    unsigned int pos[3];
    unsigned int v_pos;
    unsigned int i_pos;
    int i_shift[3];

    int v_N_shift = f4_volt_ptr->stride(0);
    int i_N_shift = f4_curr_ptr->stride(0);
    i_shift[2] = f4_curr_ptr->stride(3);

    pos[0] = startX;
    for (unsigned int posX = 0; posX < numX; ++posX) {
        i_shift[0] = (pos[0] > 0) * f4_curr_ptr->stride(1);
        for (pos[1] = 0; pos[1] < numLines1; ++pos[1]) {
            i_shift[1] = (pos[1] > 0) * f4_curr_ptr->stride(2);

            for (pos[2] = 1; pos[2] < numVectors; ++pos[2]) {
                unsigned int index = Op->m_Op_index(pos[0], pos[1], pos[2]);
                i_pos = f4_curr_ptr->linearIndex({0, pos[0], pos[1], pos[2]});
                v_pos = f4_volt_ptr->linearIndex({0, pos[0], pos[1], pos[2]});

                // x-polarization
                auto volt_x = hn::Load(d, f4_volt[v_pos].f);
                auto vv_x = hn::Load(d, Op->f4_vv_Compressed[0][index].f);
                auto vi_x = hn::Load(d, Op->f4_vi_Compressed[0][index].f);

                auto curl = hn::Sub(
                    hn::Load(d, f4_curr[i_pos + 2*i_N_shift].f),
                    hn::Load(d, f4_curr[i_pos + 2*i_N_shift - i_shift[1]].f)
                );
                curl = hn::Sub(curl, hn::Load(d, f4_curr[i_pos + i_N_shift].f));
                curl = hn::Add(curl, hn::Load(d, f4_curr[i_pos + i_N_shift - i_shift[2]].f));

                volt_x = hn::MulAdd(vi_x, curl, hn::Mul(volt_x, vv_x));
                hn::Store(volt_x, d, f4_volt[v_pos].f);

                // y-polarization
                v_pos += v_N_shift;
                auto volt_y = hn::Load(d, f4_volt[v_pos].f);
                auto vv_y = hn::Load(d, Op->f4_vv_Compressed[1][index].f);
                auto vi_y = hn::Load(d, Op->f4_vi_Compressed[1][index].f);

                curl = hn::Sub(
                    hn::Load(d, f4_curr[i_pos].f),
                    hn::Load(d, f4_curr[i_pos - i_shift[2]].f)
                );
                curl = hn::Sub(curl, hn::Load(d, f4_curr[i_pos + 2*i_N_shift].f));
                curl = hn::Add(curl, hn::Load(d, f4_curr[i_pos + 2*i_N_shift - i_shift[0]].f));

                volt_y = hn::MulAdd(vi_y, curl, hn::Mul(volt_y, vv_y));
                hn::Store(volt_y, d, f4_volt[v_pos].f);

                // z-polarization
                v_pos += v_N_shift;
                auto volt_z = hn::Load(d, f4_volt[v_pos].f);
                auto vv_z = hn::Load(d, Op->f4_vv_Compressed[2][index].f);
                auto vi_z = hn::Load(d, Op->f4_vi_Compressed[2][index].f);

                curl = hn::Sub(
                    hn::Load(d, f4_curr[i_pos + i_N_shift].f),
                    hn::Load(d, f4_curr[i_pos + i_N_shift - i_shift[0]].f)
                );
                curl = hn::Sub(curl, hn::Load(d, f4_curr[i_pos].f));
                curl = hn::Add(curl, hn::Load(d, f4_curr[i_pos - i_shift[1]].f));

                volt_z = hn::MulAdd(vi_z, curl, hn::Mul(volt_z, vv_z));
                hn::Store(volt_z, d, f4_volt[v_pos].f);
            }

            // Handle z=0 boundary
            unsigned int i_pos_z_start = f4_curr_ptr->linearIndex({0, pos[0], pos[1], 0});
            unsigned int i_pos_z_end = f4_curr_ptr->linearIndex({0, pos[0], pos[1], numVectors-1});
            unsigned int index = Op->m_Op_index(pos[0], pos[1], 0);

            // Shift for boundary: insert 0 at position 0, shift others right
            alignas(16) float temp_arr[4];
            temp_arr[0] = 0;
            temp_arr[1] = f4_curr[i_pos_z_end + i_N_shift].f[0];
            temp_arr[2] = f4_curr[i_pos_z_end + i_N_shift].f[1];
            temp_arr[3] = f4_curr[i_pos_z_end + i_N_shift].f[2];

            v_pos = f4_volt_ptr->linearIndex({0, pos[0], pos[1], 0});
            auto volt = hn::Load(d, f4_volt[v_pos].f);
            auto vv = hn::Load(d, Op->f4_vv_Compressed[0][index].f);
            auto vi = hn::Load(d, Op->f4_vi_Compressed[0][index].f);
            auto curl = hn::Sub(
                hn::Load(d, f4_curr[i_pos_z_start + i_N_shift*2].f),
                hn::Load(d, f4_curr[i_pos_z_start + i_N_shift*2 - i_shift[1]].f)
            );
            curl = hn::Sub(curl, hn::Load(d, f4_curr[i_pos_z_start + i_N_shift].f));
            curl = hn::Add(curl, hn::Load(d, temp_arr));
            volt = hn::MulAdd(vi, curl, hn::Mul(volt, vv));
            hn::Store(volt, d, f4_volt[v_pos].f);

            temp_arr[0] = 0;
            temp_arr[1] = f4_curr[i_pos_z_end].f[0];
            temp_arr[2] = f4_curr[i_pos_z_end].f[1];
            temp_arr[3] = f4_curr[i_pos_z_end].f[2];

            v_pos += v_N_shift;
            volt = hn::Load(d, f4_volt[v_pos].f);
            vv = hn::Load(d, Op->f4_vv_Compressed[1][index].f);
            vi = hn::Load(d, Op->f4_vi_Compressed[1][index].f);
            curl = hn::Sub(hn::Load(d, f4_curr[i_pos_z_start].f), hn::Load(d, temp_arr));
            curl = hn::Sub(curl, hn::Load(d, f4_curr[i_pos_z_start + i_N_shift*2].f));
            curl = hn::Add(curl, hn::Load(d, f4_curr[i_pos_z_start + i_N_shift*2 - i_shift[0]].f));
            volt = hn::MulAdd(vi, curl, hn::Mul(volt, vv));
            hn::Store(volt, d, f4_volt[v_pos].f);

            v_pos += v_N_shift;
            volt = hn::Load(d, f4_volt[v_pos].f);
            vv = hn::Load(d, Op->f4_vv_Compressed[2][index].f);
            vi = hn::Load(d, Op->f4_vi_Compressed[2][index].f);
            curl = hn::Sub(
                hn::Load(d, f4_curr[i_pos_z_start + i_N_shift].f),
                hn::Load(d, f4_curr[i_pos_z_start + i_N_shift - i_shift[0]].f)
            );
            curl = hn::Sub(curl, hn::Load(d, f4_curr[i_pos_z_start].f));
            curl = hn::Add(curl, hn::Load(d, f4_curr[i_pos_z_start - i_shift[1]].f));
            volt = hn::MulAdd(vi, curl, hn::Mul(volt, vv));
            hn::Store(volt, d, f4_volt[v_pos].f);
        }
        ++pos[0];
    }
}

/**
 * @brief Update currents using Highway SIMD with FixedTag<float, 4>
 */
void UpdateCurrentsHwy(
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

    unsigned int v_pos;
    unsigned int i_pos;
    int v_shift[3];
    unsigned int pos[3];

    int v_N_shift = f4_volt_ptr->stride(0);
    int i_N_shift = f4_curr_ptr->stride(0);

    for (unsigned int n = 0; n < 3; ++n)
        v_shift[n] = f4_volt_ptr->stride(n + 1);

    pos[0] = startX;
    for (unsigned int posX = 0; posX < numX; ++posX) {
        for (pos[1] = 0; pos[1] < numLines1 - 1; ++pos[1]) {
            for (pos[2] = 0; pos[2] < numVectors - 1; ++pos[2]) {
                unsigned int index = Op->m_Op_index(pos[0], pos[1], pos[2]);
                i_pos = f4_curr_ptr->linearIndex({0, pos[0], pos[1], pos[2]});
                v_pos = f4_volt_ptr->linearIndex({0, pos[0], pos[1], pos[2]});

                // x-pol
                auto curr = hn::Load(d, f4_curr[i_pos].f);
                auto ii = hn::Load(d, Op->f4_ii_Compressed[0][index].f);
                auto iv = hn::Load(d, Op->f4_iv_Compressed[0][index].f);
                auto curl = hn::Sub(
                    hn::Load(d, f4_volt[v_pos + 2*v_N_shift].f),
                    hn::Load(d, f4_volt[v_pos + 2*v_N_shift + v_shift[1]].f)
                );
                curl = hn::Sub(curl, hn::Load(d, f4_volt[v_pos + v_N_shift].f));
                curl = hn::Add(curl, hn::Load(d, f4_volt[v_pos + v_N_shift + v_shift[2]].f));
                curr = hn::MulAdd(iv, curl, hn::Mul(curr, ii));
                hn::Store(curr, d, f4_curr[i_pos].f);

                // y-pol
                i_pos += i_N_shift;
                curr = hn::Load(d, f4_curr[i_pos].f);
                ii = hn::Load(d, Op->f4_ii_Compressed[1][index].f);
                iv = hn::Load(d, Op->f4_iv_Compressed[1][index].f);
                curl = hn::Sub(
                    hn::Load(d, f4_volt[v_pos].f),
                    hn::Load(d, f4_volt[v_pos + v_shift[2]].f)
                );
                curl = hn::Sub(curl, hn::Load(d, f4_volt[v_pos + 2*v_N_shift].f));
                curl = hn::Add(curl, hn::Load(d, f4_volt[v_pos + 2*v_N_shift + v_shift[0]].f));
                curr = hn::MulAdd(iv, curl, hn::Mul(curr, ii));
                hn::Store(curr, d, f4_curr[i_pos].f);

                // z-pol
                i_pos += i_N_shift;
                curr = hn::Load(d, f4_curr[i_pos].f);
                ii = hn::Load(d, Op->f4_ii_Compressed[2][index].f);
                iv = hn::Load(d, Op->f4_iv_Compressed[2][index].f);
                curl = hn::Sub(
                    hn::Load(d, f4_volt[v_pos + v_N_shift].f),
                    hn::Load(d, f4_volt[v_pos + v_N_shift + v_shift[0]].f)
                );
                curl = hn::Sub(curl, hn::Load(d, f4_volt[v_pos].f));
                curl = hn::Add(curl, hn::Load(d, f4_volt[v_pos + v_shift[1]].f));
                curr = hn::MulAdd(iv, curl, hn::Mul(curr, ii));
                hn::Store(curr, d, f4_curr[i_pos].f);
            }

            // Handle z=numVectors-1 boundary
            unsigned int v_pos_z_start = f4_volt_ptr->linearIndex({0, pos[0], pos[1], 0});
            unsigned int v_pos_z_end = f4_volt_ptr->linearIndex({0, pos[0], pos[1], numVectors-1});
            unsigned int index = Op->m_Op_index(pos[0], pos[1], numVectors-1);

            alignas(16) float temp_arr[4];
            temp_arr[0] = f4_volt[v_pos_z_start + v_N_shift].f[1];
            temp_arr[1] = f4_volt[v_pos_z_start + v_N_shift].f[2];
            temp_arr[2] = f4_volt[v_pos_z_start + v_N_shift].f[3];
            temp_arr[3] = 0;

            i_pos = f4_curr_ptr->linearIndex({0, pos[0], pos[1], numVectors-1});
            auto curr = hn::Load(d, f4_curr[i_pos].f);
            auto ii = hn::Load(d, Op->f4_ii_Compressed[0][index].f);
            auto iv = hn::Load(d, Op->f4_iv_Compressed[0][index].f);
            auto curl = hn::Sub(
                hn::Load(d, f4_volt[v_pos_z_end + 2*v_N_shift].f),
                hn::Load(d, f4_volt[v_pos_z_end + 2*v_N_shift + v_shift[1]].f)
            );
            curl = hn::Sub(curl, hn::Load(d, f4_volt[v_pos_z_end + v_N_shift].f));
            curl = hn::Add(curl, hn::Load(d, temp_arr));
            curr = hn::MulAdd(iv, curl, hn::Mul(curr, ii));
            hn::Store(curr, d, f4_curr[i_pos].f);

            temp_arr[0] = f4_volt[v_pos_z_start].f[1];
            temp_arr[1] = f4_volt[v_pos_z_start].f[2];
            temp_arr[2] = f4_volt[v_pos_z_start].f[3];
            temp_arr[3] = 0;

            i_pos += i_N_shift;
            curr = hn::Load(d, f4_curr[i_pos].f);
            ii = hn::Load(d, Op->f4_ii_Compressed[1][index].f);
            iv = hn::Load(d, Op->f4_iv_Compressed[1][index].f);
            curl = hn::Sub(hn::Load(d, f4_volt[v_pos_z_end].f), hn::Load(d, temp_arr));
            curl = hn::Sub(curl, hn::Load(d, f4_volt[v_pos_z_end + 2*v_N_shift].f));
            curl = hn::Add(curl, hn::Load(d, f4_volt[v_pos_z_end + 2*v_N_shift + v_shift[0]].f));
            curr = hn::MulAdd(iv, curl, hn::Mul(curr, ii));
            hn::Store(curr, d, f4_curr[i_pos].f);

            i_pos += i_N_shift;
            curr = hn::Load(d, f4_curr[i_pos].f);
            ii = hn::Load(d, Op->f4_ii_Compressed[2][index].f);
            iv = hn::Load(d, Op->f4_iv_Compressed[2][index].f);
            curl = hn::Sub(
                hn::Load(d, f4_volt[v_pos_z_end + v_N_shift].f),
                hn::Load(d, f4_volt[v_pos_z_end + v_N_shift + v_shift[0]].f)
            );
            curl = hn::Sub(curl, hn::Load(d, f4_volt[v_pos_z_end].f));
            curl = hn::Add(curl, hn::Load(d, f4_volt[v_pos_z_end + v_shift[1]].f));
            curr = hn::MulAdd(iv, curl, hn::Mul(curr, ii));
            hn::Store(curr, d, f4_curr[i_pos].f);
        }
        ++pos[0];
    }
}

}  // namespace HWY_NAMESPACE
}  // namespace openEMS
HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace openEMS {
HWY_EXPORT(UpdateVoltagesHwy);
HWY_EXPORT(UpdateCurrentsHwy);

void CallUpdateVoltagesHwy(
    f4vector* f4_volt, f4vector* f4_curr,
    const Operator_SSE_Compressed* Op,
    const ArrayLib::ArrayENG<f4vector>* f4_volt_ptr,
    const ArrayLib::ArrayENG<f4vector>* f4_curr_ptr,
    unsigned int numLines1, unsigned int numVectors,
    unsigned int startX, unsigned int numX)
{
    HWY_DYNAMIC_DISPATCH(UpdateVoltagesHwy)(
        f4_volt, f4_curr, Op, f4_volt_ptr, f4_curr_ptr,
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
    HWY_DYNAMIC_DISPATCH(UpdateCurrentsHwy)(
        f4_curr, f4_volt, Op, f4_volt_ptr, f4_curr_ptr,
        numLines1, numVectors, startX, numX);
}
}  // namespace openEMS

using std::cout;
using std::endl;

// ============================================================================
// Engine_Hwy Implementation - inherits threading from Engine_Multithread
// ============================================================================

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
