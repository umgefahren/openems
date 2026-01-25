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
#include "operator_sse_compressed.h"

// Highway SIMD library
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "FDTD/engine_hwy.cpp"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

HWY_BEFORE_NAMESPACE();
namespace openEMS {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

/**
 * @brief Update voltages using Highway SIMD for the inner loop
 *
 * This processes multiple f4vectors at once using wider SIMD (AVX2/AVX-512).
 * The key insight is that adjacent f4vectors in memory can be processed together.
 */
void UpdateVoltagesHwy(
    f4vector* __restrict f4_volt,
    const f4vector* __restrict f4_curr,
    const Operator_SSE_Compressed* Op,
    const ArrayLib::ArrayENG<f4vector>* f4_volt_ptr,
    const ArrayLib::ArrayENG<f4vector>* f4_curr_ptr,
    unsigned int numLines1,
    unsigned int numVectors,
    unsigned int startX,
    unsigned int numX)
{
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);

    // Array strides
    const int v_N_shift = f4_volt_ptr->stride(0);
    const int i_N_shift = f4_curr_ptr->stride(0);
    const int i_shift_z = f4_curr_ptr->stride(3);

    unsigned int pos0 = startX;

    for (unsigned int posX = 0; posX < numX; ++posX) {
        const int i_shift_x = (pos0 > 0) * f4_curr_ptr->stride(1);

        for (unsigned int pos1 = 0; pos1 < numLines1; ++pos1) {
            const int i_shift_y = (pos1 > 0) * f4_curr_ptr->stride(2);

            // Process pos[2] = 1 to numVectors-1 (main loop, can use wide SIMD)
            // Each f4vector is 4 floats, so we process multiple f4vectors at once
            const size_t floats_per_f4 = 4;
            const size_t f4_per_vec = N / floats_per_f4;

            unsigned int pos2 = 1;

            // Process multiple f4vectors at once when possible
            if (f4_per_vec >= 1 && numVectors > 1) {
                // For AVX-512: N=16, f4_per_vec=4, process 4 f4vectors at once
                // For AVX2: N=8, f4_per_vec=2, process 2 f4vectors at once
                // For SSE: N=4, f4_per_vec=1, process 1 f4vector at a time

                for (; pos2 + f4_per_vec <= numVectors; pos2 += f4_per_vec) {
                    const unsigned int index_base = Op->m_Op_index(pos0, pos1, pos2);
                    const unsigned int i_pos_base = f4_curr_ptr->linearIndex({0, pos0, pos1, pos2});
                    const unsigned int v_pos_base = f4_volt_ptr->linearIndex({0, pos0, pos1, pos2});

                    // Process all polarizations for this batch of positions
                    for (size_t f4_idx = 0; f4_idx < f4_per_vec; ++f4_idx) {
                        const unsigned int index = Op->m_Op_index(pos0, pos1, pos2 + f4_idx);
                        const unsigned int i_pos = i_pos_base + f4_idx * i_shift_z;
                        unsigned int v_pos = v_pos_base + f4_idx * f4_volt_ptr->stride(3);

                        // x-polarization
                        auto vv_x = hn::LoadU(d, Op->f4_vv_Compressed[0][index].f);
                        auto vi_x = hn::LoadU(d, Op->f4_vi_Compressed[0][index].f);
                        auto volt_x = hn::LoadU(d, f4_volt[v_pos].f);

                        auto curr_z_here = hn::LoadU(d, f4_curr[i_pos + 2*i_N_shift].f);
                        auto curr_z_prev_y = hn::LoadU(d, f4_curr[i_pos + 2*i_N_shift - i_shift_y].f);
                        auto curr_y_here = hn::LoadU(d, f4_curr[i_pos + i_N_shift].f);
                        auto curr_y_prev_z = hn::LoadU(d, f4_curr[i_pos + i_N_shift - i_shift_z].f);

                        auto curl_x = hn::Sub(curr_z_here, curr_z_prev_y);
                        curl_x = hn::Sub(curl_x, curr_y_here);
                        curl_x = hn::Add(curl_x, curr_y_prev_z);

                        volt_x = hn::Mul(volt_x, vv_x);
                        volt_x = hn::MulAdd(vi_x, curl_x, volt_x);
                        hn::StoreU(volt_x, d, f4_volt[v_pos].f);

                        // y-polarization
                        v_pos += v_N_shift;
                        auto vv_y = hn::LoadU(d, Op->f4_vv_Compressed[1][index].f);
                        auto vi_y = hn::LoadU(d, Op->f4_vi_Compressed[1][index].f);
                        auto volt_y = hn::LoadU(d, f4_volt[v_pos].f);

                        auto curr_x_here = hn::LoadU(d, f4_curr[i_pos].f);
                        auto curr_x_prev_z = hn::LoadU(d, f4_curr[i_pos - i_shift_z].f);
                        auto curr_z_here2 = hn::LoadU(d, f4_curr[i_pos + 2*i_N_shift].f);
                        auto curr_z_prev_x = hn::LoadU(d, f4_curr[i_pos + 2*i_N_shift - i_shift_x].f);

                        auto curl_y = hn::Sub(curr_x_here, curr_x_prev_z);
                        curl_y = hn::Sub(curl_y, curr_z_here2);
                        curl_y = hn::Add(curl_y, curr_z_prev_x);

                        volt_y = hn::Mul(volt_y, vv_y);
                        volt_y = hn::MulAdd(vi_y, curl_y, volt_y);
                        hn::StoreU(volt_y, d, f4_volt[v_pos].f);

                        // z-polarization
                        v_pos += v_N_shift;
                        auto vv_z = hn::LoadU(d, Op->f4_vv_Compressed[2][index].f);
                        auto vi_z = hn::LoadU(d, Op->f4_vi_Compressed[2][index].f);
                        auto volt_z = hn::LoadU(d, f4_volt[v_pos].f);

                        auto curr_y_here2 = hn::LoadU(d, f4_curr[i_pos + i_N_shift].f);
                        auto curr_y_prev_x = hn::LoadU(d, f4_curr[i_pos + i_N_shift - i_shift_x].f);
                        auto curr_x_here2 = hn::LoadU(d, f4_curr[i_pos].f);
                        auto curr_x_prev_y = hn::LoadU(d, f4_curr[i_pos - i_shift_y].f);

                        auto curl_z = hn::Sub(curr_y_here2, curr_y_prev_x);
                        curl_z = hn::Sub(curl_z, curr_x_here2);
                        curl_z = hn::Add(curl_z, curr_x_prev_y);

                        volt_z = hn::Mul(volt_z, vv_z);
                        volt_z = hn::MulAdd(vi_z, curl_z, volt_z);
                        hn::StoreU(volt_z, d, f4_volt[v_pos].f);
                    }
                }
            }

            // Handle remaining positions with scalar fallback
            for (; pos2 < numVectors; ++pos2) {
                const unsigned int index = Op->m_Op_index(pos0, pos1, pos2);
                const unsigned int i_pos = f4_curr_ptr->linearIndex({0, pos0, pos1, pos2});
                unsigned int v_pos = f4_volt_ptr->linearIndex({0, pos0, pos1, pos2});

                // x-polarization (scalar)
                f4_volt[v_pos].v *= Op->f4_vv_Compressed[0][index].v;
                f4_volt[v_pos].v +=
                    Op->f4_vi_Compressed[0][index].v * (
                        f4_curr[i_pos + 2*i_N_shift].v -
                        f4_curr[i_pos + 2*i_N_shift - i_shift_y].v -
                        f4_curr[i_pos + i_N_shift].v +
                        f4_curr[i_pos + i_N_shift - i_shift_z].v
                    );

                // y-polarization
                v_pos += v_N_shift;
                f4_volt[v_pos].v *= Op->f4_vv_Compressed[1][index].v;
                f4_volt[v_pos].v +=
                    Op->f4_vi_Compressed[1][index].v * (
                        f4_curr[i_pos].v -
                        f4_curr[i_pos - i_shift_z].v -
                        f4_curr[i_pos + 2*i_N_shift].v +
                        f4_curr[i_pos + 2*i_N_shift - i_shift_x].v
                    );

                // z-polarization
                v_pos += v_N_shift;
                f4_volt[v_pos].v *= Op->f4_vv_Compressed[2][index].v;
                f4_volt[v_pos].v +=
                    Op->f4_vi_Compressed[2][index].v * (
                        f4_curr[i_pos + i_N_shift].v -
                        f4_curr[i_pos + i_N_shift - i_shift_x].v -
                        f4_curr[i_pos].v +
                        f4_curr[i_pos - i_shift_y].v
                    );
            }

            // Handle z=0 boundary (special case with wrap-around)
            const unsigned int i_pos_z_start = f4_curr_ptr->linearIndex({0, pos0, pos1, 0});
            const unsigned int i_pos_z_end = f4_curr_ptr->linearIndex({0, pos0, pos1, numVectors-1});
            const unsigned int index0 = Op->m_Op_index(pos0, pos1, 0);

            f4vector temp;

            // x-polarization at z=0
            temp.f[0] = 0;
            temp.f[1] = f4_curr[i_pos_z_end + i_N_shift].f[0];
            temp.f[2] = f4_curr[i_pos_z_end + i_N_shift].f[1];
            temp.f[3] = f4_curr[i_pos_z_end + i_N_shift].f[2];

            unsigned int v_pos0 = f4_volt_ptr->linearIndex({0, pos0, pos1, 0});
            f4_volt[v_pos0].v *= Op->f4_vv_Compressed[0][index0].v;
            f4_volt[v_pos0].v +=
                Op->f4_vi_Compressed[0][index0].v * (
                    f4_curr[i_pos_z_start + i_N_shift*2].v -
                    f4_curr[i_pos_z_start + i_N_shift*2 - i_shift_y].v -
                    f4_curr[i_pos_z_start + i_N_shift].v +
                    temp.v
                );

            // y-polarization at z=0
            temp.f[0] = 0;
            temp.f[1] = f4_curr[i_pos_z_end].f[0];
            temp.f[2] = f4_curr[i_pos_z_end].f[1];
            temp.f[3] = f4_curr[i_pos_z_end].f[2];

            v_pos0 += v_N_shift;
            f4_volt[v_pos0].v *= Op->f4_vv_Compressed[1][index0].v;
            f4_volt[v_pos0].v +=
                Op->f4_vi_Compressed[1][index0].v * (
                    f4_curr[i_pos_z_start].v -
                    temp.v -
                    f4_curr[i_pos_z_start + i_N_shift*2].v +
                    f4_curr[i_pos_z_start + i_N_shift*2 - i_shift_x].v
                );

            // z-polarization at z=0
            v_pos0 += v_N_shift;
            f4_volt[v_pos0].v *= Op->f4_vv_Compressed[2][index0].v;
            f4_volt[v_pos0].v +=
                Op->f4_vi_Compressed[2][index0].v * (
                    f4_curr[i_pos_z_start + i_N_shift].v -
                    f4_curr[i_pos_z_start + i_N_shift - i_shift_x].v -
                    f4_curr[i_pos_z_start].v +
                    f4_curr[i_pos_z_start - i_shift_y].v
                );
        }
        ++pos0;
    }
}

/**
 * @brief Update currents using Highway SIMD for the inner loop
 */
void UpdateCurrentsHwy(
    f4vector* __restrict f4_curr,
    const f4vector* __restrict f4_volt,
    const Operator_SSE_Compressed* Op,
    const ArrayLib::ArrayENG<f4vector>* f4_volt_ptr,
    const ArrayLib::ArrayENG<f4vector>* f4_curr_ptr,
    unsigned int numLines1,
    unsigned int numVectors,
    unsigned int startX,
    unsigned int numX)
{
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);

    // Array strides
    const int v_N_shift = f4_volt_ptr->stride(0);
    const int i_N_shift = f4_curr_ptr->stride(0);

    int v_shift[3];
    for (unsigned int n = 0; n < 3; ++n)
        v_shift[n] = f4_volt_ptr->stride(n + 1);

    unsigned int pos0 = startX;

    for (unsigned int posX = 0; posX < numX; ++posX) {
        for (unsigned int pos1 = 0; pos1 < numLines1 - 1; ++pos1) {
            const size_t floats_per_f4 = 4;
            const size_t f4_per_vec = N / floats_per_f4;

            unsigned int pos2 = 0;

            // Process multiple f4vectors at once when possible
            if (f4_per_vec >= 1 && numVectors > 1) {
                for (; pos2 + f4_per_vec <= numVectors - 1; pos2 += f4_per_vec) {
                    for (size_t f4_idx = 0; f4_idx < f4_per_vec; ++f4_idx) {
                        const unsigned int index = Op->m_Op_index(pos0, pos1, pos2 + f4_idx);
                        const unsigned int i_pos_base = f4_curr_ptr->linearIndex({0, pos0, pos1, pos2 + f4_idx});
                        const unsigned int v_pos = f4_volt_ptr->linearIndex({0, pos0, pos1, pos2 + f4_idx});

                        // x-polarization
                        auto ii_x = hn::LoadU(d, Op->f4_ii_Compressed[0][index].f);
                        auto iv_x = hn::LoadU(d, Op->f4_iv_Compressed[0][index].f);
                        auto curr_x = hn::LoadU(d, f4_curr[i_pos_base].f);

                        auto volt_z_here = hn::LoadU(d, f4_volt[v_pos + 2*v_N_shift].f);
                        auto volt_z_next_y = hn::LoadU(d, f4_volt[v_pos + 2*v_N_shift + v_shift[1]].f);
                        auto volt_y_here = hn::LoadU(d, f4_volt[v_pos + v_N_shift].f);
                        auto volt_y_next_z = hn::LoadU(d, f4_volt[v_pos + v_N_shift + v_shift[2]].f);

                        auto curl_x = hn::Sub(volt_z_here, volt_z_next_y);
                        curl_x = hn::Sub(curl_x, volt_y_here);
                        curl_x = hn::Add(curl_x, volt_y_next_z);

                        curr_x = hn::Mul(curr_x, ii_x);
                        curr_x = hn::MulAdd(iv_x, curl_x, curr_x);
                        hn::StoreU(curr_x, d, f4_curr[i_pos_base].f);

                        // y-polarization
                        unsigned int i_pos_y = i_pos_base + i_N_shift;
                        auto ii_y = hn::LoadU(d, Op->f4_ii_Compressed[1][index].f);
                        auto iv_y = hn::LoadU(d, Op->f4_iv_Compressed[1][index].f);
                        auto curr_y = hn::LoadU(d, f4_curr[i_pos_y].f);

                        auto volt_x_here = hn::LoadU(d, f4_volt[v_pos].f);
                        auto volt_x_next_z = hn::LoadU(d, f4_volt[v_pos + v_shift[2]].f);
                        auto volt_z_here2 = hn::LoadU(d, f4_volt[v_pos + 2*v_N_shift].f);
                        auto volt_z_next_x = hn::LoadU(d, f4_volt[v_pos + 2*v_N_shift + v_shift[0]].f);

                        auto curl_y = hn::Sub(volt_x_here, volt_x_next_z);
                        curl_y = hn::Sub(curl_y, volt_z_here2);
                        curl_y = hn::Add(curl_y, volt_z_next_x);

                        curr_y = hn::Mul(curr_y, ii_y);
                        curr_y = hn::MulAdd(iv_y, curl_y, curr_y);
                        hn::StoreU(curr_y, d, f4_curr[i_pos_y].f);

                        // z-polarization
                        unsigned int i_pos_z = i_pos_base + 2*i_N_shift;
                        auto ii_z = hn::LoadU(d, Op->f4_ii_Compressed[2][index].f);
                        auto iv_z = hn::LoadU(d, Op->f4_iv_Compressed[2][index].f);
                        auto curr_z = hn::LoadU(d, f4_curr[i_pos_z].f);

                        auto volt_y_here2 = hn::LoadU(d, f4_volt[v_pos + v_N_shift].f);
                        auto volt_y_next_x = hn::LoadU(d, f4_volt[v_pos + v_N_shift + v_shift[0]].f);
                        auto volt_x_here2 = hn::LoadU(d, f4_volt[v_pos].f);
                        auto volt_x_next_y = hn::LoadU(d, f4_volt[v_pos + v_shift[1]].f);

                        auto curl_z = hn::Sub(volt_y_here2, volt_y_next_x);
                        curl_z = hn::Sub(curl_z, volt_x_here2);
                        curl_z = hn::Add(curl_z, volt_x_next_y);

                        curr_z = hn::Mul(curr_z, ii_z);
                        curr_z = hn::MulAdd(iv_z, curl_z, curr_z);
                        hn::StoreU(curr_z, d, f4_curr[i_pos_z].f);
                    }
                }
            }

            // Handle remaining positions with scalar fallback
            for (; pos2 < numVectors - 1; ++pos2) {
                const unsigned int index = Op->m_Op_index(pos0, pos1, pos2);
                unsigned int i_pos = f4_curr_ptr->linearIndex({0, pos0, pos1, pos2});
                const unsigned int v_pos = f4_volt_ptr->linearIndex({0, pos0, pos1, pos2});

                // x-pol
                f4_curr[i_pos].v *= Op->f4_ii_Compressed[0][index].v;
                f4_curr[i_pos].v +=
                    Op->f4_iv_Compressed[0][index].v * (
                        f4_volt[v_pos + 2*v_N_shift].v -
                        f4_volt[v_pos + 2*v_N_shift + v_shift[1]].v -
                        f4_volt[v_pos + v_N_shift].v +
                        f4_volt[v_pos + v_N_shift + v_shift[2]].v
                    );

                // y-pol
                i_pos += i_N_shift;
                f4_curr[i_pos].v *= Op->f4_ii_Compressed[1][index].v;
                f4_curr[i_pos].v +=
                    Op->f4_iv_Compressed[1][index].v * (
                        f4_volt[v_pos].v -
                        f4_volt[v_pos + v_shift[2]].v -
                        f4_volt[v_pos + 2*v_N_shift].v +
                        f4_volt[v_pos + 2*v_N_shift + v_shift[0]].v
                    );

                // z-pol
                i_pos += i_N_shift;
                f4_curr[i_pos].v *= Op->f4_ii_Compressed[2][index].v;
                f4_curr[i_pos].v +=
                    Op->f4_iv_Compressed[2][index].v * (
                        f4_volt[v_pos + v_N_shift].v -
                        f4_volt[v_pos + v_N_shift + v_shift[0]].v -
                        f4_volt[v_pos].v +
                        f4_volt[v_pos + v_shift[1]].v
                    );
            }

            // Handle z = numVectors-1 boundary (special case with wrap-around)
            const unsigned int v_pos_z_start = f4_volt_ptr->linearIndex({0, pos0, pos1, 0});
            const unsigned int v_pos_z_end = f4_volt_ptr->linearIndex({0, pos0, pos1, numVectors-1});
            const unsigned int index_end = Op->m_Op_index(pos0, pos1, numVectors-1);

            f4vector temp;

            // x-pol at z=numVectors-1
            temp.f[0] = f4_volt[v_pos_z_start + v_N_shift].f[1];
            temp.f[1] = f4_volt[v_pos_z_start + v_N_shift].f[2];
            temp.f[2] = f4_volt[v_pos_z_start + v_N_shift].f[3];
            temp.f[3] = 0;

            unsigned int i_pos_end = f4_curr_ptr->linearIndex({0, pos0, pos1, numVectors-1});
            f4_curr[i_pos_end].v *= Op->f4_ii_Compressed[0][index_end].v;
            f4_curr[i_pos_end].v +=
                Op->f4_iv_Compressed[0][index_end].v * (
                    f4_volt[v_pos_z_end + 2*v_N_shift].v -
                    f4_volt[v_pos_z_end + 2*v_N_shift + v_shift[1]].v -
                    f4_volt[v_pos_z_end + v_N_shift].v +
                    temp.v
                );

            // y-pol at z=numVectors-1
            temp.f[0] = f4_volt[v_pos_z_start].f[1];
            temp.f[1] = f4_volt[v_pos_z_start].f[2];
            temp.f[2] = f4_volt[v_pos_z_start].f[3];
            temp.f[3] = 0;

            i_pos_end += i_N_shift;
            f4_curr[i_pos_end].v *= Op->f4_ii_Compressed[1][index_end].v;
            f4_curr[i_pos_end].v +=
                Op->f4_iv_Compressed[1][index_end].v * (
                    f4_volt[v_pos_z_end].v -
                    temp.v -
                    f4_volt[v_pos_z_end + 2*v_N_shift].v +
                    f4_volt[v_pos_z_end + 2*v_N_shift + v_shift[0]].v
                );

            // z-pol at z=numVectors-1
            i_pos_end += i_N_shift;
            f4_curr[i_pos_end].v *= Op->f4_ii_Compressed[2][index_end].v;
            f4_curr[i_pos_end].v +=
                Op->f4_iv_Compressed[2][index_end].v * (
                    f4_volt[v_pos_z_end + v_N_shift].v -
                    f4_volt[v_pos_z_end + v_N_shift + v_shift[0]].v -
                    f4_volt[v_pos_z_end].v +
                    f4_volt[v_pos_z_end + v_shift[1]].v
                );
        }
        ++pos0;
    }
}

}  // namespace HWY_NAMESPACE
}  // namespace openEMS
HWY_AFTER_NAMESPACE();

// Generate implementations for all enabled targets
#if HWY_ONCE

namespace openEMS {

// Dynamic dispatch tables
HWY_EXPORT(UpdateVoltagesHwy);
HWY_EXPORT(UpdateCurrentsHwy);

// Wrapper functions that call the best available implementation
void CallUpdateVoltagesHwy(
    f4vector* f4_volt,
    const f4vector* f4_curr,
    const Operator_SSE_Compressed* Op,
    const ArrayLib::ArrayENG<f4vector>* f4_volt_ptr,
    const ArrayLib::ArrayENG<f4vector>* f4_curr_ptr,
    unsigned int numLines1,
    unsigned int numVectors,
    unsigned int startX,
    unsigned int numX)
{
    HWY_DYNAMIC_DISPATCH(UpdateVoltagesHwy)(
        f4_volt, f4_curr, Op, f4_volt_ptr, f4_curr_ptr,
        numLines1, numVectors, startX, numX);
}

void CallUpdateCurrentsHwy(
    f4vector* f4_curr,
    const f4vector* f4_volt,
    const Operator_SSE_Compressed* Op,
    const ArrayLib::ArrayENG<f4vector>* f4_volt_ptr,
    const ArrayLib::ArrayENG<f4vector>* f4_curr_ptr,
    unsigned int numLines1,
    unsigned int numVectors,
    unsigned int startX,
    unsigned int numX)
{
    HWY_DYNAMIC_DISPATCH(UpdateCurrentsHwy)(
        f4_curr, f4_volt, Op, f4_volt_ptr, f4_curr_ptr,
        numLines1, numVectors, startX, numX);
}

}  // namespace openEMS

using std::cout;
using std::endl;

Engine_Hwy* Engine_Hwy::New(const Operator_SSE_Compressed* op)
{
    cout << "Create FDTD engine (Highway SIMD - auto-selecting best available: ";

    // Print which SIMD target will be used
    const auto target = hwy::SupportedTargets();
    if (target & HWY_AVX3_DL) cout << "AVX-512 DL";
    else if (target & HWY_AVX3) cout << "AVX-512";
    else if (target & HWY_AVX2) cout << "AVX2";
    else if (target & HWY_SSE4) cout << "SSE4";
    else if (target & HWY_SSE2) cout << "SSE2";
    else cout << "Scalar";
    cout << ")" << endl;

    Engine_Hwy* e = new Engine_Hwy(op);
    e->Init();
    return e;
}

Engine_Hwy::Engine_Hwy(const Operator_SSE_Compressed* op) : Engine_SSE_Compressed(op)
{
    m_type = SSE;  // Compatible with SSE type for interface purposes
}

Engine_Hwy::~Engine_Hwy()
{
}

void Engine_Hwy::UpdateVoltages(unsigned int startX, unsigned int numX)
{
    openEMS::CallUpdateVoltagesHwy(
        f4_volt_ptr->data(),
        f4_curr_ptr->data(),
        Op,
        f4_volt_ptr,
        f4_curr_ptr,
        numLines[1],
        numVectors,
        startX,
        numX);
}

void Engine_Hwy::UpdateCurrents(unsigned int startX, unsigned int numX)
{
    openEMS::CallUpdateCurrentsHwy(
        f4_curr_ptr->data(),
        f4_volt_ptr->data(),
        Op,
        f4_volt_ptr,
        f4_curr_ptr,
        numLines[1],
        numVectors,
        startX,
        numX);
}

#endif  // HWY_ONCE
