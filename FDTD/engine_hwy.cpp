/*
 *  Copyright (C) 2025
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

// Highway SIMD implementation of FDTD engine
// Uses Google Highway for portable SIMD across SSE, AVX, AVX-512, NEON, etc.

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "FDTD/engine_hwy.cpp"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"

#include "engine_hwy.h"
#include "tools/denormal.h"

#include <iostream>

HWY_BEFORE_NAMESPACE();
namespace openems {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

// Tag for float vectors - uses widest available SIMD
using D = hn::ScalableTag<float>;
static const D d;

// Get the number of float lanes in the current SIMD width
static const size_t kLanes = hn::Lanes(d);

// Update voltages using Highway SIMD
void UpdateVoltagesHwy(
    float* HWY_RESTRICT volt,
    const float* HWY_RESTRICT curr,
    const float* HWY_RESTRICT vv,
    const float* HWY_RESTRICT vi,
    const unsigned int numLines[3],
    const unsigned int numVectors,
    const unsigned int startX,
    const unsigned int numX)
{
    // Strides for array indexing (I-J-K-N layout)
    const size_t stride_n = 1;
    const size_t stride_k = 3;
    const size_t stride_j = numVectors * 4 * 3;
    const size_t stride_i = numLines[1] * stride_j;

    for (unsigned int x = startX; x < startX + numX; ++x)
    {
        const size_t shift_x = (x > 0) ? stride_i : 0;

        for (unsigned int y = 0; y < numLines[1]; ++y)
        {
            const size_t shift_y = (y > 0) ? stride_j : 0;
            const size_t base_xy = x * stride_i + y * stride_j;

            // Process in chunks of kLanes floats
            // For z >= 1 (main loop)
            for (unsigned int z = kLanes; z < numVectors * 4; z += kLanes)
            {
                const size_t base = base_xy + z * stride_k;

                // Load voltage and operator arrays
                auto volt_x = hn::Load(d, volt + base + 0 * stride_n);
                auto volt_y = hn::Load(d, volt + base + 1 * stride_n);
                auto volt_z = hn::Load(d, volt + base + 2 * stride_n);

                auto vv_x = hn::Load(d, vv + base + 0 * stride_n);
                auto vv_y = hn::Load(d, vv + base + 1 * stride_n);
                auto vv_z = hn::Load(d, vv + base + 2 * stride_n);

                auto vi_x = hn::Load(d, vi + base + 0 * stride_n);
                auto vi_y = hn::Load(d, vi + base + 1 * stride_n);
                auto vi_z = hn::Load(d, vi + base + 2 * stride_n);

                // Load current values for curl computation
                // For x-polarization: curl = curr_z(y) - curr_z(y-1) - curr_y(z) + curr_y(z-1)
                auto curr_z_y = hn::Load(d, curr + base + 2 * stride_n);
                auto curr_z_ym = hn::Load(d, curr + base + 2 * stride_n - shift_y);
                auto curr_y_z = hn::Load(d, curr + base + 1 * stride_n);
                auto curr_y_zm = hn::Load(d, curr + base + 1 * stride_n - kLanes * stride_k);

                auto curl_x = hn::Sub(hn::Sub(hn::Add(curr_z_y, curr_y_zm), curr_z_ym), curr_y_z);

                // For y-polarization: curl = curr_x(z) - curr_x(z-1) - curr_z(x) + curr_z(x-1)
                auto curr_x_z = hn::Load(d, curr + base + 0 * stride_n);
                auto curr_x_zm = hn::Load(d, curr + base + 0 * stride_n - kLanes * stride_k);
                auto curr_z_x = hn::Load(d, curr + base + 2 * stride_n);
                auto curr_z_xm = hn::Load(d, curr + base + 2 * stride_n - shift_x);

                auto curl_y = hn::Sub(hn::Sub(hn::Add(curr_x_z, curr_z_xm), curr_x_zm), curr_z_x);

                // For z-polarization: curl = curr_y(x) - curr_y(x-1) - curr_x(y) + curr_x(y-1)
                auto curr_y_x = hn::Load(d, curr + base + 1 * stride_n);
                auto curr_y_xm = hn::Load(d, curr + base + 1 * stride_n - shift_x);
                auto curr_x_y = hn::Load(d, curr + base + 0 * stride_n);
                auto curr_x_ym = hn::Load(d, curr + base + 0 * stride_n - shift_y);

                auto curl_z = hn::Sub(hn::Sub(hn::Add(curr_y_x, curr_x_ym), curr_y_xm), curr_x_y);

                // Update: volt = volt * vv + vi * curl
                volt_x = hn::MulAdd(vi_x, curl_x, hn::Mul(volt_x, vv_x));
                volt_y = hn::MulAdd(vi_y, curl_y, hn::Mul(volt_y, vv_y));
                volt_z = hn::MulAdd(vi_z, curl_z, hn::Mul(volt_z, vv_z));

                // Store results
                hn::Store(volt_x, d, volt + base + 0 * stride_n);
                hn::Store(volt_y, d, volt + base + 1 * stride_n);
                hn::Store(volt_z, d, volt + base + 2 * stride_n);
            }

            // Handle z = 0 boundary (simplified - just decay)
            {
                const size_t base = base_xy;
                for (size_t z = 0; z < kLanes && z < numVectors * 4; z += kLanes)
                {
                    auto volt_x = hn::Load(d, volt + base + z * stride_k + 0 * stride_n);
                    auto volt_y = hn::Load(d, volt + base + z * stride_k + 1 * stride_n);
                    auto volt_z = hn::Load(d, volt + base + z * stride_k + 2 * stride_n);

                    auto vv_x = hn::Load(d, vv + base + z * stride_k + 0 * stride_n);
                    auto vv_y = hn::Load(d, vv + base + z * stride_k + 1 * stride_n);
                    auto vv_z = hn::Load(d, vv + base + z * stride_k + 2 * stride_n);

                    volt_x = hn::Mul(volt_x, vv_x);
                    volt_y = hn::Mul(volt_y, vv_y);
                    volt_z = hn::Mul(volt_z, vv_z);

                    hn::Store(volt_x, d, volt + base + z * stride_k + 0 * stride_n);
                    hn::Store(volt_y, d, volt + base + z * stride_k + 1 * stride_n);
                    hn::Store(volt_z, d, volt + base + z * stride_k + 2 * stride_n);
                }
            }
        }
    }
}

// Update currents using Highway SIMD
void UpdateCurrentsHwy(
    float* HWY_RESTRICT curr,
    const float* HWY_RESTRICT volt,
    const float* HWY_RESTRICT ii,
    const float* HWY_RESTRICT iv,
    const unsigned int numLines[3],
    const unsigned int numVectors,
    const unsigned int startX,
    const unsigned int numX)
{
    const size_t stride_n = 1;
    const size_t stride_k = 3;
    const size_t stride_j = numVectors * 4 * 3;
    const size_t stride_i = numLines[1] * stride_j;

    for (unsigned int x = startX; x < startX + numX; ++x)
    {
        const size_t shift_x = stride_i;

        for (unsigned int y = 0; y < numLines[1] - 1; ++y)
        {
            const size_t shift_y = stride_j;
            const size_t base_xy = x * stride_i + y * stride_j;

            for (unsigned int z = 0; z < (numVectors - 1) * 4; z += kLanes)
            {
                const size_t base = base_xy + z * stride_k;

                auto curr_x = hn::Load(d, curr + base + 0 * stride_n);
                auto curr_y = hn::Load(d, curr + base + 1 * stride_n);
                auto curr_z = hn::Load(d, curr + base + 2 * stride_n);

                auto ii_x = hn::Load(d, ii + base + 0 * stride_n);
                auto ii_y = hn::Load(d, ii + base + 1 * stride_n);
                auto ii_z = hn::Load(d, ii + base + 2 * stride_n);

                auto iv_x = hn::Load(d, iv + base + 0 * stride_n);
                auto iv_y = hn::Load(d, iv + base + 1 * stride_n);
                auto iv_z = hn::Load(d, iv + base + 2 * stride_n);

                // For x-polarization: curl = volt_z(y) - volt_z(y+1) - volt_y(z) + volt_y(z+1)
                auto volt_z_y = hn::Load(d, volt + base + 2 * stride_n);
                auto volt_z_yp = hn::Load(d, volt + base + 2 * stride_n + shift_y);
                auto volt_y_z = hn::Load(d, volt + base + 1 * stride_n);
                auto volt_y_zp = hn::Load(d, volt + base + 1 * stride_n + kLanes * stride_k);

                auto curl_x = hn::Sub(hn::Sub(hn::Add(volt_z_y, volt_y_zp), volt_z_yp), volt_y_z);

                // For y-polarization
                auto volt_x_z = hn::Load(d, volt + base + 0 * stride_n);
                auto volt_x_zp = hn::Load(d, volt + base + 0 * stride_n + kLanes * stride_k);
                auto volt_z_x = hn::Load(d, volt + base + 2 * stride_n);
                auto volt_z_xp = hn::Load(d, volt + base + 2 * stride_n + shift_x);

                auto curl_y = hn::Sub(hn::Sub(hn::Add(volt_x_z, volt_z_xp), volt_x_zp), volt_z_x);

                // For z-polarization
                auto volt_y_x = hn::Load(d, volt + base + 1 * stride_n);
                auto volt_y_xp = hn::Load(d, volt + base + 1 * stride_n + shift_x);
                auto volt_x_y = hn::Load(d, volt + base + 0 * stride_n);
                auto volt_x_yp = hn::Load(d, volt + base + 0 * stride_n + shift_y);

                auto curl_z = hn::Sub(hn::Sub(hn::Add(volt_y_x, volt_x_yp), volt_y_xp), volt_x_y);

                // Update: curr = curr * ii + iv * curl
                curr_x = hn::MulAdd(iv_x, curl_x, hn::Mul(curr_x, ii_x));
                curr_y = hn::MulAdd(iv_y, curl_y, hn::Mul(curr_y, ii_y));
                curr_z = hn::MulAdd(iv_z, curl_z, hn::Mul(curr_z, ii_z));

                hn::Store(curr_x, d, curr + base + 0 * stride_n);
                hn::Store(curr_y, d, curr + base + 1 * stride_n);
                hn::Store(curr_z, d, curr + base + 2 * stride_n);
            }
        }
    }
}

}  // namespace HWY_NAMESPACE
}  // namespace openems
HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace openems {

// Dynamic dispatch table
HWY_EXPORT(UpdateVoltagesHwy);
HWY_EXPORT(UpdateCurrentsHwy);

// Wrapper that calls best available implementation
void CallUpdateVoltagesHwy(
    float* volt, const float* curr, const float* vv, const float* vi,
    const unsigned int numLines[3], const unsigned int numVectors,
    const unsigned int startX, const unsigned int numX)
{
    HWY_DYNAMIC_DISPATCH(UpdateVoltagesHwy)(volt, curr, vv, vi, numLines, numVectors, startX, numX);
}

void CallUpdateCurrentsHwy(
    float* curr, const float* volt, const float* ii, const float* iv,
    const unsigned int numLines[3], const unsigned int numVectors,
    const unsigned int startX, const unsigned int numX)
{
    HWY_DYNAMIC_DISPATCH(UpdateCurrentsHwy)(curr, volt, ii, iv, numLines, numVectors, startX, numX);
}

}  // namespace openems

// Engine implementation
using std::cout;
using std::endl;

Engine_Hwy* Engine_Hwy::New(const Operator_sse* op)
{
    cout << "Create FDTD engine (Highway SIMD - auto-vectorized)" << endl;

    // Print which SIMD target is being used
    cout << "Highway target: " << hwy::TargetName(hwy::DispatchedTarget()) << endl;

    Engine_Hwy* e = new Engine_Hwy(op);
    e->Init();
    return e;
}

Engine_Hwy::Engine_Hwy(const Operator_sse* op) : Engine(op)
{
    m_type = SSE;  // Use SSE type for compatibility
    Op = op;
    f4_volt_ptr = NULL;
    f4_curr_ptr = NULL;
    numVectors = ceil((double)numLines[2] / 4.0);

    // Disable denormals for performance
    Denormal::Disable();
}

Engine_Hwy::~Engine_Hwy()
{
    Reset();
}

void Engine_Hwy::Init()
{
    Engine::Init();

    // Free base class arrays - we use our own
    delete volt_ptr;
    volt_ptr = NULL;
    delete curr_ptr;
    curr_ptr = NULL;

    // Allocate Highway-compatible arrays (aligned)
    f4_volt_ptr = new ArrayLib::ArrayENG<f4vector>(
        "f4_volt", {numLines[0], numLines[1], numVectors}
    );
    f4_curr_ptr = new ArrayLib::ArrayENG<f4vector>(
        "f4_curr", {numLines[0], numLines[1], numVectors}
    );
}

void Engine_Hwy::Reset()
{
    Engine::Reset();
    delete f4_volt_ptr;
    f4_volt_ptr = NULL;
    delete f4_curr_ptr;
    f4_curr_ptr = NULL;
}

void Engine_Hwy::UpdateVoltages(unsigned int startX, unsigned int numX)
{
    // Get raw pointers to data
    float* volt = reinterpret_cast<float*>(f4_volt_ptr->data());
    const float* curr = reinterpret_cast<const float*>(f4_curr_ptr->data());
    const float* vv = reinterpret_cast<const float*>(Op->f4_vv_ptr->data());
    const float* vi = reinterpret_cast<const float*>(Op->f4_vi_ptr->data());

    openems::CallUpdateVoltagesHwy(volt, curr, vv, vi, numLines, numVectors, startX, numX);
}

void Engine_Hwy::UpdateCurrents(unsigned int startX, unsigned int numX)
{
    float* curr = reinterpret_cast<float*>(f4_curr_ptr->data());
    const float* volt = reinterpret_cast<const float*>(f4_volt_ptr->data());
    const float* ii = reinterpret_cast<const float*>(Op->f4_ii_ptr->data());
    const float* iv = reinterpret_cast<const float*>(Op->f4_iv_ptr->data());

    openems::CallUpdateCurrentsHwy(curr, volt, ii, iv, numLines, numVectors, startX, numX);
}

#endif  // HWY_ONCE
