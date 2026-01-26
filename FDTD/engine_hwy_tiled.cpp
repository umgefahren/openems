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

#include "engine_hwy_tiled.h"
#include "operator_sse_compressed.h"
#include "extensions/engine_extension.h"
#include "tools/array_ops.h"

#include <iostream>
#include <thread>
#include <vector>
#include <algorithm>

// Highway SIMD library
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "FDTD/engine_hwy_tiled.cpp"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

HWY_BEFORE_NAMESPACE();
namespace openEMS {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;
using F32x4 = hn::FixedTag<float, 4>;

/**
 * @brief Update voltages for a tile region
 */
HWY_ATTR void UpdateVoltagesTileHwy(
    f4vector* __restrict f4_volt,
    f4vector* __restrict f4_curr,
    const Operator_SSE_Compressed* Op,
    const ArrayLib::ArrayENG<f4vector>* f4_volt_ptr,
    const ArrayLib::ArrayENG<f4vector>* f4_curr_ptr,
    unsigned int x0, unsigned int x1,
    unsigned int y0, unsigned int y1,
    unsigned int z0, unsigned int z1)
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

    const f4vector* vv0 = Op->f4_vv_Compressed[0].data();
    const f4vector* vv1 = Op->f4_vv_Compressed[1].data();
    const f4vector* vv2 = Op->f4_vv_Compressed[2].data();
    const f4vector* vi0 = Op->f4_vi_Compressed[0].data();
    const f4vector* vi1 = Op->f4_vi_Compressed[1].data();
    const f4vector* vi2 = Op->f4_vi_Compressed[2].data();

    for (unsigned int posX = x0; posX < x1; ++posX) {
        const int i_shift_x = (posX > 0) ? i_stride_x : 0;
        const int volt_base_x = posX * v_stride_x;
        const int curr_base_x = posX * i_stride_x;
        const unsigned int idx_base_x = posX * idx_stride_x;

        for (unsigned int posY = y0; posY < y1; ++posY) {
            const int i_shift_y = (posY > 0) ? i_stride_y : 0;
            const int volt_base_xy = volt_base_x + posY * v_stride_y;
            const int curr_base_xy = curr_base_x + posY * i_stride_y;
            const unsigned int idx_base_xy = idx_base_x + posY * idx_stride_y;

            // Process z range (skip z=0 boundary, handle separately)
            unsigned int zStart = (z0 == 0) ? 1 : z0;
            for (unsigned int posZ = zStart; posZ < z1; ++posZ) {
                const unsigned int index = op_idx_data[idx_base_xy + posZ];
                const int v_off = volt_base_xy + posZ * v_stride_z;
                const int i_off = curr_base_xy + posZ * i_stride_z;

                // Load current field values ONCE and reuse
                const auto curr_hx = hn::Load(d, f4_curr[i_off].f);
                const auto curr_hy = hn::Load(d, f4_curr[i_off + i_stride_n].f);
                const auto curr_hz = hn::Load(d, f4_curr[i_off + 2*i_stride_n].f);

                // x-polarization
                auto volt_x = hn::Load(d, f4_volt[v_off].f);
                auto vv_x = hn::Load(d, vv0[index].f);
                auto vi_x = hn::Load(d, vi0[index].f);
                auto curl_x = hn::Sub(curr_hz, hn::Load(d, f4_curr[i_off + 2*i_stride_n - i_shift_y].f));
                curl_x = hn::Sub(curl_x, curr_hy);
                curl_x = hn::Add(curl_x, hn::Load(d, f4_curr[i_off + i_stride_n - i_stride_z].f));
                volt_x = hn::MulAdd(vi_x, curl_x, hn::Mul(volt_x, vv_x));
                hn::Store(volt_x, d, f4_volt[v_off].f);

                // y-polarization
                const int v_off_y = v_off + v_stride_n;
                auto volt_y = hn::Load(d, f4_volt[v_off_y].f);
                auto vv_y = hn::Load(d, vv1[index].f);
                auto vi_y = hn::Load(d, vi1[index].f);
                auto curl_y = hn::Sub(curr_hx, hn::Load(d, f4_curr[i_off - i_stride_z].f));
                curl_y = hn::Sub(curl_y, curr_hz);
                curl_y = hn::Add(curl_y, hn::Load(d, f4_curr[i_off + 2*i_stride_n - i_shift_x].f));
                volt_y = hn::MulAdd(vi_y, curl_y, hn::Mul(volt_y, vv_y));
                hn::Store(volt_y, d, f4_volt[v_off_y].f);

                // z-polarization
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

            // Handle z=0 boundary if in this tile
            if (z0 == 0) {
                const unsigned int numVecs = f4_volt_ptr->extent(3);
                const unsigned int index = op_idx_data[idx_base_xy];
                const int v_off = volt_base_xy;
                const int i_off_start = curr_base_xy;
                const int i_off_end = curr_base_xy + (numVecs - 1) * i_stride_z;

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
}

/**
 * @brief Update currents for a tile region
 */
HWY_ATTR void UpdateCurrentsTileHwy(
    f4vector* __restrict f4_curr,
    f4vector* __restrict f4_volt,
    const Operator_SSE_Compressed* Op,
    const ArrayLib::ArrayENG<f4vector>* f4_volt_ptr,
    const ArrayLib::ArrayENG<f4vector>* f4_curr_ptr,
    unsigned int numLines1,
    unsigned int numVectors,
    unsigned int x0, unsigned int x1,
    unsigned int y0, unsigned int y1,
    unsigned int z0, unsigned int z1)
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

    // Clamp y1 to numLines1-1 (current update doesn't include last Y)
    unsigned int y1_clamped = std::min(y1, numLines1 - 1);
    // Clamp z1 to numVectors-1 (current update doesn't include last Z)
    unsigned int z1_clamped = std::min(z1, numVectors - 1);

    for (unsigned int posX = x0; posX < x1; ++posX) {
        const int volt_base_x = posX * v_stride_x;
        const int curr_base_x = posX * i_stride_x;
        const unsigned int idx_base_x = posX * idx_stride_x;

        for (unsigned int posY = y0; posY < y1_clamped; ++posY) {
            const int volt_base_xy = volt_base_x + posY * v_stride_y;
            const int curr_base_xy = curr_base_x + posY * i_stride_y;
            const unsigned int idx_base_xy = idx_base_x + posY * idx_stride_y;

            for (unsigned int posZ = z0; posZ < z1_clamped; ++posZ) {
                const unsigned int index = op_idx_data[idx_base_xy + posZ];
                const int v_off = volt_base_xy + posZ * v_stride_z;
                const int i_off = curr_base_xy + posZ * i_stride_z;

                // Load voltage field values ONCE and reuse
                const auto volt_ex = hn::Load(d, f4_volt[v_off].f);
                const auto volt_ey = hn::Load(d, f4_volt[v_off + v_stride_n].f);
                const auto volt_ez = hn::Load(d, f4_volt[v_off + 2*v_stride_n].f);

                // x-polarization
                auto curr_x = hn::Load(d, f4_curr[i_off].f);
                auto ii_x = hn::Load(d, ii0[index].f);
                auto iv_x = hn::Load(d, iv0[index].f);
                auto curl_x = hn::Sub(volt_ez, hn::Load(d, f4_volt[v_off + 2*v_stride_n + v_stride_y].f));
                curl_x = hn::Sub(curl_x, volt_ey);
                curl_x = hn::Add(curl_x, hn::Load(d, f4_volt[v_off + v_stride_n + v_stride_z].f));
                curr_x = hn::MulAdd(iv_x, curl_x, hn::Mul(curr_x, ii_x));
                hn::Store(curr_x, d, f4_curr[i_off].f);

                // y-polarization
                const int i_off_y = i_off + i_stride_n;
                auto curr_y = hn::Load(d, f4_curr[i_off_y].f);
                auto ii_y = hn::Load(d, ii1[index].f);
                auto iv_y = hn::Load(d, iv1[index].f);
                auto curl_y = hn::Sub(volt_ex, hn::Load(d, f4_volt[v_off + v_stride_z].f));
                curl_y = hn::Sub(curl_y, volt_ez);
                curl_y = hn::Add(curl_y, hn::Load(d, f4_volt[v_off + 2*v_stride_n + v_stride_x].f));
                curr_y = hn::MulAdd(iv_y, curl_y, hn::Mul(curr_y, ii_y));
                hn::Store(curr_y, d, f4_curr[i_off_y].f);

                // z-polarization
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

            // Handle z=numVectors-1 boundary if in this tile
            if (z1 >= numVectors && z0 < numVectors) {
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
}

}  // namespace HWY_NAMESPACE
}  // namespace openEMS
HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace openEMS {

HWY_EXPORT(UpdateVoltagesTileHwy);
HWY_EXPORT(UpdateCurrentsTileHwy);

using UpdateVoltagesTilePtr = void (*)(
    f4vector*, f4vector*, const Operator_SSE_Compressed*,
    const ArrayLib::ArrayENG<f4vector>*, const ArrayLib::ArrayENG<f4vector>*,
    unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int);

using UpdateCurrentsTilePtr = void (*)(
    f4vector*, f4vector*, const Operator_SSE_Compressed*,
    const ArrayLib::ArrayENG<f4vector>*, const ArrayLib::ArrayENG<f4vector>*,
    unsigned int, unsigned int,
    unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int);

static UpdateVoltagesTilePtr g_UpdateVoltagesTilePtr = nullptr;
static UpdateCurrentsTilePtr g_UpdateCurrentsTilePtr = nullptr;

void InitTiledFunctionPointers() {
    if (!g_UpdateVoltagesTilePtr) {
        g_UpdateVoltagesTilePtr = HWY_DYNAMIC_POINTER(UpdateVoltagesTileHwy);
        g_UpdateCurrentsTilePtr = HWY_DYNAMIC_POINTER(UpdateCurrentsTileHwy);
    }
}

}  // namespace openEMS

using std::cout;
using std::endl;

Engine_Hwy_Tiled* Engine_Hwy_Tiled::New(const Operator_SSE_Compressed* op, unsigned int numThreads)
{
    cout << "Create FDTD engine (Highway Tiled - cache blocking + multi-threading)" << endl;
    cout << "  Tile size: " << TILE_X << "x" << TILE_Y << "x" << (TILE_Z*4) << " cells" << endl;

    openEMS::InitTiledFunctionPointers();

    Engine_Hwy_Tiled* e = new Engine_Hwy_Tiled(op);
    e->m_numThreads = (numThreads > 0) ? numThreads : std::thread::hardware_concurrency();
    e->Init();
    return e;
}

Engine_Hwy_Tiled::Engine_Hwy_Tiled(const Operator_SSE_Compressed* op) : Engine_sse(op)
{
    m_type = SSE;
    m_numThreads = 1;
    m_Op_Compressed = op;
}

Engine_Hwy_Tiled::~Engine_Hwy_Tiled()
{
}

void Engine_Hwy_Tiled::UpdateVoltagesTile(
    unsigned int x0, unsigned int x1,
    unsigned int y0, unsigned int y1,
    unsigned int z0, unsigned int z1)
{
    openEMS::g_UpdateVoltagesTilePtr(
        f4_volt_ptr->data(), f4_curr_ptr->data(), m_Op_Compressed,
        f4_volt_ptr, f4_curr_ptr,
        x0, x1, y0, y1, z0, z1);
}

void Engine_Hwy_Tiled::UpdateCurrentsTile(
    unsigned int x0, unsigned int x1,
    unsigned int y0, unsigned int y1,
    unsigned int z0, unsigned int z1)
{
    openEMS::g_UpdateCurrentsTilePtr(
        f4_curr_ptr->data(), f4_volt_ptr->data(), m_Op_Compressed,
        f4_volt_ptr, f4_curr_ptr,
        numLines[1], numVectors,
        x0, x1, y0, y1, z0, z1);
}

void Engine_Hwy_Tiled::UpdateTile(
    unsigned int x0, unsigned int x1,
    unsigned int y0, unsigned int y1,
    unsigned int z0, unsigned int z1)
{
    // This method is no longer used in the new tiled approach
    // Kept for API compatibility
    UpdateVoltagesTile(x0, x1, y0, y1, z0, z1);
    UpdateCurrentsTile(x0, x1, y0, y1, z0, z1);
}

bool Engine_Hwy_Tiled::IterateTS(unsigned int iterTS)
{
    // Spatial blocking with correct temporal ordering:
    // - All voltage updates must complete before any current updates
    // - This maintains FDTD leapfrog correctness
    // - Cache benefit: processing in tiles keeps data more local

    const unsigned int nx = numLines[0];
    const unsigned int ny = numLines[1];
    const unsigned int nz = numVectors;

    // Calculate number of tiles
    const unsigned int numTilesX = (nx + TILE_X - 1) / TILE_X;
    const unsigned int numTilesY = (ny + TILE_Y - 1) / TILE_Y;
    const unsigned int numTilesZ = (nz + TILE_Z - 1) / TILE_Z;

    for (unsigned int iter = 0; iter < iterTS; ++iter) {
        // ---- VOLTAGE PHASE ----
        DoPreVoltageUpdates();

        // Update all voltage tiles
        for (unsigned int tx = 0; tx < numTilesX; ++tx) {
            const unsigned int x0 = tx * TILE_X;
            const unsigned int x1 = std::min(x0 + TILE_X, nx);

            for (unsigned int ty = 0; ty < numTilesY; ++ty) {
                const unsigned int y0 = ty * TILE_Y;
                const unsigned int y1 = std::min(y0 + TILE_Y, ny);

                for (unsigned int tz = 0; tz < numTilesZ; ++tz) {
                    const unsigned int z0 = tz * TILE_Z;
                    const unsigned int z1 = std::min(z0 + TILE_Z, nz);
                    UpdateVoltagesTile(x0, x1, y0, y1, z0, z1);
                }
            }
        }

        DoPostVoltageUpdates();
        Apply2Voltages();

        // ---- CURRENT PHASE ----
        DoPreCurrentUpdates();

        // Update all current tiles (in same order for cache locality)
        for (unsigned int tx = 0; tx < numTilesX; ++tx) {
            const unsigned int x0 = tx * TILE_X;
            const unsigned int x1 = std::min(x0 + TILE_X, nx);

            for (unsigned int ty = 0; ty < numTilesY; ++ty) {
                const unsigned int y0 = ty * TILE_Y;
                const unsigned int y1 = std::min(y0 + TILE_Y, ny);

                for (unsigned int tz = 0; tz < numTilesZ; ++tz) {
                    const unsigned int z0 = tz * TILE_Z;
                    const unsigned int z1 = std::min(z0 + TILE_Z, nz);
                    UpdateCurrentsTile(x0, x1, y0, y1, z0, z1);
                }
            }
        }

        DoPostCurrentUpdates();
        Apply2Current();

        ++numTS;
    }
    return true;
}

#endif  // HWY_ONCE
