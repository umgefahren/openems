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

#ifndef ENGINE_HWY_TILED_H
#define ENGINE_HWY_TILED_H

#include "engine_sse.h"

class Operator_SSE_Compressed;

/**
 * @brief Temporally-blocked FDTD engine using cache tiling
 *
 * This engine improves cache efficiency by processing the grid in tiles
 * that fit in L2 cache. For each tile, it performs both voltage and current
 * updates before moving to the next tile, keeping both field arrays hot
 * in cache.
 *
 * Tile size is chosen to fit in L2 cache:
 * - 6 arrays (volt_x/y/z, curr_x/y/z) × tile_volume × 4 bytes
 * - Default tile: 32×32×32 = 32K cells × 24 bytes = 768KB
 *
 * This approach trades some computational redundancy at tile boundaries
 * for significantly better cache utilization.
 */
class Engine_Hwy_Tiled : public Engine_sse
{
public:
    static Engine_Hwy_Tiled* New(const Operator_SSE_Compressed* op, unsigned int numThreads = 0);
    virtual ~Engine_Hwy_Tiled();

    // Override the main iteration to use tiled updates
    virtual bool IterateTS(unsigned int iterTS) override;

protected:
    Engine_Hwy_Tiled(const Operator_SSE_Compressed* op);

    // Tiled update functions
    void UpdateTile(unsigned int x0, unsigned int x1,
                    unsigned int y0, unsigned int y1,
                    unsigned int z0, unsigned int z1);

    void UpdateVoltagesTile(unsigned int x0, unsigned int x1,
                            unsigned int y0, unsigned int y1,
                            unsigned int z0, unsigned int z1);

    void UpdateCurrentsTile(unsigned int x0, unsigned int x1,
                            unsigned int y0, unsigned int y1,
                            unsigned int z0, unsigned int z1);

    // Tile dimensions (tunable for different cache sizes)
    static constexpr unsigned int TILE_X = 16;  // X tile size
    static constexpr unsigned int TILE_Y = 32;  // Y tile size
    static constexpr unsigned int TILE_Z = 8;   // Z tile size (in f4vectors, so 32 floats)

    unsigned int m_numThreads;
    const Operator_SSE_Compressed* m_Op_Compressed;
};

#endif // ENGINE_HWY_TILED_H
