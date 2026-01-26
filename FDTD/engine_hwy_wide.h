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

#ifndef ENGINE_HWY_WIDE_H
#define ENGINE_HWY_WIDE_H

#include "engine_multithread.h"

class Operator_Hwy_Wide;

/**
 * @brief High-performance FDTD engine using Google Highway for true AVX-512 SIMD
 *
 * This engine uses wide SIMD operations (512 bits on AVX-512) to process 16 floats
 * per instruction. Unlike Engine_Hwy which is constrained to 128-bit f4vector
 * operations, this engine uses flat aligned arrays that enable full SIMD width.
 *
 * Key features:
 * - Uses Highway's ScalableTag<float> for maximum SIMD width
 * - Processes 16 floats (512 bits) per operation on AVX-512
 * - Falls back to 8 floats on AVX2, 4 floats on SSE
 * - 64-byte aligned memory access for optimal cache line usage
 * - Z-dimension padded to SIMD width for aligned vectorization
 */
class Engine_Hwy_Wide : public Engine_Multithread
{
public:
    static Engine_Hwy_Wide* New(const Operator_Hwy_Wide* op, unsigned int numThreads = 0);
    virtual ~Engine_Hwy_Wide();

    virtual void Init() override;
    virtual void Reset() override;

    // Field accessors for extensions
    virtual FDTD_FLOAT GetVolt(unsigned int n, unsigned int x, unsigned int y, unsigned int z) const override;
    virtual FDTD_FLOAT GetVolt(unsigned int n, const unsigned int pos[3]) const override;
    virtual FDTD_FLOAT GetCurr(unsigned int n, unsigned int x, unsigned int y, unsigned int z) const override;
    virtual FDTD_FLOAT GetCurr(unsigned int n, const unsigned int pos[3]) const override;

    virtual void SetVolt(unsigned int n, unsigned int x, unsigned int y, unsigned int z, FDTD_FLOAT value) override;
    virtual void SetVolt(unsigned int n, const unsigned int pos[3], FDTD_FLOAT value) override;
    virtual void SetCurr(unsigned int n, unsigned int x, unsigned int y, unsigned int z, FDTD_FLOAT value) override;
    virtual void SetCurr(unsigned int n, const unsigned int pos[3], FDTD_FLOAT value) override;

protected:
    Engine_Hwy_Wide(const Operator_Hwy_Wide* op);

    virtual void UpdateVoltages(unsigned int startX, unsigned int numX) override;
    virtual void UpdateCurrents(unsigned int startX, unsigned int numX) override;

    const Operator_Hwy_Wide* m_Op_Wide;

    // Flat aligned field arrays
    float* m_volt[3];  // [polarization]
    float* m_curr[3];

    // Dimensions
    unsigned int m_nx, m_ny, m_nz;
    unsigned int m_paddedNz;
    size_t m_stride_x, m_stride_y;
    size_t m_totalSize;

    // Helper for linear index
    inline size_t LinearIndex(unsigned int x, unsigned int y, unsigned int z) const {
        return static_cast<size_t>(x) * m_stride_x + static_cast<size_t>(y) * m_stride_y + z;
    }

    // Aligned allocation helper
    template<typename T>
    static T* AlignedAlloc(size_t count, size_t alignment = 64) {
        void* ptr = std::aligned_alloc(alignment, count * sizeof(T));
        if (!ptr) throw std::bad_alloc();
        return static_cast<T*>(ptr);
    }
};

#endif // ENGINE_HWY_WIDE_H
