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

#ifndef OPERATOR_HWY_WIDE_H
#define OPERATOR_HWY_WIDE_H

#include "operator_multithread.h"
#include <memory>
#include <cstdlib>

/**
 * @brief AVX-512 optimized operator with wide SIMD data layout
 *
 * This operator stores field and coefficient data in flat, 64-byte aligned
 * arrays optimized for 512-bit SIMD operations. Unlike the compressed SSE
 * operator which uses 4-float f4vectors, this uses a Structure-of-Arrays
 * layout that enables processing 16 floats (512 bits) per operation.
 *
 * Memory layout:
 * - Fields (volt, curr): flat float arrays indexed as [x*stride_x + y*stride_y + z]
 * - Coefficients (vv, vi, iv, ii): same layout, directly indexed without compression
 * - All arrays are 64-byte aligned for optimal AVX-512 cache line access
 * - Z-dimension is padded to multiple of 16 for aligned wide loads
 */
class Operator_Hwy_Wide : public Operator_Multithread
{
    friend class Engine_Hwy_Wide;
public:
    static Operator_Hwy_Wide* New(unsigned int numThreads = 0);
    virtual ~Operator_Hwy_Wide();

    virtual Engine* CreateEngine() override;

    // Override coefficient accessors to use wide layout
    virtual FDTD_FLOAT GetVV(unsigned int n, unsigned int x, unsigned int y, unsigned int z) const override;
    virtual FDTD_FLOAT GetVI(unsigned int n, unsigned int x, unsigned int y, unsigned int z) const override;
    virtual FDTD_FLOAT GetII(unsigned int n, unsigned int x, unsigned int y, unsigned int z) const override;
    virtual FDTD_FLOAT GetIV(unsigned int n, unsigned int x, unsigned int y, unsigned int z) const override;

    virtual void SetVV(unsigned int n, unsigned int x, unsigned int y, unsigned int z, FDTD_FLOAT value) override;
    virtual void SetVI(unsigned int n, unsigned int x, unsigned int y, unsigned int z, FDTD_FLOAT value) override;
    virtual void SetII(unsigned int n, unsigned int x, unsigned int y, unsigned int z, FDTD_FLOAT value) override;
    virtual void SetIV(unsigned int n, unsigned int x, unsigned int y, unsigned int z, FDTD_FLOAT value) override;

    // Get padded dimensions
    unsigned int GetPaddedNz() const { return m_paddedNz; }
    unsigned int GetStrideX() const { return m_stride_x; }
    unsigned int GetStrideY() const { return m_stride_y; }

protected:
    Operator_Hwy_Wide();

    virtual void Init() override;
    virtual void Reset() override;
    virtual int CalcECOperator(DebugFlags debugFlags = None) override;

    // Allocate aligned memory
    void AllocateWideArrays();
    void DeleteWideArrays();

    // Convert linear index to array position
    inline size_t LinearIndex(unsigned int x, unsigned int y, unsigned int z) const {
        return static_cast<size_t>(x) * m_stride_x + static_cast<size_t>(y) * m_stride_y + z;
    }

    // Padded dimensions for aligned access
    unsigned int m_paddedNz;      // Z padded to multiple of 16
    size_t m_stride_x;            // = ny * paddedNz
    size_t m_stride_y;            // = paddedNz
    size_t m_totalSize;           // = nx * ny * paddedNz

    // 64-byte aligned coefficient arrays (one per polarization)
    // vv: voltage coefficient from old voltage
    // vi: voltage coefficient from old current
    // iv: current coefficient from old voltage
    // ii: current coefficient from old current
    float* m_vv[3];  // [polarization]
    float* m_vi[3];
    float* m_iv[3];
    float* m_ii[3];

    // Helper for aligned allocation
    template<typename T>
    static T* AlignedAlloc(size_t count, size_t alignment = 64) {
        void* ptr = std::aligned_alloc(alignment, count * sizeof(T));
        if (!ptr) throw std::bad_alloc();
        return static_cast<T*>(ptr);
    }
};

#endif // OPERATOR_HWY_WIDE_H
