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

#include "operator_hwy_wide.h"
#include "engine_hwy_wide.h"

#include <cstring>
#include <iostream>

using std::cout;
using std::endl;

Operator_Hwy_Wide* Operator_Hwy_Wide::New(unsigned int numThreads)
{
    cout << "Create FDTD operator (Highway Wide SIMD + multi-threading)" << endl;
    Operator_Hwy_Wide* op = new Operator_Hwy_Wide();
    op->setNumThreads(numThreads);
    op->Init();
    return op;
}

Operator_Hwy_Wide::Operator_Hwy_Wide() : Operator_Multithread()
{
    m_paddedNz = 0;
    m_stride_x = 0;
    m_stride_y = 0;
    m_totalSize = 0;

    for (int n = 0; n < 3; ++n) {
        m_vv[n] = nullptr;
        m_vi[n] = nullptr;
        m_iv[n] = nullptr;
        m_ii[n] = nullptr;
    }
}

Operator_Hwy_Wide::~Operator_Hwy_Wide()
{
    DeleteWideArrays();
}

void Operator_Hwy_Wide::Init()
{
    Operator_Multithread::Init();
}

void Operator_Hwy_Wide::Reset()
{
    DeleteWideArrays();
    Operator_Multithread::Reset();
}

void Operator_Hwy_Wide::AllocateWideArrays()
{
    DeleteWideArrays();

    // Pad Z dimension to multiple of 16 for AVX-512 alignment
    const unsigned int nz = numLines[2];
    m_paddedNz = ((nz + 15) / 16) * 16;  // Round up to multiple of 16

    m_stride_y = m_paddedNz;
    m_stride_x = numLines[1] * m_paddedNz;
    m_totalSize = static_cast<size_t>(numLines[0]) * m_stride_x;

    cout << "  Wide arrays: " << numLines[0] << "x" << numLines[1] << "x" << nz
         << " (padded z: " << m_paddedNz << ")" << endl;
    cout << "  Total coefficient memory: "
         << (m_totalSize * sizeof(float) * 12 / (1024*1024)) << " MB" << endl;

    // Allocate 64-byte aligned arrays for each polarization
    for (int n = 0; n < 3; ++n) {
        m_vv[n] = AlignedAlloc<float>(m_totalSize);
        m_vi[n] = AlignedAlloc<float>(m_totalSize);
        m_iv[n] = AlignedAlloc<float>(m_totalSize);
        m_ii[n] = AlignedAlloc<float>(m_totalSize);

        // Initialize to safe defaults
        std::memset(m_vv[n], 0, m_totalSize * sizeof(float));
        std::memset(m_vi[n], 0, m_totalSize * sizeof(float));
        std::memset(m_iv[n], 0, m_totalSize * sizeof(float));
        std::memset(m_ii[n], 0, m_totalSize * sizeof(float));
    }
}

void Operator_Hwy_Wide::DeleteWideArrays()
{
    for (int n = 0; n < 3; ++n) {
        if (m_vv[n]) { std::free(m_vv[n]); m_vv[n] = nullptr; }
        if (m_vi[n]) { std::free(m_vi[n]); m_vi[n] = nullptr; }
        if (m_iv[n]) { std::free(m_iv[n]); m_iv[n] = nullptr; }
        if (m_ii[n]) { std::free(m_ii[n]); m_ii[n] = nullptr; }
    }
    m_totalSize = 0;
}

int Operator_Hwy_Wide::CalcECOperator(DebugFlags debugFlags)
{
    // First, let base class calculate coefficients using standard method
    int result = Operator_Multithread::CalcECOperator(debugFlags);
    if (result != 0)
        return result;

    // Now convert from compressed SSE format to wide format
    AllocateWideArrays();

    cout << "  Converting coefficients to wide SIMD layout..." << endl;

    // Copy coefficients from compressed format to wide format
    for (unsigned int x = 0; x < numLines[0]; ++x) {
        for (unsigned int y = 0; y < numLines[1]; ++y) {
            for (unsigned int z = 0; z < numLines[2]; ++z) {
                const size_t idx = LinearIndex(x, y, z);

                for (int n = 0; n < 3; ++n) {
                    // Get from parent's compressed storage
                    m_vv[n][idx] = Operator_SSE_Compressed::GetVV(n, x, y, z);
                    m_vi[n][idx] = Operator_SSE_Compressed::GetVI(n, x, y, z);
                    m_iv[n][idx] = Operator_SSE_Compressed::GetIV(n, x, y, z);
                    m_ii[n][idx] = Operator_SSE_Compressed::GetII(n, x, y, z);
                }
            }
        }
    }

    return 0;
}

Engine* Operator_Hwy_Wide::CreateEngine()
{
    m_Engine = Engine_Hwy_Wide::New(this, m_numThreads);
    return m_Engine;
}

// Coefficient accessors using wide layout
FDTD_FLOAT Operator_Hwy_Wide::GetVV(unsigned int n, unsigned int x, unsigned int y, unsigned int z) const
{
    if (m_vv[n])
        return m_vv[n][LinearIndex(x, y, z)];
    return Operator_SSE_Compressed::GetVV(n, x, y, z);
}

FDTD_FLOAT Operator_Hwy_Wide::GetVI(unsigned int n, unsigned int x, unsigned int y, unsigned int z) const
{
    if (m_vi[n])
        return m_vi[n][LinearIndex(x, y, z)];
    return Operator_SSE_Compressed::GetVI(n, x, y, z);
}

FDTD_FLOAT Operator_Hwy_Wide::GetII(unsigned int n, unsigned int x, unsigned int y, unsigned int z) const
{
    if (m_ii[n])
        return m_ii[n][LinearIndex(x, y, z)];
    return Operator_SSE_Compressed::GetII(n, x, y, z);
}

FDTD_FLOAT Operator_Hwy_Wide::GetIV(unsigned int n, unsigned int x, unsigned int y, unsigned int z) const
{
    if (m_iv[n])
        return m_iv[n][LinearIndex(x, y, z)];
    return Operator_SSE_Compressed::GetIV(n, x, y, z);
}

void Operator_Hwy_Wide::SetVV(unsigned int n, unsigned int x, unsigned int y, unsigned int z, FDTD_FLOAT value)
{
    if (m_vv[n])
        m_vv[n][LinearIndex(x, y, z)] = value;
    Operator_SSE_Compressed::SetVV(n, x, y, z, value);
}

void Operator_Hwy_Wide::SetVI(unsigned int n, unsigned int x, unsigned int y, unsigned int z, FDTD_FLOAT value)
{
    if (m_vi[n])
        m_vi[n][LinearIndex(x, y, z)] = value;
    Operator_SSE_Compressed::SetVI(n, x, y, z, value);
}

void Operator_Hwy_Wide::SetII(unsigned int n, unsigned int x, unsigned int y, unsigned int z, FDTD_FLOAT value)
{
    if (m_ii[n])
        m_ii[n][LinearIndex(x, y, z)] = value;
    Operator_SSE_Compressed::SetII(n, x, y, z, value);
}

void Operator_Hwy_Wide::SetIV(unsigned int n, unsigned int x, unsigned int y, unsigned int z, FDTD_FLOAT value)
{
    if (m_iv[n])
        m_iv[n][LinearIndex(x, y, z)] = value;
    Operator_SSE_Compressed::SetIV(n, x, y, z, value);
}
