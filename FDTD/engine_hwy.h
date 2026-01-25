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

#ifndef ENGINE_HWY_H
#define ENGINE_HWY_H

#include "engine_sse_compressed.h"

/**
 * @brief High-performance FDTD engine using Google Highway for portable SIMD
 *
 * This engine extends Engine_SSE_Compressed and uses Google Highway to
 * automatically select the best available SIMD instruction set (SSE, AVX2,
 * AVX-512, NEON, etc.) at runtime.
 *
 * The data layout remains compatible with the compressed SSE operator,
 * but the update kernels use wider SIMD operations where available.
 */
class Engine_Hwy : public Engine_SSE_Compressed
{
public:
	static Engine_Hwy* New(const Operator_SSE_Compressed* op);
	virtual ~Engine_Hwy();

protected:
	Engine_Hwy(const Operator_SSE_Compressed* op);

	virtual void UpdateVoltages(unsigned int startX, unsigned int numX) override;
	virtual void UpdateCurrents(unsigned int startX, unsigned int numX) override;
};

#endif // ENGINE_HWY_H
