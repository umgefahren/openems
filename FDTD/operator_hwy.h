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

#ifndef OPERATOR_HWY_H
#define OPERATOR_HWY_H

#include "operator_sse_compressed.h"

/**
 * @brief Operator for the Highway SIMD engine
 *
 * This operator uses the same compressed coefficient storage as
 * Operator_SSE_Compressed but creates the Highway-based engine
 * which uses wider SIMD operations (AVX2/AVX-512) when available.
 */
class Operator_Hwy : public Operator_SSE_Compressed
{
public:
	static Operator_Hwy* New();
	virtual ~Operator_Hwy();

	virtual Engine* CreateEngine() override;

protected:
	Operator_Hwy();
};

#endif // OPERATOR_HWY_H
