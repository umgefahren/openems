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

#include "operator_hwy.h"
#include "engine_hwy.h"
#include "engine_sse.h"

using std::cout;
using std::endl;

Operator_Hwy* Operator_Hwy::New()
{
	cout << "Create FDTD operator (Highway SIMD)" << endl;
	Operator_Hwy* op = new Operator_Hwy();
	op->Init();
	return op;
}

Operator_Hwy::Operator_Hwy() : Operator_SSE_Compressed()
{
}

Operator_Hwy::~Operator_Hwy()
{
}

Engine* Operator_Hwy::CreateEngine()
{
	if (!m_Use_Compression)
	{
		// If compression failed, fall back to basic SSE engine
		m_Engine = Engine_sse::New(this);
	}
	else
	{
		m_Engine = Engine_Hwy::New(this);
	}
	return m_Engine;
}
