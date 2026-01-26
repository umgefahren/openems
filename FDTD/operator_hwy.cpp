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

#ifdef USE_HWY

#include "operator_hwy.h"
#include "engine_hwy.h"

using std::cout;
using std::endl;

Operator_Hwy* Operator_Hwy::New(unsigned int numThreads)
{
	cout << "Create FDTD operator (Highway SIMD + multi-threading)" << endl;
	Operator_Hwy* op = new Operator_Hwy();
	op->setNumThreads(numThreads);
	op->Init();
	return op;
}

Operator_Hwy::Operator_Hwy() : Operator_Multithread()
{
}

Operator_Hwy::~Operator_Hwy()
{
}

Engine* Operator_Hwy::CreateEngine()
{
	m_Engine = Engine_Hwy::New(this, m_numThreads);
	return m_Engine;
}

#endif  // USE_HWY
