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

#ifndef ENGINE_HWY_H
#define ENGINE_HWY_H

#include "operator_sse.h"
#include "engine.h"

// Forward declare the Highway-specific implementation
class Engine_Hwy : public Engine
{
public:
	static Engine_Hwy* New(const Operator_sse* op);
	virtual ~Engine_Hwy();

	virtual void Init();
	virtual void Reset();

	virtual unsigned int GetNumberOfTimesteps() const { return numTS; }

	// Field access - uses the same f4vector arrays as SSE engine
	inline virtual FDTD_FLOAT GetVolt(unsigned int n, unsigned int x, unsigned int y, unsigned int z) const;
	inline virtual FDTD_FLOAT GetVolt(unsigned int n, const unsigned int pos[3]) const;
	inline virtual FDTD_FLOAT GetCurr(unsigned int n, unsigned int x, unsigned int y, unsigned int z) const;
	inline virtual FDTD_FLOAT GetCurr(unsigned int n, const unsigned int pos[3]) const;

	inline virtual void SetVolt(unsigned int n, unsigned int x, unsigned int y, unsigned int z, FDTD_FLOAT value);
	inline virtual void SetVolt(unsigned int n, const unsigned int pos[3], FDTD_FLOAT value);
	inline virtual void SetCurr(unsigned int n, unsigned int x, unsigned int y, unsigned int z, FDTD_FLOAT value);
	inline virtual void SetCurr(unsigned int n, const unsigned int pos[3], FDTD_FLOAT value);

protected:
	Engine_Hwy(const Operator_sse* op);

	const Operator_sse* Op;

	virtual void UpdateVoltages(unsigned int startX, unsigned int numX);
	virtual void UpdateCurrents(unsigned int startX, unsigned int numX);

	unsigned int numVectors;

	// Use same array structure as SSE for compatibility
	ArrayLib::ArrayENG<f4vector>* f4_volt_ptr;
	ArrayLib::ArrayENG<f4vector>* f4_curr_ptr;
};

// Inline implementations for field access
inline FDTD_FLOAT Engine_Hwy::GetVolt(unsigned int n, unsigned int x, unsigned int y, unsigned int z) const
{
	ArrayLib::ArrayENG<f4vector>& f4_volt = *f4_volt_ptr;
	return f4_volt(n, x, y, z/4).f[z%4];
}

inline FDTD_FLOAT Engine_Hwy::GetVolt(unsigned int n, const unsigned int pos[3]) const
{
	ArrayLib::ArrayENG<f4vector>& f4_volt = *f4_volt_ptr;
	return f4_volt(n, pos[0], pos[1], pos[2]/4).f[pos[2]%4];
}

inline FDTD_FLOAT Engine_Hwy::GetCurr(unsigned int n, unsigned int x, unsigned int y, unsigned int z) const
{
	ArrayLib::ArrayENG<f4vector>& f4_curr = *f4_curr_ptr;
	return f4_curr(n, x, y, z/4).f[z%4];
}

inline FDTD_FLOAT Engine_Hwy::GetCurr(unsigned int n, const unsigned int pos[3]) const
{
	ArrayLib::ArrayENG<f4vector>& f4_curr = *f4_curr_ptr;
	return f4_curr(n, pos[0], pos[1], pos[2]/4).f[pos[2]%4];
}

inline void Engine_Hwy::SetVolt(unsigned int n, unsigned int x, unsigned int y, unsigned int z, FDTD_FLOAT value)
{
	ArrayLib::ArrayENG<f4vector>& f4_volt = *f4_volt_ptr;
	f4_volt(n, x, y, z/4).f[z%4] = value;
}

inline void Engine_Hwy::SetVolt(unsigned int n, const unsigned int pos[3], FDTD_FLOAT value)
{
	ArrayLib::ArrayENG<f4vector>& f4_volt = *f4_volt_ptr;
	f4_volt(n, pos[0], pos[1], pos[2]/4).f[pos[2]%4] = value;
}

inline void Engine_Hwy::SetCurr(unsigned int n, unsigned int x, unsigned int y, unsigned int z, FDTD_FLOAT value)
{
	ArrayLib::ArrayENG<f4vector>& f4_curr = *f4_curr_ptr;
	f4_curr(n, x, y, z/4).f[z%4] = value;
}

inline void Engine_Hwy::SetCurr(unsigned int n, const unsigned int pos[3], FDTD_FLOAT value)
{
	ArrayLib::ArrayENG<f4vector>& f4_curr = *f4_curr_ptr;
	f4_curr(n, pos[0], pos[1], pos[2]/4).f[pos[2]%4] = value;
}

#endif // ENGINE_HWY_H
