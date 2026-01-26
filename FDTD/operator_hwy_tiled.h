/*
*	Copyright (C) 2025 OpenEMS Contributors
*
*	This program is free software: you can redistribute it and/or modify
*	it under the terms of the GNU General Public License as published by
*	the Free Software Foundation, either version 3 of the License, or
*	(at your option) any later version.
*/

#ifndef OPERATOR_HWY_TILED_H
#define OPERATOR_HWY_TILED_H

#include "operator_sse_compressed.h"

/**
 * @brief Operator for temporally-blocked (tiled) FDTD engine
 *
 * Uses the same compressed coefficient storage as Operator_SSE_Compressed
 * but creates Engine_Hwy_Tiled which uses cache tiling for better
 * memory bandwidth utilization.
 */
class Operator_Hwy_Tiled : public Operator_SSE_Compressed
{
public:
    static Operator_Hwy_Tiled* New(unsigned int numThreads = 0);
    virtual ~Operator_Hwy_Tiled();

    virtual Engine* CreateEngine() override;

protected:
    Operator_Hwy_Tiled();

    unsigned int m_numThreads;
};

#endif // OPERATOR_HWY_TILED_H
