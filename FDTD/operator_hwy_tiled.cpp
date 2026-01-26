/*
*	Copyright (C) 2025 OpenEMS Contributors
*
*	This program is free software: you can redistribute it and/or modify
*	it under the terms of the GNU General Public License as published by
*	the Free Software Foundation, either version 3 of the License, or
*	(at your option) any later version.
*/

#include "operator_hwy_tiled.h"
#include "engine_hwy_tiled.h"

#include <iostream>
#include <thread>

using std::cout;
using std::endl;

Operator_Hwy_Tiled* Operator_Hwy_Tiled::New(unsigned int numThreads)
{
    cout << "Create FDTD operator (Highway Tiled - cache blocking)" << endl;
    Operator_Hwy_Tiled* op = new Operator_Hwy_Tiled();
    op->m_numThreads = (numThreads > 0) ? numThreads : std::thread::hardware_concurrency();
    op->Init();
    return op;
}

Operator_Hwy_Tiled::Operator_Hwy_Tiled() : Operator_SSE_Compressed()
{
    m_numThreads = 1;
}

Operator_Hwy_Tiled::~Operator_Hwy_Tiled()
{
}

Engine* Operator_Hwy_Tiled::CreateEngine()
{
    m_Engine = Engine_Hwy_Tiled::New(this, m_numThreads);
    return m_Engine;
}
