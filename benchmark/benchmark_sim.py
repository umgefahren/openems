#!/usr/bin/env python3
"""
Benchmark simulation for profiling OpenEMS performance.
Creates a 3D cavity simulation with enough cells to stress the engine.
"""

import os
import sys
import tempfile
import time
import argparse

import numpy as np
from CSXCAD import ContinuousStructure
from openEMS import openEMS
from openEMS.physical_constants import C0

def create_benchmark_simulation(grid_size=100, num_timesteps=5000, engine='fastest'):
    """
    Create a benchmark simulation with a 3D resonant cavity.

    Args:
        grid_size: Number of cells in each dimension (total cells = grid_size^3)
        num_timesteps: Number of time steps to simulate
        engine: Engine type ('basic', 'sse', 'sse-compressed', 'multithreaded', 'fastest')
    """
    sim_path = os.path.join(tempfile.gettempdir(), f'openems_benchmark_{grid_size}')

    # Physical dimensions (in mm)
    unit = 1e-3
    cavity_size = [100, 100, 100]  # 100mm x 100mm x 100mm cavity

    # Calculate mesh resolution
    f_max = 3e9  # 3 GHz
    resolution = C0 / f_max / unit / 10  # lambda/10 resolution

    # Force specific grid size
    dx = cavity_size[0] / grid_size
    dy = cavity_size[1] / grid_size
    dz = cavity_size[2] / grid_size

    # Create FDTD object
    FDTD = openEMS(NrTS=num_timesteps, EndCriteria=0)  # EndCriteria=0 means run all timesteps
    FDTD.SetGaussExcite(f_max/2, f_max/2)
    FDTD.SetBoundaryCond(['PEC', 'PEC', 'PEC', 'PEC', 'PEC', 'PEC'])

    # Create structure
    CSX = ContinuousStructure()
    FDTD.SetCSX(CSX)
    mesh = CSX.GetGrid()
    mesh.SetDeltaUnit(unit)

    # Create regular mesh
    mesh.AddLine('x', np.linspace(0, cavity_size[0], grid_size + 1))
    mesh.AddLine('y', np.linspace(0, cavity_size[1], grid_size + 1))
    mesh.AddLine('z', np.linspace(0, cavity_size[2], grid_size + 1))

    # Add excitation probe in center
    exc = CSX.AddExcitation('excite', exc_type=0, exc_val=[1, 0, 0])  # E-field excitation
    center = [cavity_size[0]/2, cavity_size[1]/2, cavity_size[2]/2]
    exc.AddBox(
        [center[0]-dx, center[1]-dy, center[2]-dz],
        [center[0]+dx, center[1]+dy, center[2]+dz]
    )

    # Add a voltage probe
    probe = CSX.AddProbe('V_probe', p_type=0)
    probe.AddBox([center[0], center[1], center[2]], [center[0], center[1], center[2]])

    return FDTD, sim_path, grid_size**3

def run_benchmark(grid_size=100, num_timesteps=5000, engine='fastest', verbose=True):
    """Run the benchmark and return timing information."""

    FDTD, sim_path, num_cells = create_benchmark_simulation(grid_size, num_timesteps, engine)

    if verbose:
        print(f"Benchmark Configuration:")
        print(f"  Grid size: {grid_size}x{grid_size}x{grid_size} = {num_cells:,} cells")
        print(f"  Timesteps: {num_timesteps:,}")
        print(f"  Engine: {engine}")
        print(f"  Simulation path: {sim_path}")
        print()

    # Run simulation
    start_time = time.perf_counter()
    FDTD.Run(sim_path, cleanup=True, verbose=3 if verbose else 0,
             debug_pec=False, debug_material=False, debug_boxes=False,
             engine=engine)
    end_time = time.perf_counter()

    elapsed = end_time - start_time
    cells_per_second = (num_cells * num_timesteps) / elapsed

    if verbose:
        print(f"\nResults:")
        print(f"  Total time: {elapsed:.2f} seconds")
        print(f"  Performance: {cells_per_second/1e6:.2f} Mcells/s")
        print(f"  Per timestep: {elapsed/num_timesteps*1000:.3f} ms")

    return {
        'elapsed': elapsed,
        'num_cells': num_cells,
        'num_timesteps': num_timesteps,
        'cells_per_second': cells_per_second,
        'engine': engine
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OpenEMS Benchmark')
    parser.add_argument('--grid-size', type=int, default=80,
                        help='Grid cells per dimension (default: 80)')
    parser.add_argument('--timesteps', type=int, default=2000,
                        help='Number of timesteps (default: 2000)')
    parser.add_argument('--engine', type=str, default='fastest',
                        choices=['basic', 'sse', 'sse-compressed', 'multithreaded', 'fastest', 'hwy'],
                        help='Engine type (default: fastest)')

    args = parser.parse_args()

    results = run_benchmark(args.grid_size, args.timesteps, args.engine)
