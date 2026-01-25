#!/usr/bin/env python3
"""
Run simulation and generate XML for profiling with valgrind/perf.
"""
import os
import sys
import tempfile
import subprocess
import numpy as np
from CSXCAD import ContinuousStructure, CSXCAD
from openEMS import openEMS
from openEMS.physical_constants import C0

def create_simulation_xml(output_dir, grid_size=60, num_timesteps=500):
    """Create simulation XML file for profiling with external tools."""

    os.makedirs(output_dir, exist_ok=True)

    unit = 1e-3
    cavity_size = [100, 100, 100]

    dx = cavity_size[0] / grid_size
    dy = cavity_size[1] / grid_size
    dz = cavity_size[2] / grid_size

    f_max = 3e9

    FDTD = openEMS(NrTS=num_timesteps, EndCriteria=0)
    FDTD.SetGaussExcite(f_max/2, f_max/2)
    FDTD.SetBoundaryCond(['PEC', 'PEC', 'PEC', 'PEC', 'PEC', 'PEC'])

    CSX = ContinuousStructure()
    FDTD.SetCSX(CSX)
    mesh = CSX.GetGrid()
    mesh.SetDeltaUnit(unit)

    mesh.AddLine('x', np.linspace(0, cavity_size[0], grid_size + 1))
    mesh.AddLine('y', np.linspace(0, cavity_size[1], grid_size + 1))
    mesh.AddLine('z', np.linspace(0, cavity_size[2], grid_size + 1))

    exc = CSX.AddExcitation('excite', exc_type=0, exc_val=[1, 0, 0])
    center = [cavity_size[0]/2, cavity_size[1]/2, cavity_size[2]/2]
    exc.AddBox(
        [center[0]-dx, center[1]-dy, center[2]-dz],
        [center[0]+dx, center[1]+dy, center[2]+dz]
    )

    # Write the XML file
    xml_path = os.path.join(output_dir, 'benchmark.xml')
    FDTD.Write2XML(xml_path)

    print(f"Simulation XML created: {xml_path}")
    print(f"Grid: {grid_size}x{grid_size}x{grid_size} = {grid_size**3:,} cells")
    print(f"Timesteps: {num_timesteps}")

    return xml_path

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default='/tmp/openems_profile')
    parser.add_argument('--grid-size', type=int, default=60)
    parser.add_argument('--timesteps', type=int, default=500)
    args = parser.parse_args()

    xml_path = create_simulation_xml(args.output, args.grid_size, args.timesteps)

    print(f"\nTo profile with callgrind:")
    print(f"  LD_LIBRARY_PATH=/home/user/opt/openEMS/lib valgrind --tool=callgrind /home/user/opt/openEMS/bin/openEMS {xml_path} --engine=basic")
    print(f"\nTo profile with perf (requires kernel access):")
    print(f"  LD_LIBRARY_PATH=/home/user/opt/openEMS/lib perf record -g /home/user/opt/openEMS/bin/openEMS {xml_path} --engine=basic")
