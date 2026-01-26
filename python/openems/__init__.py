"""
openEMS - High-Performance FDTD Electromagnetic Field Solver
=============================================================

openEMS is a free and open-source electromagnetic field solver using the
Finite-Difference Time-Domain (FDTD) method. This Python package provides
bindings to the high-performance Rust implementation.

Quick Start
-----------
>>> import openems
>>> # Create a uniform grid
>>> grid = openems.Grid.uniform(100, 100, 50, delta=0.5e-3)
>>> print(f"Grid has {grid.num_cells()} cells")
Grid has 500000 cells

>>> # Create and configure simulation
>>> sim = openems.OpenEMS(num_timesteps=10000)
>>> sim.set_grid(grid)
>>> sim.set_boundary_cond([0, 0, 0, 0, 3, 3])  # PEC sides, PML z
>>> sim.set_gauss_excite(2.4e9, 0.5e9)  # 2.4 GHz center, 500 MHz bandwidth

Classes
-------
Grid
    Computational grid for FDTD simulation
OpenEMS
    Main simulation controller

Constants
---------
C0 : float
    Speed of light in vacuum (299792458 m/s)
EPS0 : float
    Permittivity of free space (8.854187817e-12 F/m)
MU0 : float
    Permeability of free space (4*pi*1e-7 H/m)
Z0 : float
    Impedance of free space (~376.73 Ohm)
VERSION : str
    Library version string

Examples
--------
Basic simulation setup:

>>> import openems
>>> import numpy as np
>>>
>>> # Define simulation parameters
>>> f0 = 2.4e9  # Center frequency (2.4 GHz)
>>> fc = 0.5e9  # Bandwidth (500 MHz)
>>>
>>> # Create computational grid
>>> grid = openems.Grid.uniform(100, 100, 50, delta=0.5e-3)
>>>
>>> # Initialize simulation
>>> sim = openems.OpenEMS(num_timesteps=20000)
>>> sim.set_grid(grid)
>>> sim.set_gauss_excite(f0, fc)
>>> sim.set_boundary_cond([0, 0, 0, 0, 3, 3])
>>>
>>> # Run simulation
>>> # sim.run('./sim_output', cleanup=True)

See Also
--------
- Project homepage: https://openems.de
- GitHub repository: https://github.com/thliebig/openEMS
- Documentation: https://docs.openems.de

Notes
-----
This is the Rust-based implementation of openEMS, providing improved
performance and cross-platform support compared to the original C++ version.
"""

from __future__ import annotations

# Import from Rust extension module
from ._rust import (
    Grid,
    OpenEMS,
    C0,
    EPS0,
    MU0,
    Z0,
    VERSION,
)

__all__ = [
    # Classes
    "Grid",
    "OpenEMS",
    # Constants
    "C0",
    "EPS0",
    "MU0",
    "Z0",
    "VERSION",
    # Submodules
    "constants",
]

__version__ = VERSION


class constants:
    """
    Physical constants for electromagnetic simulations.

    This class provides easy access to fundamental physical constants
    used in electromagnetic simulations.

    Attributes
    ----------
    C0 : float
        Speed of light in vacuum (299792458 m/s)
    EPS0 : float
        Permittivity of free space (8.854187817e-12 F/m)
    MU0 : float
        Permeability of free space (4*pi*1e-7 H/m)
    Z0 : float
        Impedance of free space (~376.73 Ohm)

    Examples
    --------
    >>> import openems
    >>> print(f"Speed of light: {openems.constants.C0:.0f} m/s")
    Speed of light: 299792458 m/s
    >>> print(f"Free space impedance: {openems.constants.Z0:.2f} Ohm")
    Free space impedance: 376.73 Ohm

    Calculate wavelength at a given frequency:

    >>> freq = 2.4e9  # 2.4 GHz
    >>> wavelength = openems.constants.C0 / freq
    >>> print(f"Wavelength at {freq/1e9} GHz: {wavelength*1000:.1f} mm")
    Wavelength at 2.4 GHz: 124.9 mm
    """

    C0 = C0
    EPS0 = EPS0
    MU0 = MU0
    Z0 = Z0
