"""
Type stubs for the openEMS Rust extension module.

This module provides type hints for the compiled Rust extension.
"""

from __future__ import annotations
from typing import List, Tuple

# Physical constants
C0: float
"""Speed of light in vacuum (299792458 m/s)."""

EPS0: float
"""Permittivity of free space (8.854187817e-12 F/m)."""

MU0: float
"""Permeability of free space (4*pi*1e-7 H/m)."""

Z0: float
"""Impedance of free space (~376.73 Ohm)."""

VERSION: str
"""Library version string."""

class Grid:
    """
    Computational grid for FDTD simulation.

    The Grid class defines the spatial discretization for the FDTD simulation.
    It supports uniform grids and non-uniform grids defined by mesh lines.

    The grid uses a Yee cell arrangement where electric and magnetic field
    components are staggered in space.

    Examples
    --------
    Create a uniform grid with 100x100x50 cells and 0.5mm resolution:

    >>> import openems
    >>> grid = openems.Grid.uniform(100, 100, 50, delta=0.5e-3)
    >>> print(f"Total cells: {grid.num_cells()}")
    Total cells: 500000

    Create a non-uniform grid from mesh lines:

    >>> import numpy as np
    >>> x = np.linspace(0, 0.1, 101)  # 0 to 100mm, 100 cells
    >>> y = np.linspace(0, 0.05, 51)  # 0 to 50mm, 50 cells
    >>> z = np.linspace(0, 0.02, 21)  # 0 to 20mm, 20 cells
    >>> grid = openems.Grid.from_lines(x.tolist(), y.tolist(), z.tolist())

    Notes
    -----
    - Grid dimensions are in meters
    - For accurate simulations, cell size should be at least lambda/10
      where lambda is the wavelength at the highest frequency of interest
    - Non-uniform grids allow finer resolution in areas of interest
    """

    @staticmethod
    def uniform(nx: int, ny: int, nz: int, delta: float) -> Grid:
        """
        Create a uniform Cartesian grid.

        Creates a grid with equal cell sizes in all directions. This is the
        simplest grid type and is suitable for many simulations.

        Parameters
        ----------
        nx : int
            Number of cells in x-direction
        ny : int
            Number of cells in y-direction
        nz : int
            Number of cells in z-direction
        delta : float
            Cell size in all directions (meters)

        Returns
        -------
        Grid
            A new uniform grid object

        Examples
        --------
        Create a 50x50x25 cell grid with 1mm cell size:

        >>> grid = openems.Grid.uniform(50, 50, 25, delta=1e-3)
        >>> print(grid.num_cells())
        62500

        For a simulation at 10 GHz (lambda = 30mm), use cells <= 3mm:

        >>> freq = 10e9  # 10 GHz
        >>> wavelength = openems.C0 / freq  # ~30mm
        >>> delta = wavelength / 15  # ~2mm cells for safety margin
        >>> grid = openems.Grid.uniform(100, 100, 50, delta=delta)

        See Also
        --------
        from_lines : Create a non-uniform grid from mesh lines
        """
        ...

    @staticmethod
    def from_lines(x: List[float], y: List[float], z: List[float]) -> Grid:
        """
        Create a grid from mesh line arrays.

        Creates a non-uniform grid where cell boundaries are defined by
        the provided coordinate arrays. This allows variable cell sizes
        for better resolution in specific regions.

        Parameters
        ----------
        x : list of float
            X-coordinates of mesh lines (meters), must be monotonically increasing
        y : list of float
            Y-coordinates of mesh lines (meters), must be monotonically increasing
        z : list of float
            Z-coordinates of mesh lines (meters), must be monotonically increasing

        Returns
        -------
        Grid
            A new non-uniform grid object

        Examples
        --------
        Create a grid with finer resolution near the origin:

        >>> import numpy as np
        >>> # Fine mesh near origin, coarser further out
        >>> x_fine = np.linspace(0, 0.01, 21)    # 0-10mm, 0.5mm cells
        >>> x_coarse = np.linspace(0.01, 0.05, 9)[1:]  # 10-50mm, 5mm cells
        >>> x = np.concatenate([x_fine, x_coarse])
        >>> y = x.copy()  # Same for y
        >>> z = np.linspace(0, 0.02, 41)  # Uniform in z
        >>> grid = openems.Grid.from_lines(x.tolist(), y.tolist(), z.tolist())

        Notes
        -----
        - Arrays must have at least 2 elements each
        - The number of cells is len(array) - 1 in each direction
        - Mesh lines should be sorted in ascending order

        See Also
        --------
        uniform : Create a uniform grid with equal cell sizes
        """
        ...

    def num_cells(self) -> int:
        """
        Get the total number of cells in the grid.

        Returns the product of cells in each direction (nx * ny * nz).

        Returns
        -------
        int
            Total number of computational cells

        Examples
        --------
        >>> grid = openems.Grid.uniform(100, 50, 25, delta=1e-3)
        >>> print(f"Total cells: {grid.num_cells()}")
        Total cells: 125000

        Notes
        -----
        Simulation memory usage and computation time scale linearly with
        the number of cells. A typical desktop can handle 10-100 million cells.
        """
        ...

    def cell_size(self) -> Tuple[float, float, float]:
        """
        Get the minimum cell size in each direction.

        For uniform grids, this returns the cell size in each direction.
        For non-uniform grids, this returns the smallest cell size in each
        direction, which determines the time step limit.

        Returns
        -------
        tuple of float
            Minimum cell sizes (dx, dy, dz) in meters

        Examples
        --------
        >>> grid = openems.Grid.uniform(100, 100, 50, delta=0.5e-3)
        >>> dx, dy, dz = grid.cell_size()
        >>> print(f"Cell size: {dx*1000:.2f} x {dy*1000:.2f} x {dz*1000:.2f} mm")
        Cell size: 0.50 x 0.50 x 0.50 mm

        Notes
        -----
        The minimum cell size determines the maximum stable time step
        according to the Courant-Friedrichs-Lewy (CFL) condition:

            dt <= 1 / (c * sqrt(1/dx^2 + 1/dy^2 + 1/dz^2))

        where c is the speed of light.
        """
        ...


class OpenEMS:
    """
    Main FDTD simulation controller.

    The OpenEMS class manages the complete FDTD electromagnetic simulation,
    including grid setup, excitation configuration, boundary conditions,
    and simulation execution.

    Parameters
    ----------
    num_timesteps : int, optional
        Maximum number of time steps to simulate (default: 10000)

    Attributes
    ----------
    num_timesteps : int
        Maximum simulation time steps

    Examples
    --------
    Basic simulation setup:

    >>> import openems
    >>>
    >>> # Create simulation with 20000 max timesteps
    >>> sim = openems.OpenEMS(num_timesteps=20000)
    >>>
    >>> # Set up computational grid
    >>> grid = openems.Grid.uniform(100, 100, 50, delta=0.5e-3)
    >>> sim.set_grid(grid)
    >>>
    >>> # Configure excitation (Gaussian pulse)
    >>> f0 = 2.4e9   # Center frequency: 2.4 GHz
    >>> fc = 0.5e9   # Bandwidth: 500 MHz
    >>> sim.set_gauss_excite(f0, fc)
    >>>
    >>> # Set boundary conditions
    >>> # [xmin, xmax, ymin, ymax, zmin, zmax]
    >>> # 0=PEC, 1=PMC, 2=MUR, 3=PML
    >>> sim.set_boundary_cond([0, 0, 0, 0, 3, 3])
    >>>
    >>> # Run simulation
    >>> # sim.run('./sim_output', cleanup=True)

    Complete waveguide simulation example:

    >>> import openems
    >>> import numpy as np
    >>>
    >>> # Waveguide dimensions (WR-90: 22.86 x 10.16 mm)
    >>> a = 22.86e-3  # Width
    >>> b = 10.16e-3  # Height
    >>> length = 100e-3  # 100mm length
    >>>
    >>> # Frequency range (X-band: 8.2-12.4 GHz)
    >>> f_start = 8.2e9
    >>> f_stop = 12.4e9
    >>> f0 = (f_start + f_stop) / 2
    >>> fc = (f_stop - f_start) / 2
    >>>
    >>> # Create grid with appropriate resolution
    >>> wavelength = openems.C0 / f_stop
    >>> delta = wavelength / 20
    >>> nx = int(length / delta)
    >>> ny = int(a / delta)
    >>> nz = int(b / delta)
    >>> grid = openems.Grid.uniform(nx, ny, nz, delta=delta)
    >>>
    >>> # Set up simulation
    >>> sim = openems.OpenEMS(num_timesteps=50000)
    >>> sim.set_grid(grid)
    >>> sim.set_gauss_excite(f0, fc)
    >>> sim.set_boundary_cond([3, 3, 0, 0, 0, 0])  # PML at ports, PEC walls

    Notes
    -----
    - The FDTD method updates electromagnetic fields in the time domain
    - Simulation automatically calculates stable time step from grid
    - Results are typically post-processed using FFT for frequency domain analysis
    - Memory usage: approximately 100 bytes per cell for basic simulation

    See Also
    --------
    Grid : Computational grid definition
    """

    def __init__(self, num_timesteps: int = 10000) -> None:
        """
        Create a new openEMS simulation.

        Parameters
        ----------
        num_timesteps : int, optional
            Maximum number of time steps (default: 10000)

        Examples
        --------
        >>> sim = openems.OpenEMS()  # Default 10000 timesteps
        >>> sim = openems.OpenEMS(num_timesteps=50000)  # Custom timesteps
        """
        ...

    def set_grid(self, grid: Grid) -> None:
        """
        Set the computational grid for the simulation.

        The grid defines the spatial discretization and must be set before
        running the simulation.

        Parameters
        ----------
        grid : Grid
            The computational grid object

        Raises
        ------
        TypeError
            If grid is not a Grid object

        Examples
        --------
        >>> grid = openems.Grid.uniform(100, 100, 50, delta=1e-3)
        >>> sim = openems.OpenEMS()
        >>> sim.set_grid(grid)

        Notes
        -----
        The grid cannot be changed after simulation starts. Create a new
        OpenEMS instance if you need a different grid.
        """
        ...

    def set_gauss_excite(self, f0: float, fc: float) -> None:
        """
        Set Gaussian pulse excitation.

        Configures a Gaussian-modulated sinusoidal excitation signal.
        The pulse is centered at frequency f0 with bandwidth determined by fc.

        Parameters
        ----------
        f0 : float
            Center frequency in Hz
        fc : float
            Cutoff frequency (bandwidth) in Hz. The -20dB bandwidth is
            approximately 2*fc.

        Examples
        --------
        Excitation for 2.4 GHz WiFi band:

        >>> sim = openems.OpenEMS()
        >>> sim.set_gauss_excite(2.4e9, 0.5e9)  # 2.4 GHz +/- 500 MHz

        Broadband excitation for antenna characterization:

        >>> # Cover 1-10 GHz range
        >>> f0 = 5.5e9   # Center at 5.5 GHz
        >>> fc = 4.5e9   # 4.5 GHz bandwidth covers 1-10 GHz
        >>> sim.set_gauss_excite(f0, fc)

        Notes
        -----
        The Gaussian pulse in time domain is:

            E(t) = exp(-((t-t0)*fc)^2) * sin(2*pi*f0*t)

        The frequency spectrum is approximately flat from (f0-fc) to (f0+fc).

        For accurate frequency domain results, ensure:
        - The simulation runs long enough for the pulse to decay
        - The grid resolution supports the highest frequency: delta < c/(10*fmax)
        """
        ...

    def set_boundary_cond(self, bc: List[int]) -> None:
        """
        Set boundary conditions for the simulation domain.

        Defines how electromagnetic fields behave at the edges of the
        computational domain.

        Parameters
        ----------
        bc : list of int
            Six boundary condition codes in order:
            [x_min, x_max, y_min, y_max, z_min, z_max]

            Condition codes:
            - 0: PEC (Perfect Electric Conductor) - reflects waves, E_tangential = 0
            - 1: PMC (Perfect Magnetic Conductor) - reflects waves, H_tangential = 0
            - 2: MUR (1st order absorbing) - simple absorption, some reflection
            - 3: PML (Perfectly Matched Layer) - best absorption, minimal reflection

        Examples
        --------
        Open boundaries (radiating structure):

        >>> sim.set_boundary_cond([3, 3, 3, 3, 3, 3])  # PML all around

        Waveguide with metallic walls:

        >>> # PML at input/output (x), PEC walls (y, z)
        >>> sim.set_boundary_cond([3, 3, 0, 0, 0, 0])

        Symmetric structure (use PMC for magnetic symmetry):

        >>> # PMC at x_min (symmetry plane), PML elsewhere
        >>> sim.set_boundary_cond([1, 3, 3, 3, 3, 3])

        Microstrip over ground plane:

        >>> # PML top and sides, PEC ground at z_min
        >>> sim.set_boundary_cond([3, 3, 3, 3, 0, 3])

        Notes
        -----
        - PML provides the best absorption but requires 8-16 additional cells
        - Use PEC/PMC for physical metal boundaries or symmetry planes
        - MUR is a lightweight absorber, useful for quick tests
        - Symmetry boundaries (PEC/PMC) can reduce simulation size by 2-8x

        See Also
        --------
        set_gauss_excite : Configure excitation signal
        """
        ...

    def run(self, sim_path: str, cleanup: bool = False) -> None:
        """
        Run the FDTD simulation.

        Executes the time-domain simulation with the configured parameters.
        Results are stored in the specified directory.

        Parameters
        ----------
        sim_path : str
            Path to directory for simulation files and results
        cleanup : bool, optional
            If True, remove temporary files after simulation (default: False)

        Raises
        ------
        RuntimeError
            If grid is not set or simulation setup fails

        Examples
        --------
        Basic simulation run:

        >>> sim = openems.OpenEMS(num_timesteps=10000)
        >>> grid = openems.Grid.uniform(50, 50, 25, delta=1e-3)
        >>> sim.set_grid(grid)
        >>> sim.set_gauss_excite(2.4e9, 0.5e9)
        >>> sim.set_boundary_cond([3, 3, 3, 3, 3, 3])
        >>> sim.run('./my_simulation', cleanup=False)

        Run with cleanup of temporary files:

        >>> sim.run('./sim_output', cleanup=True)

        Notes
        -----
        - The simulation directory is created if it doesn't exist
        - Progress is displayed during simulation
        - Simulation may terminate early if energy decay threshold is reached
        - Field data can be saved at specified intervals for visualization

        Performance tips:
        - Use release builds for production simulations
        - Enable parallel processing for multi-core systems
        - Consider using symmetry to reduce problem size
        """
        ...
