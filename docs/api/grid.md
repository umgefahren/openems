# Grid

The `Grid` class defines the computational domain for FDTD simulations.

## Overview

In the FDTD method, space is discretized into a grid of cells (Yee cells). The `Grid` class supports:

- **Uniform grids**: Equal cell sizes in all directions
- **Non-uniform grids**: Variable cell sizes for adaptive resolution

## Class Reference

::: openems.Grid
    options:
      show_root_heading: true
      show_source: false
      members:
        - uniform
        - from_lines
        - num_cells
        - cell_size

## Creating Grids

### Uniform Grid

Create a grid with equal cell sizes:

```python
import openems

# Parameters:
# - nx, ny, nz: Number of cells in each direction
# - delta: Cell size in meters

grid = openems.Grid.uniform(
    nx=100,     # 100 cells in x
    ny=100,     # 100 cells in y
    nz=50,      # 50 cells in z
    delta=0.5e-3  # 0.5 mm cell size
)

print(f"Total cells: {grid.num_cells()}")
# Output: Total cells: 500000

dx, dy, dz = grid.cell_size()
print(f"Cell size: {dx*1000} x {dy*1000} x {dz*1000} mm")
# Output: Cell size: 0.5 x 0.5 x 0.5 mm
```

### Non-Uniform Grid

Create a grid with variable cell sizes:

```python
import numpy as np
import openems

# Define mesh lines (coordinates of cell boundaries)
# Finer mesh in the center, coarser at edges

# X-direction: fine in center
x_center = np.linspace(-0.01, 0.01, 41)  # 0.5mm cells
x_left = np.linspace(-0.05, -0.01, 9)[:-1]  # 5mm cells
x_right = np.linspace(0.01, 0.05, 9)[1:]  # 5mm cells
x = np.concatenate([x_left, x_center, x_right])

# Y-direction: same as X
y = x.copy()

# Z-direction: uniform
z = np.linspace(-0.02, 0.02, 41)  # 1mm cells

grid = openems.Grid.from_lines(
    x.tolist(),
    y.tolist(),
    z.tolist()
)

print(f"Total cells: {grid.num_cells()}")
dx, dy, dz = grid.cell_size()
print(f"Minimum cell size: {dx*1000:.2f} x {dy*1000:.2f} x {dz*1000:.2f} mm")
```

## Grid Size Guidelines

### Frequency-Based Sizing

The cell size should be at least λ/10 at the highest frequency:

```python
import openems

def calculate_cell_size(f_max: float, cells_per_wavelength: int = 10) -> float:
    """Calculate appropriate cell size for a given maximum frequency.

    Parameters
    ----------
    f_max : float
        Maximum frequency in Hz
    cells_per_wavelength : int
        Number of cells per wavelength (default: 10)

    Returns
    -------
    float
        Cell size in meters
    """
    wavelength = openems.C0 / f_max
    return wavelength / cells_per_wavelength

# Example: 10 GHz simulation
f_max = 10e9
delta = calculate_cell_size(f_max)
print(f"Cell size for {f_max/1e9} GHz: {delta*1000:.2f} mm")
# Output: Cell size for 10.0 GHz: 3.00 mm

# For higher accuracy, use 15-20 cells per wavelength
delta_fine = calculate_cell_size(f_max, cells_per_wavelength=20)
print(f"Fine cell size: {delta_fine*1000:.2f} mm")
# Output: Fine cell size: 1.50 mm
```

### Memory Estimation

Estimate memory requirements:

```python
import openems

def estimate_memory(grid: openems.Grid, bytes_per_cell: int = 100) -> float:
    """Estimate memory usage for a simulation.

    Parameters
    ----------
    grid : openems.Grid
        The computational grid
    bytes_per_cell : int
        Approximate bytes per cell (default: 100 for basic simulation)

    Returns
    -------
    float
        Estimated memory in GB
    """
    return grid.num_cells() * bytes_per_cell / (1024**3)

# Example
grid = openems.Grid.uniform(200, 200, 100, delta=1e-3)
memory_gb = estimate_memory(grid)
print(f"Cells: {grid.num_cells():,}")
print(f"Estimated memory: {memory_gb:.2f} GB")
# Output:
# Cells: 4,000,000
# Estimated memory: 0.37 GB
```

## Advanced Meshing

### Adaptive Mesh for Fine Features

Create finer mesh around small features:

```python
import numpy as np
import openems

def create_adaptive_mesh(
    bounds: tuple,
    fine_region: tuple,
    coarse_cell: float,
    fine_cell: float
) -> list:
    """Create 1D adaptive mesh with fine region.

    Parameters
    ----------
    bounds : tuple
        (min, max) bounds of the mesh
    fine_region : tuple
        (min, max) bounds of the fine region
    coarse_cell : float
        Cell size in coarse regions
    fine_cell : float
        Cell size in fine region

    Returns
    -------
    list
        Mesh line coordinates
    """
    mesh = []

    # Coarse region before fine region
    if bounds[0] < fine_region[0]:
        n_coarse = int((fine_region[0] - bounds[0]) / coarse_cell)
        mesh.extend(np.linspace(bounds[0], fine_region[0], n_coarse + 1)[:-1])

    # Fine region
    n_fine = int((fine_region[1] - fine_region[0]) / fine_cell)
    mesh.extend(np.linspace(fine_region[0], fine_region[1], n_fine + 1))

    # Coarse region after fine region
    if bounds[1] > fine_region[1]:
        n_coarse = int((bounds[1] - fine_region[1]) / coarse_cell)
        mesh.extend(np.linspace(fine_region[1], bounds[1], n_coarse + 1)[1:])

    return mesh

# Example: Fine mesh around a small antenna
x = create_adaptive_mesh(
    bounds=(-0.1, 0.1),      # -100mm to 100mm total
    fine_region=(-0.02, 0.02),  # Fine around center
    coarse_cell=5e-3,        # 5mm coarse cells
    fine_cell=0.5e-3         # 0.5mm fine cells
)

print(f"Mesh lines: {len(x)}")
print(f"Min cell: {min(np.diff(x))*1000:.2f} mm")
print(f"Max cell: {max(np.diff(x))*1000:.2f} mm")
```

### Smooth Mesh Transition

For best accuracy, cell sizes should change gradually:

```python
import numpy as np

def smooth_mesh(start: float, end: float, start_cell: float, end_cell: float, ratio: float = 1.3) -> list:
    """Create mesh with smoothly varying cell size.

    Parameters
    ----------
    start, end : float
        Start and end coordinates
    start_cell : float
        Cell size at start
    end_cell : float
        Cell size at end
    ratio : float
        Maximum ratio between adjacent cells (default: 1.3)

    Returns
    -------
    list
        Mesh line coordinates
    """
    mesh = [start]
    cell = start_cell
    pos = start

    while pos < end:
        pos += cell
        if pos > end:
            pos = end
        mesh.append(pos)

        # Gradually change cell size
        if cell < end_cell:
            cell = min(cell * ratio, end_cell)
        elif cell > end_cell:
            cell = max(cell / ratio, end_cell)

    return mesh

# Example: Transition from 0.5mm to 2mm cells
x = smooth_mesh(0, 0.05, 0.5e-3, 2e-3)
cells = np.diff(x)
print(f"Cell sizes: {cells*1000}")
```

## Best Practices

1. **Start coarse, refine as needed**: Begin with a coarse mesh, then refine areas with high field gradients

2. **Match structure features**: Cell boundaries should align with material interfaces

3. **Limit cell size ratios**: Adjacent cells should differ by at most 30% (ratio < 1.3)

4. **Add PML space**: Include extra cells (8-16) for PML boundary layers

5. **Check convergence**: Verify results don't change significantly with finer mesh
