# API Reference

This section provides detailed documentation for all classes, functions, and constants in the openEMS Python API.

## Module Overview

```python
import openems

# Main classes
openems.Grid       # Computational grid definition
openems.OpenEMS    # Simulation controller

# Physical constants
openems.C0         # Speed of light (m/s)
openems.EPS0       # Permittivity of free space (F/m)
openems.MU0        # Permeability of free space (H/m)
openems.Z0         # Impedance of free space (Ohm)

# Version information
openems.VERSION    # Library version string
```

## Classes

### [Grid](grid.md)

The `Grid` class defines the spatial discretization for FDTD simulations.

```python
# Create a uniform grid
grid = openems.Grid.uniform(nx=100, ny=100, nz=50, delta=0.5e-3)

# Create a non-uniform grid
grid = openems.Grid.from_lines(x_coords, y_coords, z_coords)

# Query grid properties
total_cells = grid.num_cells()
dx, dy, dz = grid.cell_size()
```

### [OpenEMS](openems.md)

The `OpenEMS` class is the main simulation controller.

```python
# Create simulation
sim = openems.OpenEMS(num_timesteps=10000)

# Configure simulation
sim.set_grid(grid)
sim.set_gauss_excite(f0=2.4e9, fc=0.5e9)
sim.set_boundary_cond([3, 3, 3, 3, 3, 3])
sim.set_end_criteria(-40)

# Run simulation
sim.run('./output_path', cleanup=True)
```

## Constants

### [Physical Constants](constants.md)

Fundamental physical constants for electromagnetic simulations:

| Constant | Value | Unit | Description |
|----------|-------|------|-------------|
| `C0` | 299,792,458 | m/s | Speed of light in vacuum |
| `EPS0` | 8.854187817e-12 | F/m | Permittivity of free space |
| `MU0` | 1.2566370614e-6 | H/m | Permeability of free space |
| `Z0` | 376.730313668 | Ohm | Impedance of free space |

## Quick Reference

### Creating Grids

```python
# Uniform grid: equal cell sizes everywhere
grid = openems.Grid.uniform(100, 100, 50, delta=1e-3)

# Non-uniform grid: variable cell sizes
import numpy as np
x = np.linspace(0, 0.1, 101)
y = np.linspace(0, 0.05, 51)
z = np.linspace(0, 0.02, 21)
grid = openems.Grid.from_lines(x.tolist(), y.tolist(), z.tolist())
```

### Boundary Conditions

```python
# Boundary condition codes
PEC = 0  # Perfect Electric Conductor
PMC = 1  # Perfect Magnetic Conductor
MUR = 2  # First-order absorbing
PML = 3  # Perfectly Matched Layer

# Set boundaries: [x_min, x_max, y_min, y_max, z_min, z_max]
sim.set_boundary_cond([PML, PML, PML, PML, PML, PML])  # Open space
sim.set_boundary_cond([PML, PML, PEC, PEC, PEC, PEC])  # Waveguide
sim.set_boundary_cond([PMC, PML, PML, PML, PML, PML])  # Symmetry at x_min
```

### Excitation Setup

```python
# Gaussian pulse excitation
# f0: center frequency, fc: bandwidth
sim.set_gauss_excite(f0=5e9, fc=3e9)  # 5 GHz center, 3 GHz bandwidth

# This provides coverage from (f0 - fc) to (f0 + fc)
# In this case: 2 GHz to 8 GHz
```

### Running Simulations

```python
# Run with fixed timesteps
sim = openems.OpenEMS(num_timesteps=20000)
sim.run('./output', cleanup=False)

# Run until energy decay
sim = openems.OpenEMS(num_timesteps=100000)
sim.set_end_criteria(-40)  # Stop at 40 dB decay
sim.run('./output', cleanup=True)
```

## Error Handling

openEMS raises `RuntimeError` for common issues:

```python
import openems

sim = openems.OpenEMS()

# Error: Grid not set
try:
    sim.run('./output')
except RuntimeError as e:
    print(f"Error: {e}")
    # Output: Error: Grid not set. Call set_grid() first.

# Error: Invalid boundary conditions
try:
    sim.set_boundary_cond([0, 0, 0])  # Wrong length
except RuntimeError as e:
    print(f"Error: {e}")
    # Output: Error: Boundary conditions must have exactly 6 values
```
