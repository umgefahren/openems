# Quick Start Guide

This guide will walk you through your first electromagnetic simulation with openEMS.

## Overview

An openEMS simulation consists of several key steps:

1. **Define the computational grid** - The spatial discretization of your problem
2. **Set up the excitation** - The electromagnetic source that drives the simulation
3. **Configure boundary conditions** - How fields behave at domain edges
4. **Run the simulation** - Execute the FDTD time-stepping

## Your First Simulation

Let's create a simple simulation of electromagnetic waves in free space.

```python
import openems

# Step 1: Create the computational grid
# 50x50x50 cells with 1mm resolution
grid = openems.Grid.uniform(50, 50, 50, delta=1e-3)

print(f"Created grid with {grid.num_cells()} cells")
# Output: Created grid with 125000 cells

# Step 2: Initialize the simulation
sim = openems.OpenEMS(num_timesteps=10000)
sim.set_grid(grid)

# Step 3: Configure excitation
# Gaussian pulse centered at 5 GHz with 3 GHz bandwidth
sim.set_gauss_excite(5e9, 3e9)

# Step 4: Set boundary conditions
# PML (absorbing) on all sides
sim.set_boundary_cond([3, 3, 3, 3, 3, 3])

# Step 5: Run the simulation
sim.run('./my_first_simulation', cleanup=True)
```

## Understanding the Grid

The computational grid divides space into small cells (Yee cells). The cell size determines:

- **Maximum frequency**: Smaller cells allow higher frequencies
- **Simulation accuracy**: Smaller cells give more accurate results
- **Memory and time**: Smaller cells require more resources

### Rule of thumb
Use at least 10 cells per wavelength at your highest frequency:

```python
import openems

# Target frequency: 10 GHz
f_max = 10e9

# Calculate wavelength
wavelength = openems.C0 / f_max  # ~30 mm

# Cell size: wavelength / 10 = 3 mm
delta = wavelength / 10

print(f"Wavelength at {f_max/1e9} GHz: {wavelength*1000:.1f} mm")
print(f"Recommended cell size: {delta*1000:.1f} mm")

# Create grid
grid = openems.Grid.uniform(100, 100, 50, delta=delta)
```

## Understanding Boundary Conditions

Boundary conditions define what happens to electromagnetic waves at the edges of your simulation domain:

| Code | Type | Description | Use Case |
|------|------|-------------|----------|
| 0 | PEC | Perfect Electric Conductor | Metal walls, ground planes |
| 1 | PMC | Perfect Magnetic Conductor | Symmetry planes, magnetic walls |
| 2 | MUR | First-order absorbing | Quick tests, low accuracy |
| 3 | PML | Perfectly Matched Layer | Open boundaries, best absorption |

### Example: Waveguide Boundaries

```python
# Rectangular waveguide: metal walls (PEC) on sides, PML at ports
# [x_min, x_max, y_min, y_max, z_min, z_max]
sim.set_boundary_cond([3, 3, 0, 0, 0, 0])
#                      |  |  |  |  |  |
#                      |  |  |  |  |  └─ z_max: PEC (top wall)
#                      |  |  |  |  └──── z_min: PEC (bottom wall)
#                      |  |  |  └─────── y_max: PEC (side wall)
#                      |  |  └────────── y_min: PEC (side wall)
#                      |  └───────────── x_max: PML (output port)
#                      └──────────────── x_min: PML (input port)
```

## Understanding Excitation

The Gaussian excitation provides a broadband signal for frequency-domain analysis:

```python
# Define frequency range
f_start = 1e9   # 1 GHz
f_stop = 10e9   # 10 GHz

# Calculate center frequency and bandwidth
f0 = (f_start + f_stop) / 2  # 5.5 GHz
fc = (f_stop - f_start) / 2  # 4.5 GHz

sim.set_gauss_excite(f0, fc)
```

The resulting pulse:
- Has significant energy from `f0 - fc` to `f0 + fc`
- Is a modulated Gaussian in the time domain
- Decays to near-zero, allowing accurate FFT analysis

## Non-Uniform Grids

For structures with fine features, use a non-uniform grid:

```python
import numpy as np
import openems

# Fine mesh in the center, coarser towards edges
x_fine = np.linspace(-0.01, 0.01, 41)    # -10mm to 10mm, 0.5mm cells
x_left = np.linspace(-0.05, -0.01, 9)[:-1]  # Coarser left side
x_right = np.linspace(0.01, 0.05, 9)[1:]    # Coarser right side

x = np.concatenate([x_left, x_fine, x_right])
y = x.copy()  # Same for y
z = np.linspace(-0.02, 0.02, 21)  # Uniform in z

grid = openems.Grid.from_lines(x.tolist(), y.tolist(), z.tolist())

print(f"Non-uniform grid: {grid.num_cells()} cells")
dx, dy, dz = grid.cell_size()
print(f"Minimum cell size: {dx*1000:.2f} x {dy*1000:.2f} x {dz*1000:.2f} mm")
```

## Using End Criteria

Instead of running a fixed number of timesteps, you can stop when the field energy has decayed:

```python
sim = openems.OpenEMS(num_timesteps=100000)  # Maximum timesteps
sim.set_end_criteria(-40)  # Stop at 40 dB energy decay

# Simulation will stop when:
# - 100000 timesteps are reached, OR
# - Field energy drops to 1/10000 of peak (40 dB)
```

## Next Steps

Now that you understand the basics, explore:

- [Waveguide Example](../examples/waveguide.md) - Simulate a rectangular waveguide
- [Patch Antenna Example](../examples/patch_antenna.md) - Design a microstrip patch antenna
- [API Reference](../api/index.md) - Detailed documentation of all classes and methods
