# OpenEMS

The `OpenEMS` class is the main simulation controller for FDTD electromagnetic simulations.

## Overview

The `OpenEMS` class manages all aspects of an FDTD simulation:

- Grid configuration
- Excitation setup
- Boundary conditions
- Simulation execution

## Class Reference

::: openems.OpenEMS
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - set_grid
        - set_gauss_excite
        - set_boundary_cond
        - set_end_criteria
        - run

## Basic Usage

### Creating a Simulation

```python
import openems

# Create simulation with default settings (10000 timesteps)
sim = openems.OpenEMS()

# Create simulation with custom timestep limit
sim = openems.OpenEMS(num_timesteps=50000)

# Check simulation state
print(sim)
# Output: OpenEMS(timesteps=50000, grid=not set, excite=not set)
```

### Complete Simulation Setup

```python
import openems

# 1. Create simulation
sim = openems.OpenEMS(num_timesteps=20000)

# 2. Set computational grid
grid = openems.Grid.uniform(100, 100, 50, delta=1e-3)
sim.set_grid(grid)

# 3. Configure excitation
sim.set_gauss_excite(f0=5e9, fc=3e9)  # 5 GHz center, 3 GHz bandwidth

# 4. Set boundary conditions
sim.set_boundary_cond([3, 3, 3, 3, 3, 3])  # PML all around

# 5. Run simulation
sim.run('./simulation_output', cleanup=True)
```

## Excitation Configuration

### Gaussian Pulse

The `set_gauss_excite` method configures a Gaussian-modulated sinusoidal excitation:

```python
import openems
import numpy as np

sim = openems.OpenEMS()

# Define frequency range to cover
f_min = 1e9   # 1 GHz
f_max = 10e9  # 10 GHz

# Calculate excitation parameters
f0 = (f_min + f_max) / 2  # Center frequency: 5.5 GHz
fc = (f_max - f_min) / 2  # Bandwidth: 4.5 GHz

sim.set_gauss_excite(f0, fc)

# The excitation signal in time domain:
# E(t) = exp(-((t-t0)*fc)^2) * sin(2*pi*f0*t)

# Approximate the pulse signal
t = np.linspace(0, 2e-9, 1000)
t0 = 1e-9  # Pulse center time
pulse = np.exp(-((t - t0) * fc)**2) * np.sin(2 * np.pi * f0 * t)
```

### Frequency Coverage

The Gaussian excitation provides coverage from approximately `f0 - fc` to `f0 + fc`:

```python
# Examples for different frequency bands

# WiFi 2.4 GHz band (2.4 - 2.5 GHz)
sim.set_gauss_excite(f0=2.45e9, fc=0.1e9)

# WiFi 5 GHz band (5.15 - 5.85 GHz)
sim.set_gauss_excite(f0=5.5e9, fc=0.5e9)

# X-band radar (8.2 - 12.4 GHz)
sim.set_gauss_excite(f0=10.3e9, fc=2.1e9)

# Broadband antenna characterization (1 - 18 GHz)
sim.set_gauss_excite(f0=9.5e9, fc=8.5e9)
```

## Boundary Conditions

### Boundary Types

| Code | Type | Description |
|------|------|-------------|
| 0 | PEC | Perfect Electric Conductor - total reflection, E_tangential = 0 |
| 1 | PMC | Perfect Magnetic Conductor - total reflection, H_tangential = 0 |
| 2 | MUR | First-order Mur ABC - partial absorption |
| 3 | PML | Perfectly Matched Layer - best absorption |

### Setting Boundaries

```python
import openems

sim = openems.OpenEMS()

# Boundary order: [x_min, x_max, y_min, y_max, z_min, z_max]

# Open space (antenna, RCS)
sim.set_boundary_cond([3, 3, 3, 3, 3, 3])  # PML everywhere

# Waveguide (metal walls, open ports)
sim.set_boundary_cond([3, 3, 0, 0, 0, 0])  # PML at x-ports, PEC walls

# Microstrip (ground plane at z_min)
sim.set_boundary_cond([3, 3, 3, 3, 0, 3])  # PEC ground, PML elsewhere

# Electric symmetry at y=0 plane
sim.set_boundary_cond([3, 3, 0, 3, 3, 3])  # PEC at y_min = symmetry

# Magnetic symmetry at x=0 plane
sim.set_boundary_cond([1, 3, 3, 3, 3, 3])  # PMC at x_min = symmetry
```

### Using Symmetry

Symmetry boundaries can reduce simulation size by 2x, 4x, or 8x:

```python
import openems

# Full simulation: 100x100x100 cells = 1,000,000 cells
grid_full = openems.Grid.uniform(100, 100, 100, delta=1e-3)

# Using one symmetry plane (PMC at x_min)
# Only simulate half the structure: 50x100x100 cells
grid_half = openems.Grid.uniform(50, 100, 100, delta=1e-3)
sim = openems.OpenEMS()
sim.set_grid(grid_half)
sim.set_boundary_cond([1, 3, 3, 3, 3, 3])  # PMC symmetry at x_min

# Using two symmetry planes
# Quarter the structure: 50x50x100 cells
grid_quarter = openems.Grid.uniform(50, 50, 100, delta=1e-3)
sim.set_boundary_cond([1, 3, 1, 3, 3, 3])  # PMC at x_min and y_min
```

## End Criteria

### Energy Decay

Stop simulation when field energy has decayed sufficiently:

```python
import openems

sim = openems.OpenEMS(num_timesteps=100000)  # Maximum timesteps
sim.set_end_criteria(-40)  # Stop at 40 dB decay

# The simulation stops when:
# 10 * log10(current_energy / peak_energy) < -40 dB
# This means energy has dropped to 1/10000 of peak

# Common values:
# -30 dB: Fast, ~99.9% decay, adequate for many cases
# -40 dB: Standard, ~99.99% decay, good accuracy
# -50 dB: High accuracy, ~99.999% decay, for low-loss structures
# -60 dB: Very high accuracy, for resonant structures
```

### Fixed Timesteps

For quick tests or when energy decay is slow:

```python
sim = openems.OpenEMS(num_timesteps=5000)
# Simulation runs exactly 5000 timesteps
# No early termination
```

## Running Simulations

### Basic Run

```python
sim.run('./output_directory')
```

### With Cleanup

Remove temporary files after simulation:

```python
sim.run('./output', cleanup=True)
```

### Error Handling

```python
import openems

sim = openems.OpenEMS()

# Error: grid not set
try:
    sim.run('./output')
except RuntimeError as e:
    print(f"Error: {e}")
    # Error: Grid not set. Call set_grid() first.

# Error: invalid boundary conditions
try:
    sim.set_boundary_cond([0, 0, 0])  # Wrong length
except RuntimeError as e:
    print(f"Error: {e}")
    # Error: Boundary conditions must have exactly 6 values
```

## Complete Examples

### Free Space Propagation

```python
import openems

# Create simulation
sim = openems.OpenEMS(num_timesteps=10000)

# 50x50x100 mm domain with 1mm cells
grid = openems.Grid.uniform(50, 50, 100, delta=1e-3)
sim.set_grid(grid)

# Broadband excitation: 1-10 GHz
sim.set_gauss_excite(5.5e9, 4.5e9)

# Open boundaries
sim.set_boundary_cond([3, 3, 3, 3, 3, 3])

# Run
sim.run('./free_space_sim', cleanup=True)
```

### Rectangular Waveguide

```python
import openems

# WR-90 waveguide: 22.86mm x 10.16mm cross-section
a = 22.86e-3  # Width
b = 10.16e-3  # Height
length = 100e-3  # 100mm length

# X-band: 8.2-12.4 GHz
f0 = 10.3e9
fc = 2.1e9

# Grid: 1mm cells
delta = 1e-3
nx = int(length / delta)
ny = int(a / delta)
nz = int(b / delta)

sim = openems.OpenEMS(num_timesteps=30000)
grid = openems.Grid.uniform(nx, ny, nz, delta=delta)
sim.set_grid(grid)
sim.set_gauss_excite(f0, fc)

# PML at ports (x), PEC walls (y, z)
sim.set_boundary_cond([3, 3, 0, 0, 0, 0])

# Run until 40 dB decay
sim.set_end_criteria(-40)
sim.run('./waveguide_sim', cleanup=True)
```

## Performance Tips

1. **Use symmetry** when applicable to reduce problem size

2. **Start with coarse mesh** for quick iteration, then refine

3. **Use energy decay criteria** to avoid running longer than necessary

4. **Minimize domain size** - use PML close to structures of interest

5. **Check cell count** before running:
   ```python
   print(f"Cells: {grid.num_cells():,}")
   # Cells: 1,000,000
   ```
