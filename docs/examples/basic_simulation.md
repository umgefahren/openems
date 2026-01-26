# Basic Simulation

This example demonstrates a simple electromagnetic simulation of wave propagation in free space.

## Overview

We'll simulate a Gaussian pulse propagating through an empty computational domain with absorbing boundaries. This is the simplest possible FDTD simulation and serves as a foundation for more complex setups.

## Complete Code

```python
"""
Basic FDTD Simulation - Free Space Propagation

This example demonstrates:
- Creating a uniform computational grid
- Configuring Gaussian excitation
- Setting PML boundary conditions
- Running a basic simulation
"""

import openems

def main():
    # ==========================================================================
    # Simulation Parameters
    # ==========================================================================

    # Frequency range
    f_min = 1e9   # 1 GHz
    f_max = 10e9  # 10 GHz

    # Calculate excitation parameters
    f0 = (f_min + f_max) / 2  # Center frequency: 5.5 GHz
    fc = (f_max - f_min) / 2  # Bandwidth: 4.5 GHz

    # Calculate appropriate cell size
    # Rule: at least 10 cells per wavelength at highest frequency
    wavelength_min = openems.C0 / f_max  # ~30 mm at 10 GHz
    delta = wavelength_min / 10  # ~3 mm cells

    print(f"Frequency range: {f_min/1e9:.1f} - {f_max/1e9:.1f} GHz")
    print(f"Center frequency: {f0/1e9:.1f} GHz")
    print(f"Minimum wavelength: {wavelength_min*1000:.1f} mm")
    print(f"Cell size: {delta*1000:.1f} mm")

    # ==========================================================================
    # Grid Setup
    # ==========================================================================

    # Domain size: 150mm x 150mm x 150mm
    domain_size = 0.15  # 150 mm

    # Calculate number of cells
    n_cells = int(domain_size / delta)

    print(f"\nGrid: {n_cells}x{n_cells}x{n_cells} cells")
    print(f"Total cells: {n_cells**3:,}")

    # Create uniform grid
    grid = openems.Grid.uniform(n_cells, n_cells, n_cells, delta=delta)

    # Verify grid properties
    print(f"Grid created: {grid.num_cells():,} cells")
    dx, dy, dz = grid.cell_size()
    print(f"Cell size: {dx*1000:.2f} x {dy*1000:.2f} x {dz*1000:.2f} mm")

    # ==========================================================================
    # Simulation Setup
    # ==========================================================================

    # Create simulation
    sim = openems.OpenEMS(num_timesteps=5000)
    sim.set_grid(grid)

    # Set excitation
    sim.set_gauss_excite(f0, fc)

    # Set boundary conditions - PML on all sides (absorbing)
    # [x_min, x_max, y_min, y_max, z_min, z_max]
    # 3 = PML (Perfectly Matched Layer)
    sim.set_boundary_cond([3, 3, 3, 3, 3, 3])

    # Optional: Set end criteria for energy decay
    sim.set_end_criteria(-30)  # Stop at 30 dB decay

    print("\nSimulation configured:")
    print(f"  Max timesteps: 5000")
    print(f"  End criteria: -30 dB")
    print(f"  Boundaries: PML all around")

    # ==========================================================================
    # Run Simulation
    # ==========================================================================

    print("\nStarting simulation...")
    sim.run('./basic_simulation_output', cleanup=True)
    print("Simulation complete!")


if __name__ == "__main__":
    main()
```

## Code Explanation

### 1. Frequency Parameters

```python
f_min = 1e9   # 1 GHz
f_max = 10e9  # 10 GHz
f0 = (f_min + f_max) / 2  # Center frequency
fc = (f_max - f_min) / 2  # Bandwidth
```

The Gaussian excitation is defined by:
- `f0`: Center frequency - where the pulse has maximum energy
- `fc`: Bandwidth - controls the frequency span

The pulse provides significant energy from `f0 - fc` to `f0 + fc`.

### 2. Cell Size Calculation

```python
wavelength_min = openems.C0 / f_max  # Minimum wavelength
delta = wavelength_min / 10          # Cell size
```

The FDTD method requires at least 10 cells per wavelength to accurately resolve the fields. Using `wavelength_min` (at `f_max`) ensures all frequencies are properly sampled.

### 3. Grid Creation

```python
grid = openems.Grid.uniform(n_cells, n_cells, n_cells, delta=delta)
```

Creates a uniform 3D grid. The total number of cells determines:
- Memory usage (~100 bytes/cell)
- Simulation time (proportional to cell count)

### 4. Boundary Conditions

```python
sim.set_boundary_cond([3, 3, 3, 3, 3, 3])  # PML everywhere
```

PML (Perfectly Matched Layer) absorbs outgoing waves with minimal reflection, simulating an infinite domain.

### 5. End Criteria

```python
sim.set_end_criteria(-30)  # Stop at 30 dB decay
```

Instead of running a fixed number of timesteps, the simulation stops when field energy has decayed by 30 dB (to 0.1% of peak).

## Expected Output

```
Frequency range: 1.0 - 10.0 GHz
Center frequency: 5.5 GHz
Minimum wavelength: 30.0 mm
Cell size: 3.0 mm

Grid: 50x50x50 cells
Total cells: 125,000
Grid created: 125,000 cells
Cell size: 3.00 x 3.00 x 3.00 mm

Simulation configured:
  Max timesteps: 5000
  End criteria: -30 dB
  Boundaries: PML all around

Starting simulation...
Simulation complete!
```

## Modifications to Try

### Higher Frequency

For 60 GHz simulation:

```python
f_min = 50e9
f_max = 70e9
# Smaller cells needed: ~0.5 mm
```

### Larger Domain

For a 500mm domain:

```python
domain_size = 0.5  # 500 mm
# More cells, longer simulation
```

### Asymmetric Domain

For a rectangular domain:

```python
nx = 100  # X cells
ny = 50   # Y cells
nz = 25   # Z cells
grid = openems.Grid.uniform(nx, ny, nz, delta=delta)
```

## Next Steps

- [Waveguide Example](waveguide.md) - Add metallic boundaries
- [Patch Antenna](patch_antenna.md) - Add materials and structures
