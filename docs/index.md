# openEMS Python Documentation

**High-Performance FDTD Electromagnetic Field Solver**

openEMS is a free and open-source electromagnetic field solver using the Finite-Difference Time-Domain (FDTD) method. This Python package provides bindings to the high-performance Rust implementation.

## Features

- **High Performance**: Rust-based implementation with SIMD acceleration and parallel processing
- **Cross-Platform**: Runs on Linux, macOS, and Windows
- **Easy to Use**: Pythonic API with comprehensive documentation
- **Flexible Meshing**: Support for uniform and non-uniform grids
- **Multiple Boundary Types**: PEC, PMC, MUR, and PML boundary conditions

## Quick Example

```python
import openems

# Create a uniform computational grid
# 100x100x50 cells with 0.5mm resolution
grid = openems.Grid.uniform(100, 100, 50, delta=0.5e-3)

# Initialize the simulation
sim = openems.OpenEMS(num_timesteps=20000)
sim.set_grid(grid)

# Configure Gaussian excitation centered at 2.4 GHz
sim.set_gauss_excite(2.4e9, 0.5e9)

# Set boundary conditions (PML all around)
sim.set_boundary_cond([3, 3, 3, 3, 3, 3])

# Run the simulation
sim.run('./simulation_output', cleanup=True)
```

## Physical Constants

openEMS provides fundamental physical constants for electromagnetic simulations:

| Constant | Description | Value |
|----------|-------------|-------|
| `C0` | Speed of light in vacuum | 299,792,458 m/s |
| `EPS0` | Permittivity of free space | 8.854e-12 F/m |
| `MU0` | Permeability of free space | 1.257e-6 H/m |
| `Z0` | Impedance of free space | ~376.73 Ohm |

```python
import openems

# Calculate wavelength at 2.4 GHz
freq = 2.4e9
wavelength = openems.C0 / freq
print(f"Wavelength: {wavelength * 1000:.1f} mm")  # ~125 mm
```

## Next Steps

- [Installation Guide](getting-started/installation.md) - How to install openEMS
- [Quick Start Tutorial](getting-started/quickstart.md) - Get started with your first simulation
- [API Reference](api/index.md) - Detailed API documentation
- [Examples](examples/index.md) - Complete working examples

## License

openEMS is released under the GNU General Public License v3.0 (GPL-3.0).
