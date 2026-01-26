# Examples

This section provides complete, working examples that demonstrate various electromagnetic simulations using openEMS.

## Overview

Each example includes:

- Complete Python code
- Explanation of the physics
- Expected results
- Tips for modifications

## Examples by Category

### Getting Started

| Example | Description | Difficulty |
|---------|-------------|------------|
| [Basic Simulation](basic_simulation.md) | Free space propagation | Beginner |

### Waveguides

| Example | Description | Difficulty |
|---------|-------------|------------|
| [Rectangular Waveguide](waveguide.md) | WR-90 waveguide S-parameters | Intermediate |

### Antennas

| Example | Description | Difficulty |
|---------|-------------|------------|
| [Patch Antenna](patch_antenna.md) | Microstrip patch antenna at 2.4 GHz | Intermediate |

### Transmission Lines

| Example | Description | Difficulty |
|---------|-------------|------------|
| [Microstrip Line](microstrip.md) | Microstrip characteristic impedance | Intermediate |

## Running Examples

All examples follow the same pattern:

```python
import openems

# 1. Create grid
grid = openems.Grid.uniform(...)

# 2. Configure simulation
sim = openems.OpenEMS(num_timesteps=...)
sim.set_grid(grid)
sim.set_gauss_excite(f0, fc)
sim.set_boundary_cond([...])

# 3. Run
sim.run('./output', cleanup=True)
```

## Tips for Modifying Examples

### Changing Frequency

To adapt an example to a different frequency:

1. Recalculate the wavelength: `λ = C0 / f`
2. Scale the grid cell size: `delta = λ / 10` minimum
3. Adjust the excitation: `set_gauss_excite(new_f0, new_fc)`

### Improving Accuracy

For better accuracy:

1. Use finer mesh: decrease `delta` or use non-uniform grid
2. Extend simulation time: increase `num_timesteps`
3. Tighten end criteria: use `-50` instead of `-40` dB

### Reducing Computation Time

For faster simulations:

1. Use coarser mesh (if accuracy permits)
2. Exploit symmetry with PEC/PMC boundaries
3. Minimize domain size
4. Use MUR instead of PML for quick tests
