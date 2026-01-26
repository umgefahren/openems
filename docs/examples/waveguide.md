# Rectangular Waveguide

This example simulates a WR-90 rectangular waveguide operating in the X-band (8.2-12.4 GHz).

## Overview

A rectangular waveguide is a hollow metallic tube that guides electromagnetic waves. The WR-90 waveguide is commonly used in X-band radar systems.

**Waveguide Specifications:**
- Inner dimensions: 22.86 mm × 10.16 mm (0.9" × 0.4")
- Cutoff frequency (TE₁₀): 6.56 GHz
- Operating range: 8.2 - 12.4 GHz

## Complete Code

```python
"""
Rectangular Waveguide Simulation

Simulates a WR-90 waveguide section to study:
- Wave propagation in the fundamental TE10 mode
- Cutoff frequency behavior
- Impedance characteristics

The waveguide has:
- PEC (metal) walls on all four sides
- PML absorbing boundaries at input/output ports
"""

import openems
import numpy as np

def main():
    # ==========================================================================
    # Waveguide Parameters
    # ==========================================================================

    # WR-90 internal dimensions
    a = 22.86e-3  # Width (broad wall): 22.86 mm
    b = 10.16e-3  # Height (narrow wall): 10.16 mm

    # Waveguide length
    length = 100e-3  # 100 mm

    # Calculate cutoff frequency for TE10 mode
    # fc = c / (2*a) for TE10
    fc_te10 = openems.C0 / (2 * a)
    print(f"WR-90 Waveguide Specifications:")
    print(f"  Dimensions: {a*1000:.2f} x {b*1000:.2f} mm")
    print(f"  Length: {length*1000:.0f} mm")
    print(f"  TE10 cutoff frequency: {fc_te10/1e9:.2f} GHz")

    # ==========================================================================
    # Frequency Setup
    # ==========================================================================

    # X-band operating frequency
    f_start = 8.2e9   # 8.2 GHz
    f_stop = 12.4e9   # 12.4 GHz

    # Excitation parameters
    f0 = (f_start + f_stop) / 2  # Center: 10.3 GHz
    fc = (f_stop - f_start) / 2  # Bandwidth: 2.1 GHz

    print(f"\nFrequency Range:")
    print(f"  Start: {f_start/1e9:.1f} GHz")
    print(f"  Stop: {f_stop/1e9:.1f} GHz")
    print(f"  Center: {f0/1e9:.1f} GHz")

    # ==========================================================================
    # Grid Setup
    # ==========================================================================

    # Cell size: lambda/15 at highest frequency
    wavelength_min = openems.C0 / f_stop  # ~24 mm at 12.4 GHz
    delta = wavelength_min / 15  # ~1.6 mm cells

    # Round to nice number for clarity
    delta = 1.5e-3  # 1.5 mm cells

    # Calculate grid dimensions
    # Add extra cells for PML (8 cells on each end in x)
    pml_cells = 8
    nx = int(length / delta) + 2 * pml_cells
    ny = int(a / delta)
    nz = int(b / delta)

    print(f"\nGrid Setup:")
    print(f"  Cell size: {delta*1000:.1f} mm")
    print(f"  Grid dimensions: {nx} x {ny} x {nz} cells")
    print(f"  Total cells: {nx*ny*nz:,}")

    # Create uniform grid
    grid = openems.Grid.uniform(nx, ny, nz, delta=delta)

    # ==========================================================================
    # Simulation Setup
    # ==========================================================================

    # Create simulation
    sim = openems.OpenEMS(num_timesteps=30000)
    sim.set_grid(grid)

    # Set Gaussian excitation
    sim.set_gauss_excite(f0, fc)

    # Boundary conditions:
    # - PML at input/output (x direction) for absorbing ports
    # - PEC on sidewalls (y, z directions) for metal walls
    #
    # [x_min, x_max, y_min, y_max, z_min, z_max]
    # 0 = PEC (Perfect Electric Conductor)
    # 3 = PML (Perfectly Matched Layer)
    sim.set_boundary_cond([3, 3, 0, 0, 0, 0])

    # Set end criteria
    sim.set_end_criteria(-40)  # 40 dB decay

    print(f"\nSimulation Setup:")
    print(f"  Max timesteps: 30000")
    print(f"  End criteria: -40 dB")
    print(f"  X boundaries: PML (absorbing ports)")
    print(f"  Y,Z boundaries: PEC (metal walls)")

    # ==========================================================================
    # Run Simulation
    # ==========================================================================

    print("\nStarting simulation...")
    sim.run('./waveguide_output', cleanup=True)
    print("Simulation complete!")

    # ==========================================================================
    # Theoretical Calculations
    # ==========================================================================

    print("\n" + "="*60)
    print("Theoretical TE10 Mode Properties")
    print("="*60)

    # Guide wavelength: lambda_g = lambda_0 / sqrt(1 - (fc/f)^2)
    freqs = np.linspace(f_start, f_stop, 5)
    for f in freqs:
        if f > fc_te10:
            lambda_0 = openems.C0 / f
            lambda_g = lambda_0 / np.sqrt(1 - (fc_te10/f)**2)
            # Wave impedance: Z_TE = Z0 / sqrt(1 - (fc/f)^2)
            z_te = openems.Z0 / np.sqrt(1 - (fc_te10/f)**2)
            # Phase velocity: v_p = c / sqrt(1 - (fc/f)^2)
            v_p = openems.C0 / np.sqrt(1 - (fc_te10/f)**2)

            print(f"\nAt {f/1e9:.1f} GHz:")
            print(f"  Free-space wavelength: {lambda_0*1000:.2f} mm")
            print(f"  Guide wavelength: {lambda_g*1000:.2f} mm")
            print(f"  Wave impedance: {z_te:.1f} Ohm")
            print(f"  Phase velocity: {v_p/openems.C0:.2f}c")


def calculate_s_parameters():
    """
    Example of how S-parameters would be calculated post-simulation.

    Note: This requires additional port definitions and field probes
    that would be added in a full simulation setup.
    """
    print("\nS-Parameter Calculation (conceptual):")
    print("  S11 = reflected power / incident power")
    print("  S21 = transmitted power / incident power")
    print("")
    print("  For a matched waveguide section:")
    print("  S11 ≈ 0 (no reflection)")
    print("  S21 ≈ 1 (full transmission)")


if __name__ == "__main__":
    main()
    calculate_s_parameters()
```

## Physics Background

### Waveguide Modes

In a rectangular waveguide, electromagnetic waves propagate in discrete modes:

- **TE (Transverse Electric)**: Electric field has no component along propagation direction
- **TM (Transverse Magnetic)**: Magnetic field has no component along propagation direction

The fundamental mode is TE₁₀ with cutoff frequency:

$$f_c = \frac{c}{2a}$$

where `a` is the broad wall dimension.

### Operating Range

The waveguide operates in single-mode when:

$$f_c^{TE_{10}} < f < f_c^{TE_{20}}$$

For WR-90: 6.56 GHz < f < 13.12 GHz

### Guide Wavelength

The wavelength inside the waveguide differs from free space:

$$\lambda_g = \frac{\lambda_0}{\sqrt{1 - (f_c/f)^2}}$$

At the cutoff frequency, λ_g → ∞ (no propagation).

## Boundary Conditions Explained

```
        z ↑
          │     ┌─────────────────────────┐
          │     │      PEC (top wall)      │
          │     │                          │
          │ ────┼──────────────────────────┼──── y
     PML ─┤     │                          │ ├─ PML
  (input) │     │                          │   (output)
          │     │     PEC (bottom wall)    │
          │     └─────────────────────────┘
          │
          └────────────────────────────────────→ x
                        (propagation)
```

- **PEC (code 0)**: Metal walls - reflects all waves
- **PML (code 3)**: Absorbing boundaries - simulates infinite waveguide

## Expected Results

For a matched WR-90 waveguide at 10 GHz:

| Parameter | Value |
|-----------|-------|
| Free-space wavelength | 30 mm |
| Guide wavelength | 39.7 mm |
| Wave impedance | 500 Ω |
| Phase velocity | 1.32c |

## Modifications

### Different Waveguide Sizes

```python
# WR-62 (Ku-band, 12.4-18 GHz)
a = 15.80e-3  # mm
b = 7.90e-3   # mm

# WR-42 (K-band, 18-26.5 GHz)
a = 10.67e-3
b = 4.32e-3

# WR-28 (Ka-band, 26.5-40 GHz)
a = 7.11e-3
b = 3.56e-3
```

### Add Discontinuity

To study a step discontinuity:

```python
# Create non-uniform grid with finer mesh at discontinuity
x_uniform = np.linspace(0, length/2, int(length/2/delta))
x_fine = np.linspace(length/2, length/2 + 0.01, 41)  # Fine at step
x_rest = np.linspace(length/2 + 0.01, length, int(length/2/delta))
x = np.concatenate([x_uniform[:-1], x_fine, x_rest[1:]])
```

## Next Steps

- [Patch Antenna](patch_antenna.md) - Open radiating structure
- [Microstrip Line](microstrip.md) - Planar transmission line
