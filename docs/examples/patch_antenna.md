# Patch Antenna

This example simulates a microstrip patch antenna designed for 2.4 GHz WiFi applications.

## Overview

A microstrip patch antenna consists of:
- A radiating metal patch on top
- A dielectric substrate
- A ground plane on the bottom

**Design Specifications:**
- Operating frequency: 2.4 GHz (WiFi)
- Substrate: FR-4 (εᵣ = 4.4, thickness = 1.6 mm)
- Target impedance: 50 Ω

## Complete Code

```python
"""
Microstrip Patch Antenna Simulation

Simulates a rectangular patch antenna for 2.4 GHz WiFi:
- Calculates patch dimensions from design equations
- Sets up computational domain with proper boundaries
- Demonstrates non-uniform meshing for accuracy

The antenna has:
- Rectangular radiating patch
- FR-4 dielectric substrate
- Infinite ground plane (simulated with PEC boundary)
"""

import openems
import numpy as np

def main():
    # ==========================================================================
    # Design Parameters
    # ==========================================================================

    # Operating frequency
    f_res = 2.4e9  # 2.4 GHz resonance frequency

    # Substrate properties
    eps_r = 4.4      # FR-4 relative permittivity
    h = 1.6e-3       # Substrate height: 1.6 mm
    tan_d = 0.02     # Loss tangent (for reference)

    print("Patch Antenna Design Parameters:")
    print(f"  Resonance frequency: {f_res/1e9:.1f} GHz")
    print(f"  Substrate: FR-4 (εr = {eps_r})")
    print(f"  Substrate height: {h*1000:.1f} mm")

    # ==========================================================================
    # Patch Dimension Calculations
    # ==========================================================================

    # Free-space wavelength
    lambda_0 = openems.C0 / f_res
    print(f"\nFree-space wavelength: {lambda_0*1000:.1f} mm")

    # Effective dielectric constant (approximate for W/h >> 1)
    # eps_eff ≈ (eps_r + 1)/2 + (eps_r - 1)/2 * (1 + 12*h/W)^(-0.5)
    # First approximation: assume W is about lambda_0/(2*sqrt(eps_r))

    # Patch width (for good radiation efficiency)
    W = openems.C0 / (2 * f_res) * np.sqrt(2 / (eps_r + 1))
    print(f"Patch width (W): {W*1000:.2f} mm")

    # Effective permittivity
    eps_eff = (eps_r + 1)/2 + (eps_r - 1)/2 * (1 + 12*h/W)**(-0.5)
    print(f"Effective permittivity: {eps_eff:.2f}")

    # Effective length extension due to fringing
    delta_L = 0.412 * h * ((eps_eff + 0.3) * (W/h + 0.264)) / \
                          ((eps_eff - 0.258) * (W/h + 0.8))
    print(f"Length extension (ΔL): {delta_L*1000:.2f} mm")

    # Patch length (electrical length = lambda_eff/2)
    lambda_eff = openems.C0 / (f_res * np.sqrt(eps_eff))
    L = lambda_eff / 2 - 2 * delta_L
    print(f"Patch length (L): {L*1000:.2f} mm")
    print(f"Effective wavelength: {lambda_eff*1000:.1f} mm")

    # ==========================================================================
    # Simulation Domain
    # ==========================================================================

    # Domain size: extend beyond patch by ~lambda/4 in each direction
    margin = lambda_0 / 4  # 31 mm margin

    # Domain dimensions
    domain_x = L + 2 * margin  # Along patch length
    domain_y = W + 2 * margin  # Along patch width
    domain_z = h + margin      # Above substrate (ground at z=0)

    print(f"\nSimulation Domain:")
    print(f"  X (length): {domain_x*1000:.1f} mm")
    print(f"  Y (width): {domain_y*1000:.1f} mm")
    print(f"  Z (height): {domain_z*1000:.1f} mm")

    # ==========================================================================
    # Mesh Setup
    # ==========================================================================

    # Frequency range for excitation
    f_min = 2.0e9   # 2.0 GHz
    f_max = 3.0e9   # 3.0 GHz
    f0 = (f_min + f_max) / 2
    fc = (f_max - f_min) / 2

    # Cell size requirements:
    # - lambda/15 at highest frequency in air
    # - lambda/15 at highest frequency in substrate
    # - Substrate height needs multiple cells

    lambda_min_air = openems.C0 / f_max  # 100 mm
    lambda_min_sub = openems.C0 / (f_max * np.sqrt(eps_r))  # ~48 mm

    delta_air = lambda_min_air / 15   # ~6.7 mm
    delta_sub = lambda_min_sub / 15   # ~3.2 mm

    # Use finer mesh (substrate requirement dominates)
    delta = min(delta_sub, h/4)  # At least 4 cells in substrate
    delta = 1e-3  # Round to 1 mm for simplicity

    print(f"\nMesh Parameters:")
    print(f"  Min wavelength (air): {lambda_min_air*1000:.1f} mm")
    print(f"  Min wavelength (substrate): {lambda_min_sub*1000:.1f} mm")
    print(f"  Cell size: {delta*1000:.1f} mm")

    # Create non-uniform mesh with finer resolution at patch edges
    # X-direction
    x_left = np.arange(-margin, -L/2, delta * 2)  # Coarser far field
    x_patch = np.arange(-L/2, L/2, delta)          # Fine at patch
    x_right = np.arange(L/2, domain_x/2 + delta, delta * 2)
    x = np.unique(np.concatenate([x_left, x_patch, x_right]))

    # Y-direction (similar approach)
    y_left = np.arange(-margin, -W/2, delta * 2)
    y_patch = np.arange(-W/2, W/2, delta)
    y_right = np.arange(W/2, domain_y/2 + delta, delta * 2)
    y = np.unique(np.concatenate([y_left, y_patch, y_right]))

    # Z-direction: fine in substrate, coarser above
    z_sub = np.linspace(0, h, 5)                    # 4 cells in substrate
    z_air = np.arange(h, domain_z + delta, delta * 2)
    z = np.unique(np.concatenate([z_sub, z_air]))

    grid = openems.Grid.from_lines(x.tolist(), y.tolist(), z.tolist())

    print(f"\nGrid Statistics:")
    print(f"  X mesh lines: {len(x)}")
    print(f"  Y mesh lines: {len(y)}")
    print(f"  Z mesh lines: {len(z)}")
    print(f"  Total cells: {grid.num_cells():,}")

    dx, dy, dz = grid.cell_size()
    print(f"  Min cell size: {dx*1000:.2f} x {dy*1000:.2f} x {dz*1000:.2f} mm")

    # ==========================================================================
    # Simulation Setup
    # ==========================================================================

    sim = openems.OpenEMS(num_timesteps=50000)
    sim.set_grid(grid)
    sim.set_gauss_excite(f0, fc)

    # Boundary conditions:
    # - PEC at z_min (ground plane)
    # - PML on all other sides (radiation)
    sim.set_boundary_cond([3, 3, 3, 3, 0, 3])

    # Set end criteria
    sim.set_end_criteria(-40)

    print(f"\nSimulation Configuration:")
    print(f"  Max timesteps: 50000")
    print(f"  End criteria: -40 dB")
    print(f"  Excitation: {f0/1e9:.1f} GHz ± {fc/1e9:.1f} GHz")
    print(f"  Z_min: PEC (ground plane)")
    print(f"  Other boundaries: PML (radiation)")

    # ==========================================================================
    # Run Simulation
    # ==========================================================================

    print("\nStarting simulation...")
    sim.run('./patch_antenna_output', cleanup=True)
    print("Simulation complete!")

    # ==========================================================================
    # Expected Results Summary
    # ==========================================================================

    print("\n" + "="*60)
    print("Expected Patch Antenna Characteristics")
    print("="*60)

    # Input impedance at resonance (approximate)
    # For inset-fed patch, impedance at edge ≈ 250-300 Ω
    z_edge = 90 * (eps_r**2) / (eps_r - 1) * (L/W)**2
    print(f"\nInput impedance at edge: ~{z_edge:.0f} Ω")
    print("(Use inset feed or quarter-wave transformer for 50Ω match)")

    # Bandwidth (approximate, VSWR < 2)
    Q = openems.C0 / (4 * f_res * h * np.sqrt(eps_eff))
    bw_percent = 100 / Q
    print(f"\nApproximate bandwidth: {bw_percent:.1f}%")
    print(f"(~{f_res * bw_percent / 100 / 1e6:.0f} MHz)")

    # Directivity (approximate)
    directivity_db = 10 * np.log10(4 * np.pi * W * L / lambda_0**2 * 0.8)
    print(f"\nApproximate directivity: ~{directivity_db:.1f} dBi")


if __name__ == "__main__":
    main()
```

## Design Equations

### Patch Width

For efficient radiation:

$$W = \frac{c}{2f_r}\sqrt{\frac{2}{\varepsilon_r + 1}}$$

### Effective Permittivity

$$\varepsilon_{eff} = \frac{\varepsilon_r + 1}{2} + \frac{\varepsilon_r - 1}{2}\left(1 + \frac{12h}{W}\right)^{-0.5}$$

### Length Extension (Fringing)

$$\Delta L = 0.412h\frac{(\varepsilon_{eff} + 0.3)(W/h + 0.264)}{(\varepsilon_{eff} - 0.258)(W/h + 0.8)}$$

### Patch Length

$$L = \frac{c}{2f_r\sqrt{\varepsilon_{eff}}} - 2\Delta L$$

## Mesh Strategy

The mesh uses different resolutions in different regions:

```
                    Coarse (far field)
              ┌─────────────────────────────┐
              │  ┌───────────────────────┐  │
              │  │   Fine (near patch)   │  │
              │  │  ┌─────────────────┐  │  │
              │  │  │                 │  │  │  ← Patch (very fine at edges)
              │  │  │     PATCH       │  │  │
              │  │  │                 │  │  │
              │  │  └─────────────────┘  │  │
              │  │                       │  │
              │  └───────────────────────┘  │
              │                             │
              └─────────────────────────────┘
                         PML boundary
```

## Expected Results

For a 2.4 GHz patch on FR-4:

| Parameter | Value |
|-----------|-------|
| Patch width | ~38 mm |
| Patch length | ~29 mm |
| Resonant frequency | 2.4 GHz |
| Bandwidth | ~2-3% (~60 MHz) |
| Directivity | ~6-7 dBi |
| Edge impedance | ~250 Ω |

## Modifications

### Different Frequency

```python
# 5 GHz WiFi
f_res = 5.8e9
# Patch will be ~60% smaller

# 900 MHz ISM band
f_res = 915e6
# Patch will be ~2.5x larger
```

### Different Substrate

```python
# Rogers RO4003C (lower loss)
eps_r = 3.55
h = 1.524e-3  # 60 mil

# Air (maximum bandwidth)
eps_r = 1.0
h = 5e-3
```

### Circular Patch

For a circular patch antenna:

```python
# Radius calculation
# f_res = 1.8412 * c / (2 * pi * a * sqrt(eps_r))
a = 1.8412 * openems.C0 / (2 * np.pi * f_res * np.sqrt(eps_r))
```

## Next Steps

- [Microstrip Line](microstrip.md) - Learn about transmission line design
- [Waveguide](waveguide.md) - Compare with waveguide-fed antennas
