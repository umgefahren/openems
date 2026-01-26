# Microstrip Line

This example simulates a 50 Ω microstrip transmission line to determine its characteristic impedance and effective permittivity.

## Overview

A microstrip line consists of:
- A signal trace on top of a dielectric substrate
- A ground plane on the bottom

**Design Target:**
- Characteristic impedance: 50 Ω
- Substrate: FR-4 (εᵣ = 4.4, h = 1.6 mm)

## Complete Code

```python
"""
Microstrip Transmission Line Simulation

This example demonstrates:
- Microstrip impedance design equations
- TEM-like wave propagation
- Extraction of Z0 and effective permittivity

A 50 Ω microstrip on FR-4 substrate is simulated to verify
the design equations and demonstrate transmission line behavior.
"""

import openems
import numpy as np

def main():
    # ==========================================================================
    # Design Parameters
    # ==========================================================================

    # Target characteristic impedance
    Z0_target = 50.0  # Ohms

    # Substrate properties
    eps_r = 4.4      # FR-4 relative permittivity
    h = 1.6e-3       # Substrate height: 1.6 mm
    t = 35e-6        # Copper thickness: 35 μm (1 oz copper)

    print("Microstrip Design Parameters:")
    print(f"  Target impedance: {Z0_target} Ω")
    print(f"  Substrate: FR-4 (εr = {eps_r})")
    print(f"  Substrate height: {h*1000:.1f} mm")
    print(f"  Copper thickness: {t*1e6:.0f} μm")

    # ==========================================================================
    # Calculate Trace Width for 50 Ω
    # ==========================================================================

    # Wheeler's equations for microstrip impedance
    # First, calculate W/h ratio for target impedance

    A = Z0_target / 60 * np.sqrt((eps_r + 1) / 2) + \
        (eps_r - 1) / (eps_r + 1) * (0.23 + 0.11 / eps_r)

    W_h_ratio = 8 * np.exp(A) / (np.exp(2*A) - 2)

    # If W/h < 1, use narrow strip formula
    if W_h_ratio < 1:
        W = h * W_h_ratio
    else:
        # Wide strip formula
        B = 377 * np.pi / (2 * Z0_target * np.sqrt(eps_r))
        W_h_ratio = 2/np.pi * (B - 1 - np.log(2*B - 1) + \
                              (eps_r - 1)/(2*eps_r) * (np.log(B - 1) + 0.39 - 0.61/eps_r))
        W = h * W_h_ratio

    print(f"\nCalculated Trace Width:")
    print(f"  W/h ratio: {W_h_ratio:.3f}")
    print(f"  Trace width (W): {W*1000:.2f} mm")

    # ==========================================================================
    # Calculate Effective Permittivity
    # ==========================================================================

    # Effective permittivity for wide strips (W/h > 1)
    if W_h_ratio > 1:
        eps_eff = (eps_r + 1)/2 + (eps_r - 1)/2 * (1 + 12*h/W)**(-0.5)
    else:
        eps_eff = (eps_r + 1)/2 + (eps_r - 1)/2 * \
                  ((1 + 12*h/W)**(-0.5) + 0.04*(1 - W/h)**2)

    print(f"\nEffective Permittivity:")
    print(f"  εeff: {eps_eff:.3f}")

    # Verify impedance calculation
    if W_h_ratio > 1:
        Z0_calc = (120 * np.pi) / (np.sqrt(eps_eff) * (W/h + 1.393 + 0.667*np.log(W/h + 1.444)))
    else:
        Z0_calc = (60 / np.sqrt(eps_eff)) * np.log(8*h/W + W/(4*h))

    print(f"  Calculated Z0: {Z0_calc:.1f} Ω")

    # ==========================================================================
    # Frequency Setup
    # ==========================================================================

    # Operating frequency range
    f_min = 1e9    # 1 GHz
    f_max = 10e9   # 10 GHz
    f0 = (f_min + f_max) / 2
    fc = (f_max - f_min) / 2

    # Wavelengths
    lambda_0_min = openems.C0 / f_max  # ~30 mm
    lambda_eff_min = openems.C0 / (f_max * np.sqrt(eps_eff))  # ~21 mm

    print(f"\nFrequency Range:")
    print(f"  {f_min/1e9:.0f} - {f_max/1e9:.0f} GHz")
    print(f"  Min wavelength (air): {lambda_0_min*1000:.1f} mm")
    print(f"  Min wavelength (effective): {lambda_eff_min*1000:.1f} mm")

    # ==========================================================================
    # Simulation Domain
    # ==========================================================================

    # Microstrip line length (should be several wavelengths)
    line_length = 50e-3  # 50 mm

    # Domain width: enough for fringing fields (~5W on each side)
    domain_width = W + 10 * W  # 11W total

    # Domain height: substrate + air above (~5h)
    domain_height = h + 5 * h  # 6h total

    print(f"\nSimulation Domain:")
    print(f"  Length: {line_length*1000:.0f} mm")
    print(f"  Width: {domain_width*1000:.1f} mm")
    print(f"  Height: {domain_height*1000:.1f} mm")

    # ==========================================================================
    # Mesh Setup
    # ==========================================================================

    # Cell size requirements
    # - Trace width needs at least 5 cells
    # - Substrate height needs at least 4 cells
    # - Lambda_eff/10 at max frequency

    delta_trace = W / 5           # ~0.6 mm
    delta_substrate = h / 4       # 0.4 mm
    delta_lambda = lambda_eff_min / 10  # ~2.1 mm

    delta = min(delta_trace, delta_substrate)
    delta = 0.3e-3  # Round to 0.3 mm for fine detail

    print(f"\nMesh Setup:")
    print(f"  Cell size: {delta*1000:.1f} mm")

    # Non-uniform mesh
    # X-direction (along line): uniform, with PML cells
    pml_cells = 8
    nx = int(line_length / delta) + 2 * pml_cells

    # Y-direction (across trace): fine at trace, coarser at sides
    y_left = np.linspace(-domain_width/2, -W, 10)
    y_trace = np.linspace(-W, W, int(2*W/delta) + 1)
    y_right = np.linspace(W, domain_width/2, 10)
    y = np.unique(np.concatenate([y_left, y_trace, y_right]))

    # Z-direction: fine in substrate and near interface
    z_sub = np.linspace(0, h, 6)  # 5 cells in substrate
    z_air = np.linspace(h, domain_height, 15)
    z = np.unique(np.concatenate([z_sub, z_air]))

    # X: uniform for simplicity
    x = np.linspace(0, line_length + 2*pml_cells*delta, nx + 1)

    grid = openems.Grid.from_lines(x.tolist(), y.tolist(), z.tolist())

    print(f"  X mesh lines: {len(x)}")
    print(f"  Y mesh lines: {len(y)}")
    print(f"  Z mesh lines: {len(z)}")
    print(f"  Total cells: {grid.num_cells():,}")

    dx, dy, dz = grid.cell_size()
    print(f"  Min cell: {dx*1000:.2f} x {dy*1000:.2f} x {dz*1000:.2f} mm")

    # ==========================================================================
    # Simulation Setup
    # ==========================================================================

    sim = openems.OpenEMS(num_timesteps=30000)
    sim.set_grid(grid)
    sim.set_gauss_excite(f0, fc)

    # Boundary conditions:
    # - PML at x ends (wave ports)
    # - PML at y sides (to simulate infinite ground)
    # - PEC at z_min (ground plane)
    # - PML at z_max (open above)
    sim.set_boundary_cond([3, 3, 3, 3, 0, 3])

    sim.set_end_criteria(-40)

    print(f"\nSimulation Configuration:")
    print(f"  Max timesteps: 30000")
    print(f"  End criteria: -40 dB")
    print(f"  X boundaries: PML (ports)")
    print(f"  Y boundaries: PML (open sides)")
    print(f"  Z_min: PEC (ground plane)")
    print(f"  Z_max: PML (open above)")

    # ==========================================================================
    # Run Simulation
    # ==========================================================================

    print("\nStarting simulation...")
    sim.run('./microstrip_output', cleanup=True)
    print("Simulation complete!")

    # ==========================================================================
    # Post-Processing Concepts
    # ==========================================================================

    print("\n" + "="*60)
    print("Post-Processing Methods")
    print("="*60)

    print("""
To extract characteristic impedance from simulation:

1. Time-Domain Reflectometry (TDR):
   - Inject a step signal
   - Measure reflected voltage vs time
   - Z0 = V+ * (1 + Γ) / (1 - Γ)

2. S-Parameter Method:
   - Define ports at both ends
   - Calculate S11 and S21
   - Z0 = Z_port * sqrt((1+S11)/(1-S11)) for matched condition

3. Field Integration:
   - Integrate E-field across trace-ground gap: V = ∫E·dl
   - Integrate H-field around trace: I = ∮H·dl
   - Z0 = V / I
""")

    # Theoretical values summary
    print("\n" + "="*60)
    print("Theoretical Summary")
    print("="*60)
    print(f"\nMicrostrip Line Parameters:")
    print(f"  Trace width: {W*1000:.2f} mm")
    print(f"  Substrate height: {h*1000:.1f} mm")
    print(f"  Characteristic impedance: {Z0_calc:.1f} Ω")
    print(f"  Effective permittivity: {eps_eff:.3f}")

    # Phase velocity
    v_p = openems.C0 / np.sqrt(eps_eff)
    print(f"  Phase velocity: {v_p/1e8:.2f} × 10⁸ m/s ({v_p/openems.C0:.3f}c)")

    # Wavelength at 5 GHz
    f_test = 5e9
    lambda_g = openems.C0 / (f_test * np.sqrt(eps_eff))
    print(f"\nAt {f_test/1e9:.0f} GHz:")
    print(f"  Guide wavelength: {lambda_g*1000:.1f} mm")


if __name__ == "__main__":
    main()
```

## Design Equations

### Characteristic Impedance

For a microstrip with W/h > 1 (wide trace):

$$Z_0 = \frac{120\pi}{\sqrt{\varepsilon_{eff}} \left(\frac{W}{h} + 1.393 + 0.667\ln\left(\frac{W}{h} + 1.444\right)\right)}$$

For W/h < 1 (narrow trace):

$$Z_0 = \frac{60}{\sqrt{\varepsilon_{eff}}} \ln\left(\frac{8h}{W} + \frac{W}{4h}\right)$$

### Effective Permittivity

$$\varepsilon_{eff} = \frac{\varepsilon_r + 1}{2} + \frac{\varepsilon_r - 1}{2}\left(1 + \frac{12h}{W}\right)^{-0.5}$$

### Trace Width for Target Impedance

Using Wheeler's synthesis equations to calculate W for a given Z₀.

## Field Distribution

```
     E-field lines          H-field lines
                            (around trace)
         │ │ │ │
         │ │ │ │                  ○
         ▼ ▼ ▼ ▼               ╱   ╲
    ─────┴─┴─┴─┴─────      ○─┼─────┼─○
    │   TRACE   │            │TRACE│
    └───────────┘            │     │
━━━━━━━━━━━━━━━━━━━━━    ○──┼─────┼──○
    GROUND PLANE             │     │
                           ○─┼─────┼─○
                              ╲   ╱
                                ○
```

## Expected Results

For a 50 Ω microstrip on FR-4:

| Parameter | Value |
|-----------|-------|
| Trace width | ~2.9 mm |
| W/h ratio | ~1.8 |
| Effective εᵣ | ~3.3 |
| Phase velocity | 0.55c |

## Modifications

### Different Impedance

```python
# 75 Ω (coax-compatible)
Z0_target = 75.0
# Narrower trace needed

# 100 Ω (differential pair)
Z0_target = 100.0
# Even narrower

# 28 Ω (for matching to specific devices)
Z0_target = 28.0
# Wider trace needed
```

### Different Substrates

```python
# Rogers RO4003C (RF applications)
eps_r = 3.55
h = 0.813e-3  # 32 mil

# Alumina (ceramic)
eps_r = 9.8
h = 0.635e-3  # 25 mil

# PTFE/Teflon
eps_r = 2.1
h = 0.787e-3  # 31 mil
```

### Coupled Lines (Differential)

```python
# For differential impedance:
# Two traces separated by gap 's'
W = 0.3e-3   # Trace width
s = 0.2e-3   # Gap between traces

# Differential impedance ≈ 2 * Z_odd
# where Z_odd is odd-mode impedance
```

## Design Tips

1. **Trace width tolerance**: ±10% width change → ±5% Z₀ change

2. **Ground plane**: Should extend at least 3W beyond trace edges

3. **Via spacing**: For ground vias, spacing < λ/10 at highest frequency

4. **Copper thickness**: Affects impedance slightly; use accurate models for precision

5. **Dispersion**: At high frequencies, εeff becomes frequency-dependent

## Next Steps

- [Patch Antenna](patch_antenna.md) - Use microstrip feeding
- [Waveguide](waveguide.md) - Compare with waveguide transmission
