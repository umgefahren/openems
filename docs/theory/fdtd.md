# FDTD Method

The Finite-Difference Time-Domain (FDTD) method is a numerical technique for solving Maxwell's equations directly in the time domain.

## Maxwell's Equations

In a source-free region, Maxwell's curl equations are:

$$\nabla \times \mathbf{E} = -\mu \frac{\partial \mathbf{H}}{\partial t}$$

$$\nabla \times \mathbf{H} = \varepsilon \frac{\partial \mathbf{E}}{\partial t}$$

## Yee Algorithm

The FDTD method uses the Yee algorithm (1966), which:

1. **Staggers E and H fields in space** - Electric and magnetic field components are offset by half a cell
2. **Staggers E and H in time** - Updates alternate between E and H fields (leapfrog scheme)

### Yee Cell

```
                z
                │
                │  Ez
                │   │
                ├───┼───┤ Hy
                │   │   │
            Hx ─┼───●───┼─ Hx
                │   │   │
                ├───┼───┤ Hy
                    │
                   Ez
                ────────────── y
               ╱
              ╱ Ey
             ╱
            x

    ● = Ex component at cell center
    Field components are offset by Δx/2, Δy/2, or Δz/2
```

### Update Equations

For the Ex component:

$$E_x^{n+1}(i,j,k) = E_x^n(i,j,k) + \frac{\Delta t}{\varepsilon} \left[
\frac{H_z^{n+1/2}(i,j,k) - H_z^{n+1/2}(i,j-1,k)}{\Delta y} -
\frac{H_y^{n+1/2}(i,j,k) - H_y^{n+1/2}(i,j,k-1)}{\Delta z}
\right]$$

Similar equations exist for all six field components.

## Stability Condition

The FDTD method is conditionally stable. The time step must satisfy the Courant-Friedrichs-Lewy (CFL) condition:

$$\Delta t \leq \frac{1}{c\sqrt{\frac{1}{\Delta x^2} + \frac{1}{\Delta y^2} + \frac{1}{\Delta z^2}}}$$

For a uniform grid with Δx = Δy = Δz = δ:

$$\Delta t \leq \frac{\delta}{c\sqrt{3}}$$

In openEMS, the time step is automatically calculated from the grid:

```python
import openems

grid = openems.Grid.uniform(100, 100, 100, delta=1e-3)
dx, dy, dz = grid.cell_size()

# Maximum stable time step
import math
dt_max = 1.0 / (openems.C0 * math.sqrt(1/dx**2 + 1/dy**2 + 1/dz**2))
print(f"Maximum time step: {dt_max*1e12:.3f} ps")
```

## Numerical Dispersion

FDTD introduces numerical dispersion - the phase velocity depends on frequency and propagation direction. This error decreases with finer grids.

### Dispersion Error

For a wave propagating along a grid axis:

$$\frac{1}{c_{num}} = \frac{1}{c} \left[1 + \frac{1}{12}\left(\frac{2\pi f \delta}{c}\right)^2 + O(\delta^4)\right]$$

The error is minimized when:
- Using at least 10-20 cells per wavelength
- Using the "magic time step" Δt = δ/(c√3) for 3D

### Practical Guidelines

| Cells per λ | Phase Error |
|-------------|-------------|
| 10 | ~1% |
| 15 | ~0.4% |
| 20 | ~0.2% |
| 30 | ~0.1% |

```python
import openems

def cells_per_wavelength(freq, delta):
    """Calculate cells per wavelength."""
    wavelength = openems.C0 / freq
    return wavelength / delta

# Example: 10 GHz with 1mm cells
cpp = cells_per_wavelength(10e9, 1e-3)
print(f"Cells per wavelength: {cpp:.1f}")
```

## Algorithm Steps

The FDTD time-stepping algorithm:

```
1. Initialize fields (E = H = 0 everywhere)
2. For each time step n:
   a. Update H fields (half step): H^(n+1/2) from H^(n-1/2) and E^n
   b. Apply H-field boundary conditions
   c. Update E fields (full step): E^(n+1) from E^n and H^(n+1/2)
   d. Apply E-field boundary conditions
   e. Add sources/excitations
   f. Record field probes
   g. Check termination criteria
3. Post-process results (FFT for frequency domain)
```

## Time-to-Frequency Conversion

FDTD operates in time domain. For frequency-domain results:

1. Record time-domain fields at points of interest
2. Apply FFT to get frequency-domain response
3. Normalize by excitation spectrum

```python
import numpy as np

# Example: Convert time-domain voltage to frequency-domain
def time_to_freq(time_data, dt, freq_points):
    """
    Convert time-domain data to frequency domain.

    Parameters
    ----------
    time_data : array
        Time-domain samples
    dt : float
        Time step
    freq_points : array
        Frequencies of interest

    Returns
    -------
    array
        Complex frequency-domain data
    """
    n_samples = len(time_data)
    freq_data = np.zeros(len(freq_points), dtype=complex)

    for i, f in enumerate(freq_points):
        # DFT at specific frequency
        t = np.arange(n_samples) * dt
        freq_data[i] = np.sum(time_data * np.exp(-2j * np.pi * f * t)) * dt

    return freq_data
```

## Advantages of FDTD

1. **Broadband**: Single simulation covers wide frequency range
2. **Intuitive**: Direct solution of Maxwell's equations
3. **Versatile**: Handles complex geometries and materials
4. **Parallel**: Naturally suited for parallel computing
5. **Memory efficient**: O(N) memory for N cells

## Limitations

1. **Staircase approximation**: Curved surfaces approximated by steps
2. **Dispersion**: Numerical phase error (mitigated by fine mesh)
3. **Stability limit**: Time step restricted by CFL condition
4. **Resonant structures**: Long simulation times for high-Q resonators

## Further Reading

- Taflove, A. & Hagness, S.C. "Computational Electrodynamics: The FDTD Method"
- Yee, K. "Numerical Solution of Initial Boundary Value Problems..." IEEE TAP, 1966
