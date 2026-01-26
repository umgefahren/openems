# Boundary Conditions

Boundary conditions determine how electromagnetic fields behave at the edges of the computational domain. Proper boundary selection is critical for accurate simulations.

## Overview

openEMS supports four boundary condition types:

| Code | Type | Description |
|------|------|-------------|
| 0 | PEC | Perfect Electric Conductor |
| 1 | PMC | Perfect Magnetic Conductor |
| 2 | MUR | First-order Mur ABC |
| 3 | PML | Perfectly Matched Layer |

```python
import openems

# Set boundaries: [x_min, x_max, y_min, y_max, z_min, z_max]
sim.set_boundary_cond([3, 3, 3, 3, 3, 3])  # PML all around
```

## Perfect Electric Conductor (PEC)

**Code: 0**

PEC boundaries simulate a perfect metal surface:
- Tangential E-field is zero: **E**_tan = 0
- Normal H-field is zero: **H**_norm = 0
- Waves are perfectly reflected with 180° phase shift

### Use Cases

- Metal walls and enclosures
- Ground planes
- Electric symmetry planes (E-field normal to plane)

### Example: Waveguide Walls

```python
# Rectangular waveguide with metal walls
# PML at ports (x), PEC walls (y, z)
sim.set_boundary_cond([3, 3, 0, 0, 0, 0])
```

### Field Behavior at PEC

```
        │ E_tangential = 0
        │
    ────┴──── PEC surface
        │
        │ H_tangential ≠ 0 (surface current)
        │
```

## Perfect Magnetic Conductor (PMC)

**Code: 1**

PMC boundaries are the magnetic dual of PEC:
- Tangential H-field is zero: **H**_tan = 0
- Normal E-field is zero: **E**_norm = 0
- Waves are perfectly reflected with 0° phase shift

### Use Cases

- Magnetic symmetry planes (H-field normal to plane)
- Approximate magnetic walls in some structures
- Artificial boundary for symmetry exploitation

### Example: Symmetry Exploitation

For a symmetric antenna, using PMC halves the simulation size:

```python
# Antenna symmetric about x=0 plane (magnetic symmetry)
# PMC at x_min, PML elsewhere
sim.set_boundary_cond([1, 3, 3, 3, 3, 3])

# Simulation size reduced by 2x
```

### Symmetry Selection Guide

| Symmetry Type | E-field | H-field | Boundary |
|--------------|---------|---------|----------|
| Electric | Normal | Tangential | PEC (0) |
| Magnetic | Tangential | Normal | PMC (1) |

## Mur Absorbing Boundary (MUR)

**Code: 2**

The Mur ABC is a first-order absorbing boundary condition:
- Absorbs normally incident waves
- Some reflection at oblique incidence
- Computationally lightweight

### Use Cases

- Quick test simulations
- When low accuracy is acceptable
- Memory-constrained situations

### Limitations

- Significant reflection (5-10%) at oblique incidence
- Not suitable for high-accuracy simulations
- Poor performance for evanescent waves

```python
# Quick test with Mur boundaries
sim.set_boundary_cond([2, 2, 2, 2, 2, 2])
```

## Perfectly Matched Layer (PML)

**Code: 3**

PML is the gold standard for absorbing boundaries:
- Absorbs waves at all angles of incidence
- Minimal reflection (-40 to -80 dB typically)
- Requires additional cells (8-16 per boundary)

### How PML Works

PML uses a fictitious lossy medium that:
1. Is impedance-matched to free space (no reflection at interface)
2. Attenuates waves as they propagate through it
3. Uses complex coordinate stretching

### PML Parameters

openEMS automatically configures PML with sensible defaults:
- 8 PML cells per boundary
- Polynomial grading for smooth absorption

### Use Cases

- Open-space radiation problems
- Antenna simulations
- RCS calculations
- Any simulation requiring minimal boundary reflections

```python
# Antenna in free space: PML all around
sim.set_boundary_cond([3, 3, 3, 3, 3, 3])
```

### Memory Impact

PML adds cells to the simulation domain:

```python
# Example: 100x100x100 base grid
# With PML (8 cells each side):
# Total: 116x116x116 = 1.56M cells
# vs base: 1M cells
# Overhead: ~56%
```

## Boundary Condition Comparison

| Aspect | PEC | PMC | MUR | PML |
|--------|-----|-----|-----|-----|
| Reflection | 100% | 100% | 5-10% | <0.01% |
| Memory | None | None | None | +50-100% |
| Speed | Fast | Fast | Fast | Moderate |
| Accuracy | Exact | Exact | Low | High |

## Practical Examples

### Free Space (Antenna)

```python
# Radiating antenna: need absorbing boundaries
sim.set_boundary_cond([3, 3, 3, 3, 3, 3])
```

### Waveguide

```python
# Metal walls, open ports
sim.set_boundary_cond([3, 3, 0, 0, 0, 0])
#                     ports  walls
```

### Microstrip

```python
# Ground plane at z_min, open elsewhere
sim.set_boundary_cond([3, 3, 3, 3, 0, 3])
#                               ground  open
```

### Symmetric Dipole

```python
# Half the dipole with symmetry
# Electric symmetry at y=0 (current flows along y)
sim.set_boundary_cond([3, 3, 0, 3, 3, 3])
#                           sym  pml
```

### Quarter Model (Two Symmetry Planes)

```python
# For doubly-symmetric structures
# Electric symmetry at x=0, magnetic at y=0
sim.set_boundary_cond([0, 3, 1, 3, 3, 3])
# Reduces cells by 4x!
```

## Tips for Choosing Boundaries

1. **Default to PML** for unknown boundary behavior
2. **Use PEC/PMC** for known metallic or symmetric boundaries
3. **Exploit symmetry** to reduce computation
4. **Test with MUR** for quick iterations, then use PML for final
5. **Leave space** between structures and PML (λ/4 minimum)

## Debugging Boundary Issues

### Symptom: Late-time instability
- **Cause**: Often PML parameters
- **Solution**: Ensure mesh is not too coarse at PML interface

### Symptom: Unexpected reflections
- **Cause**: PML too close to structure or wrong type
- **Solution**: Move boundaries further out, check boundary type

### Symptom: Resonances in open structure
- **Cause**: Boundary reflections
- **Solution**: Use PML, increase distance from structure
