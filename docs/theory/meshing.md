# Meshing Guidelines

Proper mesh design is crucial for accurate and efficient FDTD simulations. This guide covers best practices for creating computational grids.

## Mesh Requirements

### Wavelength Resolution

The fundamental requirement: resolve the wavelength adequately.

**Rule of thumb**: At least 10 cells per wavelength at the highest frequency.

```python
import openems

def calculate_cell_size(f_max: float, cells_per_lambda: int = 10) -> float:
    """Calculate maximum cell size for given frequency."""
    wavelength = openems.C0 / f_max
    return wavelength / cells_per_lambda

# Example: 10 GHz simulation
f_max = 10e9
delta = calculate_cell_size(f_max)
print(f"Max cell size for {f_max/1e9} GHz: {delta*1000:.1f} mm")
# Output: Max cell size for 10.0 GHz: 3.0 mm
```

### In Dielectric Materials

Inside dielectrics, wavelength is reduced:

$$\lambda_{material} = \frac{\lambda_0}{\sqrt{\varepsilon_r \mu_r}}$$

```python
def calculate_cell_size_dielectric(f_max: float, eps_r: float,
                                   cells_per_lambda: int = 10) -> float:
    """Calculate cell size accounting for dielectric."""
    wavelength = openems.C0 / (f_max * (eps_r ** 0.5))
    return wavelength / cells_per_lambda

# Example: 10 GHz in FR-4 (eps_r = 4.4)
delta = calculate_cell_size_dielectric(10e9, 4.4)
print(f"Cell size in FR-4: {delta*1000:.1f} mm")
# Output: Cell size in FR-4: 1.4 mm
```

### Feature Resolution

Small features need adequate cells:

| Feature | Minimum Cells |
|---------|---------------|
| Thin traces | 3-5 across width |
| Gaps | 3-5 across gap |
| Dielectric thickness | 4-8 through thickness |
| Curved surfaces | Enough for smooth staircase |

## Uniform vs Non-Uniform Grids

### Uniform Grid

Simplest approach: same cell size everywhere.

```python
import openems

# Uniform 1mm grid
grid = openems.Grid.uniform(100, 100, 50, delta=1e-3)
```

**Pros:**
- Simple to set up
- Predictable behavior
- Easy to estimate resources

**Cons:**
- Inefficient for multi-scale problems
- May over-resolve some regions
- High memory for large domains with fine features

### Non-Uniform Grid

Variable cell sizes for efficiency.

```python
import numpy as np
import openems

# Fine mesh in center, coarser at edges
x_center = np.linspace(-0.01, 0.01, 41)  # 0.5mm cells
x_outer = np.linspace(0.01, 0.1, 19)     # 5mm cells
x = np.concatenate([
    -x_outer[::-1][:-1],  # Negative outer
    x_center,              # Center
    x_outer[1:]            # Positive outer
])

y = x.copy()
z = np.linspace(-0.02, 0.02, 41)

grid = openems.Grid.from_lines(x.tolist(), y.tolist(), z.tolist())
```

**Pros:**
- Efficient use of cells
- Fine resolution where needed
- Smaller memory footprint

**Cons:**
- More complex setup
- Potential for mesh-transition errors
- Requires careful design

## Cell Size Transitions

### Gradual Transitions

Adjacent cells should not differ by more than 30%:

$$\frac{\Delta_{large}}{\Delta_{small}} \leq 1.3$$

```python
def create_graded_mesh(start: float, end: float,
                       start_delta: float, end_delta: float,
                       max_ratio: float = 1.3) -> list:
    """Create mesh with gradual cell size transition."""
    mesh = [start]
    delta = start_delta
    pos = start

    while pos < end:
        pos += delta
        if pos > end:
            pos = end
        mesh.append(pos)

        # Gradually change cell size
        if delta < end_delta:
            delta = min(delta * max_ratio, end_delta)
        elif delta > end_delta:
            delta = max(delta / max_ratio, end_delta)

    return mesh
```

### Avoid Abrupt Changes

Bad:
```
│ 1mm │ 1mm │ 5mm │ 5mm │  <- 5x jump causes reflections
```

Good:
```
│ 1mm │ 1.3mm │ 1.7mm │ 2.2mm │ 2.9mm │ 3.8mm │ 5mm │
```

## Mesh Quality Checks

### Check Cell Size Range

```python
def check_mesh_quality(grid: openems.Grid) -> dict:
    """Analyze mesh quality."""
    dx, dy, dz = grid.cell_size()
    total_cells = grid.num_cells()

    # For non-uniform grids, would need access to mesh lines
    # This is a simplified check

    return {
        'total_cells': total_cells,
        'min_cell_size': min(dx, dy, dz),
        'max_cell_size': max(dx, dy, dz),
        'aspect_ratio': max(dx, dy, dz) / min(dx, dy, dz)
    }

grid = openems.Grid.uniform(100, 100, 50, delta=1e-3)
quality = check_mesh_quality(grid)
print(f"Cells: {quality['total_cells']:,}")
print(f"Aspect ratio: {quality['aspect_ratio']:.2f}")
```

### Memory Estimation

```python
def estimate_memory_gb(grid: openems.Grid,
                       bytes_per_cell: int = 100) -> float:
    """Estimate memory usage in GB."""
    return grid.num_cells() * bytes_per_cell / (1024**3)

# Check before running
grid = openems.Grid.uniform(500, 500, 250, delta=0.5e-3)
memory = estimate_memory_gb(grid)
print(f"Cells: {grid.num_cells():,}")
print(f"Estimated memory: {memory:.1f} GB")

if memory > 16:  # Your available RAM
    print("WARNING: May exceed available memory!")
```

## Common Meshing Patterns

### Antenna in Free Space

```python
import numpy as np
import openems

# Antenna dimensions
antenna_size = 0.05  # 50mm

# Simulation domain: antenna + margin
margin = 0.05  # 50mm for near-field + PML

# Frequency
f_max = 5e9
wavelength = openems.C0 / f_max

# Cell sizes
delta_antenna = wavelength / 20  # Fine at antenna
delta_farfield = wavelength / 10  # Coarser in far-field

# Create non-uniform mesh
# (conceptual - actual implementation would be more detailed)
```

### Microstrip with Via

```python
# Via dimensions
via_diameter = 0.3e-3  # 0.3mm
via_pad = 0.6e-3       # 0.6mm pad

# Mesh requirements:
# - 3+ cells across via diameter
# - Smooth transition to surrounding mesh

delta_via = via_diameter / 4  # ~75um at via
delta_trace = 0.2e-3          # 200um for traces
delta_field = 1e-3            # 1mm in field regions
```

### Waveguide Bend

```python
# Mesh needs to be fine at bend
# - Inner corner: high field concentration
# - Outer corner: lower fields

# Fine mesh at inner radius
r_inner = 0.01  # 10mm inner radius
delta_inner = r_inner / 20  # Fine cells at corner
```

## Performance Tips

### 1. Use Symmetry

Reduce problem size by 2x, 4x, or 8x:

```python
# Full model: 200x200x100 = 4M cells
# Half model (one symmetry): 100x200x100 = 2M cells
# Quarter model (two symmetries): 100x100x100 = 1M cells
# Eighth model (three symmetries): 100x100x50 = 0.5M cells
```

### 2. Optimize PML Distance

Don't over-extend the domain:

```python
# Minimum PML distance from structure: lambda/4
pml_distance = openems.C0 / f_max / 4

# Don't use more than necessary:
# Bad: lambda distance to PML
# Good: lambda/4 to PML
```

### 3. Start Coarse

Develop and debug with coarse mesh, then refine:

```python
# Development mesh
delta_dev = wavelength / 8  # Quick iterations

# Production mesh
delta_prod = wavelength / 20  # Final accuracy
```

### 4. Profile Memory Before Running

```python
import openems

def will_fit_in_memory(grid: openems.Grid,
                       available_gb: float = 16.0) -> bool:
    """Check if simulation will fit in memory."""
    estimated_gb = grid.num_cells() * 100 / (1024**3)
    return estimated_gb < available_gb * 0.8  # 80% safety margin

grid = openems.Grid.uniform(300, 300, 150, delta=0.5e-3)
if not will_fit_in_memory(grid):
    print("Consider: coarser mesh, symmetry, or more RAM")
```

## Debugging Mesh Issues

### Symptom: Staircasing artifacts
- **Cause**: Curved surfaces poorly resolved
- **Solution**: Finer mesh at curves

### Symptom: Unexpected resonances
- **Cause**: Mesh transition reflections
- **Solution**: Smoother cell size transitions

### Symptom: Slow convergence
- **Cause**: Small cells create small time step
- **Solution**: Check for unnecessarily fine regions

### Symptom: Results change with mesh refinement
- **Cause**: Inadequate resolution
- **Solution**: Converge results with successive refinement
