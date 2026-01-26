# Physical Constants

openEMS provides fundamental physical constants for electromagnetic simulations.

## Constants Reference

### Speed of Light

```python
import openems

# Speed of light in vacuum
c0 = openems.C0  # 299792458.0 m/s

# Calculate wavelength
freq = 2.4e9  # 2.4 GHz
wavelength = c0 / freq
print(f"Wavelength at {freq/1e9} GHz: {wavelength*1000:.2f} mm")
# Output: Wavelength at 2.4 GHz: 124.91 mm
```

### Permittivity of Free Space

```python
import openems

# Permittivity of free space (vacuum permittivity)
eps0 = openems.EPS0  # 8.854187817e-12 F/m

# Calculate capacitance of a parallel plate capacitor
area = 1e-4  # 1 cm^2 = 1e-4 m^2
distance = 1e-3  # 1 mm
capacitance = eps0 * area / distance
print(f"Capacitance: {capacitance*1e12:.3f} pF")
# Output: Capacitance: 0.885 pF
```

### Permeability of Free Space

```python
import openems

# Permeability of free space (vacuum permeability)
mu0 = openems.MU0  # 1.2566370614359173e-6 H/m (4*pi*1e-7)

# Calculate inductance of a single-turn coil
radius = 0.01  # 1 cm
# Approximate inductance: L ≈ mu0 * radius * (ln(8*radius/wire_radius) - 2)
wire_radius = 0.001  # 1 mm wire
inductance = mu0 * radius * (8 * radius / wire_radius - 2)
print(f"Approximate inductance: {inductance*1e9:.2f} nH")
```

### Impedance of Free Space

```python
import openems

# Impedance of free space (wave impedance in vacuum)
z0 = openems.Z0  # ~376.73 Ohm

# Relationship: Z0 = sqrt(MU0 / EPS0)
import math
z0_calculated = math.sqrt(openems.MU0 / openems.EPS0)
print(f"Z0: {z0:.6f} Ohm")
print(f"Calculated: {z0_calculated:.6f} Ohm")

# Power transmitted by a plane wave
e_field = 100  # V/m
power_density = e_field**2 / (2 * z0)  # W/m^2
print(f"Power density: {power_density:.2f} W/m^2")
```

## Constants Summary

| Constant | Symbol | Value | Unit | Description |
|----------|--------|-------|------|-------------|
| `C0` | c₀ | 299,792,458 | m/s | Speed of light in vacuum |
| `EPS0` | ε₀ | 8.854187817×10⁻¹² | F/m | Permittivity of free space |
| `MU0` | μ₀ | 1.256637061×10⁻⁶ | H/m | Permeability of free space |
| `Z0` | Z₀ | 376.730313668 | Ω | Impedance of free space |

## Relationships

The constants are related by fundamental electromagnetic equations:

```python
import openems
import math

# Speed of light from permittivity and permeability
c0_calc = 1.0 / math.sqrt(openems.EPS0 * openems.MU0)
print(f"c0 = 1/sqrt(ε₀μ₀) = {c0_calc:.0f} m/s")

# Wave impedance from permittivity and permeability
z0_calc = math.sqrt(openems.MU0 / openems.EPS0)
print(f"Z0 = sqrt(μ₀/ε₀) = {z0_calc:.2f} Ω")

# Verify relationships
assert abs(c0_calc - openems.C0) < 1
assert abs(z0_calc - openems.Z0) < 0.01
```

## Common Calculations

### Wavelength and Frequency

```python
import openems

def wavelength(freq: float) -> float:
    """Calculate wavelength in meters."""
    return openems.C0 / freq

def frequency(wavelength: float) -> float:
    """Calculate frequency in Hz."""
    return openems.C0 / wavelength

# Examples
print(f"1 GHz → {wavelength(1e9)*100:.1f} cm")
print(f"10 GHz → {wavelength(10e9)*1000:.1f} mm")
print(f"60 GHz → {wavelength(60e9)*1000:.2f} mm")
print(f"300 GHz → {wavelength(300e9)*1000:.3f} mm")
```

### Wave Number

```python
import openems
import math

def wave_number(freq: float) -> float:
    """Calculate wave number k = 2π/λ = 2πf/c."""
    return 2 * math.pi * freq / openems.C0

# Example
freq = 10e9  # 10 GHz
k = wave_number(freq)
print(f"Wave number at {freq/1e9} GHz: {k:.2f} rad/m")
```

### Phase Velocity in Media

```python
import openems
import math

def phase_velocity(epsilon_r: float, mu_r: float = 1.0) -> float:
    """Calculate phase velocity in a medium.

    Parameters
    ----------
    epsilon_r : float
        Relative permittivity
    mu_r : float
        Relative permeability (default: 1.0)

    Returns
    -------
    float
        Phase velocity in m/s
    """
    return openems.C0 / math.sqrt(epsilon_r * mu_r)

# Examples
print(f"Vacuum: {phase_velocity(1.0)/1e8:.2f} × 10⁸ m/s")
print(f"FR-4 (εr=4.4): {phase_velocity(4.4)/1e8:.2f} × 10⁸ m/s")
print(f"Water (εr=80): {phase_velocity(80)/1e8:.2f} × 10⁸ m/s")
```

### Characteristic Impedance

```python
import openems
import math

def characteristic_impedance(epsilon_r: float, mu_r: float = 1.0) -> float:
    """Calculate characteristic impedance of a medium.

    Parameters
    ----------
    epsilon_r : float
        Relative permittivity
    mu_r : float
        Relative permeability (default: 1.0)

    Returns
    -------
    float
        Characteristic impedance in Ohms
    """
    return openems.Z0 * math.sqrt(mu_r / epsilon_r)

# Examples
print(f"Vacuum: {characteristic_impedance(1.0):.1f} Ω")
print(f"FR-4: {characteristic_impedance(4.4):.1f} Ω")
print(f"Teflon (εr=2.1): {characteristic_impedance(2.1):.1f} Ω")
```

## Version Information

```python
import openems

print(f"openEMS version: {openems.VERSION}")
```
