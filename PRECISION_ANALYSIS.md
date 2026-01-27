# Precision Analysis for FDTD Engine

## Overview
This document analyzes the precision requirements for various components of the FDTD simulation engine to inform the design of a hybrid f16/f32 GPU implementation. The goal is to maximize memory bandwidth by using f16 where safe, while retaining f32 for precision-sensitive operations.

## Precision-Sensitive Features

### 1. Absorbing Boundary Conditions (PML / UPML)
*   **Source:** `src/extensions/pml.rs`
*   **Mechanism:** Uniaxial PML (UPML) using auxiliary differential equations (ADE) with flux variables.
*   **Coefficients:** `vv`, `vvfn`, `vvfo`, `ii`, `iifn`, `iifo`.
*   **Precision Assessment:** **High (f32)**.
    *   PML relies on the precise cancellation of fields to achieve absorption (down to -80dB or lower).
    *   Coefficients are graded polynomially and span multiple orders of magnitude.
    *   The auxiliary state variables decay rapidly inside the PML; f16 underflow could lead to artificial reflections or "numerical floor" issues, limiting the dynamic range of the simulation.

### 2. Dispersive Materials (Lorentz, Drude, Debye)
*   **Source:** `src/extensions/dispersive.rs`
*   **Mechanism:** ADE methods solving for polarization/current densities ($P$ or $J$).
*   **Coefficients:** $a, b, c$ derived from $\omega_0, \gamma, \Delta\epsilon$.
*   **Precision Assessment:** **High (f32)**.
    *   Resonance frequencies ($\\omega_0$) and damping factors ($\\gamma$) can be very large or small.
    *   High-Q systems (low $\\gamma$) require accumulation of energy over many timesteps. Phase errors or amplitude decay due to f16 precision (11-bit significand) would significantly alter the resonance characteristics (Q-factor and center frequency).
    *   Recursive updates ($x_{n+1} = a x_n + \dots$) are prone to error accumulation.

### 3. Lumped Elements (RLC)
*   **Source:** `src/extensions/lumped_rlc.rs`
*   **Mechanism:** Localized updates involving integration of voltage/current.
*   **Precision Assessment:** **High (f32)**.
    *   Values for R, L, C can vary wildly (e.g., $1\Omega$ vs $1M\Omega$).
    *   The update equations involve small time-steps multiplied by large coefficients (or vice versa).
    *   Used for ports and precise circuit modeling.

### 4. Near-to-Far-Field Transform (NF2FF)
*   **Source:** `src/nf2ff/`
*   **Mechanism:** Surface integration of fields in the frequency domain (DFT).
*   **Precision Assessment:** **High (f32)**.
    *   Involves accumulating phase-sensitive complex numbers over the entire simulation duration.
    *   Precision errors in the DFT accumulation can destroy the far-field pattern, especially in nulls.

### 5. Standard Dielectric / Vacuum
*   **Source:** `src/fdtd/operator.rs`
*   **Mechanism:** Standard Yee algorithm ($E += C_b \nabla \times H$).
*   **Coefficients:** $C_a \approx 1$, $C_b \propto \Delta t / \epsilon$.
*   **Precision Assessment:** **Standard (f16 safe)**.
    *   For bulk wave propagation in vacuum or simple dielectrics, f16 provides sufficient dynamic range (~$10^{5}$ to $10^{-5}$ is easily covered) and precision (3-4 decimal digits).
    *   Dispersion error from the FDTD grid itself ($O(\\Delta x^2)$) typically dominates numerical precision errors.

## Design Recommendations

1.  **Bulk Regions:** Use `f16` for standard $E$ and $H$ field updates in vacuum and simple dielectric regions. This covers the majority of the simulation volume.
2.  **Sparse Precision:** Use `f32` for:
    *   PML regions (outer boundary layers).
    *   Cells containing dispersive material parameters.
    *   Cells containing lumped elements or sources.
3.  **Data Layout:**
    *   **f16 Buffer:** Stores coefficients for the entire grid (or defaults).
    *   **f32 Buffer:** Sparse/indexed buffer for precision-sensitive cells.
    *   **Classification Map:** A lightweight map (e.g., `u32` indices or bitmask) to direct the shader to the correct buffer/precision path per cell.
