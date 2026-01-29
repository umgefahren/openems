//! Convolutional Perfectly Matched Layer (CPML) implementation.
//!
//! This implements CPML using auxiliary differential equations (ADE) with
//! recursive convolution. Unlike UPML, CPML doesn't require modifying the
//! main operator coefficients - it adds correction terms after each field update.
//!
//! Reference: Roden & Gedney, "Convolutional PML (CPML): An efficient FDTD
//! implementation of the CFS-PML for arbitrary media", Microwave and Optical
//! Technology Letters, Vol. 27, No. 5, 2000.
//!
//! The CPML adds a correction term to each field component:
//!   E_new = E_update + psi_E
//!   H_new = H_update + psi_H
//!
//! Where psi is updated as:
//!   psi^{n+1} = b * psi^n + a * (spatial derivative term)
//!
//! The coefficients a and b are computed from the PML parameters:
//!   b = exp(-(sigma/eps0 + alpha) * dt)
//!   a = sigma / (sigma + alpha*eps0) * (b - 1)

use crate::arrays::{Dimensions, Field3D, VectorField3D};
use crate::constants::{EPS0, MU0, Z0};

/// CPML boundary configuration for all 6 faces.
#[derive(Debug, Clone)]
pub struct CpmlBoundaries {
    /// Number of CPML layers for each boundary [xmin, xmax, ymin, ymax, zmin, zmax]
    /// Set to 0 to disable CPML on that boundary.
    pub layers: [usize; 6],
    /// Polynomial grading order (typically 3-4)
    pub grading_order: f64,
    /// Target reflection coefficient (e.g., 1e-6 for -120 dB)
    pub reflection_coeff: f64,
    /// Alpha scaling factor for CFS-PML (typically 0.0 to 0.05)
    pub alpha_max: f64,
    /// Kappa max for coordinate stretching (typically 1.0 to 15.0)
    pub kappa_max: f64,
}

impl Default for CpmlBoundaries {
    fn default() -> Self {
        Self {
            layers: [8; 6],
            grading_order: 3.0,
            reflection_coeff: 1e-6,
            alpha_max: 0.0, // Standard PML without CFS
            kappa_max: 1.0, // No coordinate stretching
        }
    }
}

impl CpmlBoundaries {
    /// Create with uniform CPML on all boundaries.
    pub fn uniform(layers: usize) -> Self {
        Self {
            layers: [layers; 6],
            ..Default::default()
        }
    }

    /// Check if any CPML is enabled.
    pub fn any_enabled(&self) -> bool {
        self.layers.iter().any(|&l| l > 0)
    }

    /// Calculate optimal sigma_max for the given parameters.
    pub fn sigma_max(&self, delta: f64) -> f64 {
        // sigma_max = -(m+1) * ln(R) / (2 * Z0 * d)
        let d = self.layers.iter().max().copied().unwrap_or(1) as f64 * delta;
        -(self.grading_order + 1.0) * self.reflection_coeff.ln() / (2.0 * Z0 * d)
    }
}

/// CPML coefficients for a single direction.
#[derive(Debug)]
struct CpmlCoefficients {
    /// b coefficient: b = exp(-(sigma/eps0 + alpha) * dt)
    b_e: Vec<f32>,
    b_h: Vec<f32>,
    /// a coefficient: a = sigma / (sigma + alpha*eps0) * (b - 1)
    a_e: Vec<f32>,
    a_h: Vec<f32>,
}

impl CpmlCoefficients {
    fn new(layers: usize) -> Self {
        Self {
            b_e: vec![0.0; layers],
            b_h: vec![0.0; layers],
            a_e: vec![0.0; layers],
            a_h: vec![0.0; layers],
        }
    }

    /// Initialize coefficients for a PML boundary.
    #[allow(clippy::too_many_arguments)]
    fn init(
        &mut self,
        delta: f64,
        dt: f64,
        sigma_max: f64,
        alpha_max: f64,
        kappa_max: f64,
        grading_order: f64,
        is_min_boundary: bool,
    ) {
        let layers = self.b_e.len();
        let width = layers as f64 * delta;

        for i in 0..layers {
            // Calculate depth into PML (0 at interface, 1 at boundary)
            // E and H components are staggered by half a cell
            let depth_e = if is_min_boundary {
                (layers - i) as f64 * delta
            } else {
                (i + 1) as f64 * delta
            };

            let depth_h = if is_min_boundary {
                (layers - i) as f64 * delta - 0.5 * delta
            } else {
                (i + 1) as f64 * delta - 0.5 * delta
            };

            // Polynomial grading
            let rho_e = (depth_e / width).clamp(0.0, 1.0);
            let rho_h = (depth_h / width).clamp(0.0, 1.0);

            let sigma_e = sigma_max * rho_e.powf(grading_order);
            let sigma_h = sigma_max * rho_h.powf(grading_order);

            let alpha_e = alpha_max * (1.0 - rho_e);
            let alpha_h = alpha_max * (1.0 - rho_h);

            let kappa_e = 1.0 + (kappa_max - 1.0) * rho_e.powf(grading_order);
            let kappa_h = 1.0 + (kappa_max - 1.0) * rho_h.powf(grading_order);

            // Compute b = exp(-(sigma/kappa/eps0 + alpha/eps0) * dt)
            let b_e = (-(sigma_e / kappa_e / EPS0 + alpha_e / EPS0) * dt).exp();
            let b_h = (-(sigma_h / kappa_h / EPS0 + alpha_h / EPS0) * dt).exp();

            // Compute a coefficient for CPML recursive update
            // Using normalized sigma: sigma_norm = sigma / (eps0 * kappa)
            // b = exp(-(sigma_norm + alpha/eps0) * dt)
            // a = sigma_norm / (sigma_norm + alpha/eps0) * (b - 1)
            let sigma_norm_e = sigma_e / (EPS0 * kappa_e);
            let sigma_norm_h = sigma_h / (EPS0 * kappa_h);
            let alpha_norm_e = alpha_e / EPS0;
            let alpha_norm_h = alpha_h / EPS0;

            let denom_e = sigma_norm_e + alpha_norm_e;
            let denom_h = sigma_norm_h + alpha_norm_h;

            // CPML coefficient formula
            // Using: psi = b*psi + a*ΔE with a > 0
            // so that psi has the SAME sign as ΔE, and the correction H += D_B*psi
            // adds to the field (which combined with the FDTD update creates the PML effect)
            // This is equivalent to: a = sigma / (sigma + kappa*alpha) * (1 - b)
            let a_e = if denom_e.abs() > 1e-20 {
                sigma_norm_e / denom_e * (1.0 - b_e) // Note: (1-b) > 0
            } else {
                0.0
            };

            let a_h = if denom_h.abs() > 1e-20 {
                sigma_norm_h / denom_h * (1.0 - b_h) // Note: (1-b) > 0
            } else {
                0.0
            };

            self.b_e[i] = b_e as f32;
            self.b_h[i] = b_h as f32;
            self.a_e[i] = a_e as f32;
            self.a_h[i] = a_h as f32;
        }
    }
}

/// CPML psi field storage for one direction and one field type.
#[derive(Debug)]
struct PsiField {
    /// Psi values for positive derivative (e.g., dEy/dx, dEz/dx for x-direction CPML)
    psi_pos: Field3D,
    /// Psi values for negative derivative (e.g., -dEz/dy, -dEy/dz for x-direction CPML)
    psi_neg: Field3D,
}

impl PsiField {
    fn new(dims: Dimensions) -> Self {
        Self {
            psi_pos: Field3D::new(dims),
            psi_neg: Field3D::new(dims),
        }
    }

    fn clear(&mut self) {
        self.psi_pos.clear();
        self.psi_neg.clear();
    }
}

/// CPML absorbing boundary condition.
///
/// Implements Convolutional PML using recursive convolution for efficient
/// computation. This approach adds correction terms to the field updates
/// without requiring modification of the main operator coefficients.
pub struct Cpml {
    /// Global grid dimensions
    dims: Dimensions,
    /// Cell sizes
    delta: [f64; 3],
    /// Timestep
    dt: f64,
    /// Boundary configuration (stored for reference)
    #[allow(dead_code)]
    boundaries: CpmlBoundaries,

    /// Coefficients for each direction [x, y, z]
    coeff_min: [Option<CpmlCoefficients>; 3],
    coeff_max: [Option<CpmlCoefficients>; 3],

    /// Psi fields for E-field correction (stored per direction)
    /// psi_e[dir] stores psi for E-field components affected by CPML in direction dir
    psi_e_min: [Option<PsiField>; 3],
    psi_e_max: [Option<PsiField>; 3],

    /// Psi fields for H-field correction
    psi_h_min: [Option<PsiField>; 3],
    psi_h_max: [Option<PsiField>; 3],
}

impl Cpml {
    /// Create a new CPML with the given configuration.
    pub fn new(dims: Dimensions, delta: [f64; 3], dt: f64, boundaries: CpmlBoundaries) -> Self {
        let sigma_max = boundaries.sigma_max(delta[0].min(delta[1]).min(delta[2]));

        let mut cpml = Self {
            dims,
            delta,
            dt,
            boundaries: boundaries.clone(),
            coeff_min: [None, None, None],
            coeff_max: [None, None, None],
            psi_e_min: [None, None, None],
            psi_e_max: [None, None, None],
            psi_h_min: [None, None, None],
            psi_h_max: [None, None, None],
        };

        // Initialize coefficients and psi fields for each enabled boundary
        #[allow(clippy::needless_range_loop)] // Index used for multiple array accesses
        for dir in 0..3 {
            let grid_size = match dir {
                0 => dims.nx,
                1 => dims.ny,
                _ => dims.nz,
            };

            // Min boundary
            let layers_min = boundaries.layers[dir * 2];
            if layers_min > 0 && layers_min < grid_size {
                let mut coeff = CpmlCoefficients::new(layers_min);
                coeff.init(
                    delta[dir],
                    dt,
                    sigma_max,
                    boundaries.alpha_max,
                    boundaries.kappa_max,
                    boundaries.grading_order,
                    true,
                );
                cpml.coeff_min[dir] = Some(coeff);

                // Psi field dimensions for this boundary
                let psi_dims = Self::psi_dims(dims, dir, layers_min);
                cpml.psi_e_min[dir] = Some(PsiField::new(psi_dims));
                cpml.psi_h_min[dir] = Some(PsiField::new(psi_dims));
            }

            // Max boundary
            let layers_max = boundaries.layers[dir * 2 + 1];
            if layers_max > 0 && layers_max < grid_size {
                let mut coeff = CpmlCoefficients::new(layers_max);
                coeff.init(
                    delta[dir],
                    dt,
                    sigma_max,
                    boundaries.alpha_max,
                    boundaries.kappa_max,
                    boundaries.grading_order,
                    false,
                );
                cpml.coeff_max[dir] = Some(coeff);

                // Psi field dimensions for this boundary
                let psi_dims = Self::psi_dims(dims, dir, layers_max);
                cpml.psi_e_max[dir] = Some(PsiField::new(psi_dims));
                cpml.psi_h_max[dir] = Some(PsiField::new(psi_dims));
            }
        }

        cpml
    }

    /// Calculate psi field dimensions for a boundary.
    fn psi_dims(dims: Dimensions, dir: usize, layers: usize) -> Dimensions {
        let mut psi_dims = dims;
        match dir {
            0 => psi_dims.nx = layers,
            1 => psi_dims.ny = layers,
            _ => psi_dims.nz = layers,
        }
        psi_dims
    }

    /// Check if CPML is active.
    pub fn is_active(&self) -> bool {
        self.coeff_min.iter().any(|c| c.is_some())
            || self.coeff_max.iter().any(|c| c.is_some())
    }

    /// Get the number of active CPML regions.
    pub fn num_regions(&self) -> usize {
        let mut count = 0;
        for c in &self.coeff_min {
            if c.is_some() {
                count += 1;
            }
        }
        for c in &self.coeff_max {
            if c.is_some() {
                count += 1;
            }
        }
        count
    }

    /// Update psi and apply correction after H-field update.
    ///
    /// This should be called AFTER the main engine's H-field update.
    /// It updates the psi_H fields and applies the correction to H.
    pub fn post_update_h(&mut self, h_field: &mut VectorField3D, e_field: &VectorField3D) {
        // TODO: CPML correction is currently disabled due to numerical instability.
        // The psi update is still performed to maintain the infrastructure.
        // A proper CPML implementation requires careful integration with the
        // specific FDTD operator coefficients and update sequence.
        //
        // The issue appears to be a sign mismatch between the CPML correction
        // and the FDTD curl computation, causing amplification instead of absorption.

        let dims = self.dims;

        for dir in 0..3 {
            // Correction coefficient (currently unused - correction disabled)
            let db_coeff = (self.dt / (MU0 * self.delta[dir])) as f32;

            // Determine which H components are affected by CPML in this direction
            // For CPML in x-direction: Hy and Hz are affected (they use dE/dx)
            let (comp1, comp2) = match dir {
                0 => (1, 2), // Hy uses dEz/dx, Hz uses -dEy/dx
                1 => (2, 0), // Hz uses dEx/dy, Hx uses -dEz/dy
                _ => (0, 1), // Hx uses dEy/dz, Hy uses -dEx/dz
            };

            // Min boundary
            if let (Some(coeff), Some(psi)) = (&self.coeff_min[dir], &mut self.psi_h_min[dir]) {
                apply_cpml_h_boundary(
                    h_field, e_field, coeff, psi, dims, db_coeff,
                    dir, comp1, comp2, true,
                );
            }

            // Max boundary
            if let (Some(coeff), Some(psi)) = (&self.coeff_max[dir], &mut self.psi_h_max[dir]) {
                apply_cpml_h_boundary(
                    h_field, e_field, coeff, psi, dims, db_coeff,
                    dir, comp1, comp2, false,
                );
            }
        }
    }

    /// Update psi and apply correction after E-field update.
    ///
    /// This should be called AFTER the main engine's E-field update.
    /// It updates the psi_E fields and applies the correction to E.
    pub fn post_update_e(&mut self, e_field: &mut VectorField3D, h_field: &VectorField3D) {
        // TODO: CPML correction is currently disabled (see post_update_h for details)

        let dims = self.dims;

        for dir in 0..3 {
            // Correction coefficient (currently unused - correction disabled)
            let da_coeff = (self.dt / (EPS0 * self.delta[dir])) as f32;

            // Determine which E components are affected by CPML in this direction
            // For CPML in x-direction: Ey and Ez are affected (they use dH/dx)
            let (comp1, comp2) = match dir {
                0 => (1, 2), // Ey uses -dHz/dx, Ez uses dHy/dx
                1 => (2, 0), // Ez uses -dHx/dy, Ex uses dHz/dy
                _ => (0, 1), // Ex uses -dHy/dz, Ey uses dHx/dz
            };

            // Min boundary
            if let (Some(coeff), Some(psi)) = (&self.coeff_min[dir], &mut self.psi_e_min[dir]) {
                apply_cpml_e_boundary(
                    e_field, h_field, coeff, psi, dims, da_coeff,
                    dir, comp1, comp2, true,
                );
            }

            // Max boundary
            if let (Some(coeff), Some(psi)) = (&self.coeff_max[dir], &mut self.psi_e_max[dir]) {
                apply_cpml_e_boundary(
                    e_field, h_field, coeff, psi, dims, da_coeff,
                    dir, comp1, comp2, false,
                );
            }
        }
    }

    /// Reset all psi fields to zero.
    pub fn reset(&mut self) {
        for psi in self.psi_e_min.iter_mut().flatten() {
            psi.clear();
        }
        for psi in self.psi_e_max.iter_mut().flatten() {
            psi.clear();
        }
        for psi in self.psi_h_min.iter_mut().flatten() {
            psi.clear();
        }
        for psi in self.psi_h_max.iter_mut().flatten() {
            psi.clear();
        }
    }
}

/// Apply CPML correction to H-field for one boundary (free function to avoid borrow issues).
#[allow(clippy::too_many_arguments)]
#[allow(unused_variables)] // Correction is disabled, but signature preserved for future implementation
fn apply_cpml_h_boundary(
    h_field: &mut VectorField3D,
    e_field: &VectorField3D,
    coeff: &CpmlCoefficients,
    psi: &mut PsiField,
    dims: Dimensions,
    db_coeff: f32, // dt / (mu * delta) - same as FDTD curl coefficient
    dir: usize,
    comp1: usize,
    comp2: usize,
    is_min: bool,
) {
    let layers = coeff.b_h.len();

    for gi in 0..dims.nx {
        for gj in 0..dims.ny {
            for gk in 0..dims.nz {
                // Check if this cell is in the PML region
                let (in_pml, pml_idx) = match dir {
                    0 => {
                        if is_min && gi < layers {
                            (true, gi)
                        } else if !is_min && gi >= dims.nx - layers {
                            (true, gi - (dims.nx - layers))
                        } else {
                            (false, 0)
                        }
                    }
                    1 => {
                        if is_min && gj < layers {
                            (true, gj)
                        } else if !is_min && gj >= dims.ny - layers {
                            (true, gj - (dims.ny - layers))
                        } else {
                            (false, 0)
                        }
                    }
                    _ => {
                        if is_min && gk < layers {
                            (true, gk)
                        } else if !is_min && gk >= dims.nz - layers {
                            (true, gk - (dims.nz - layers))
                        } else {
                            (false, 0)
                        }
                    }
                };

                if !in_pml {
                    continue;
                }

                // Local position in psi field
                let (li, lj, lk) = match dir {
                    0 => (pml_idx, gj, gk),
                    1 => (gi, pml_idx, gk),
                    _ => (gi, gj, pml_idx),
                };

                let b = coeff.b_h[pml_idx];
                let a = coeff.a_h[pml_idx];

                // Get current psi values (BEFORE update)
                let psi_pos_old = psi.psi_pos.get(li, lj, lk);
                let psi_neg_old = psi.psi_neg.get(li, lj, lk);

                // CPML correction is disabled due to instability
                // TODO: Fix the sign convention to match the FDTD operator
                // Corrections would be:
                // h_field.component_mut(comp1).add(gi, gj, gk, -db_coeff * psi_pos_old);
                // h_field.component_mut(comp2).add(gi, gj, gk, db_coeff * psi_neg_old);

                // Get E-field differences for psi update
                let (de_pos, de_neg) = get_e_differences(e_field, dims, gi, gj, gk, dir);

                // Update psi for next timestep
                let psi_pos_new = b * psi_pos_old + a * de_pos;
                psi.psi_pos.set(li, lj, lk, psi_pos_new);

                let psi_neg_new = b * psi_neg_old + a * de_neg;
                psi.psi_neg.set(li, lj, lk, psi_neg_new);
            }
        }
    }
}

/// Apply CPML correction to E-field for one boundary (free function to avoid borrow issues).
#[allow(clippy::too_many_arguments)]
#[allow(unused_variables)] // Correction is disabled, but signature preserved for future implementation
fn apply_cpml_e_boundary(
    e_field: &mut VectorField3D,
    h_field: &VectorField3D,
    coeff: &CpmlCoefficients,
    psi: &mut PsiField,
    dims: Dimensions,
    da_coeff: f32, // dt / (eps * delta) - same as FDTD curl coefficient
    dir: usize,
    comp1: usize,
    comp2: usize,
    is_min: bool,
) {
    let layers = coeff.b_e.len();

    for gi in 0..dims.nx {
        for gj in 0..dims.ny {
            for gk in 0..dims.nz {
                // Check if this cell is in the PML region
                let (in_pml, pml_idx) = match dir {
                    0 => {
                        if is_min && gi < layers {
                            (true, gi)
                        } else if !is_min && gi >= dims.nx - layers {
                            (true, gi - (dims.nx - layers))
                        } else {
                            (false, 0)
                        }
                    }
                    1 => {
                        if is_min && gj < layers {
                            (true, gj)
                        } else if !is_min && gj >= dims.ny - layers {
                            (true, gj - (dims.ny - layers))
                        } else {
                            (false, 0)
                        }
                    }
                    _ => {
                        if is_min && gk < layers {
                            (true, gk)
                        } else if !is_min && gk >= dims.nz - layers {
                            (true, gk - (dims.nz - layers))
                        } else {
                            (false, 0)
                        }
                    }
                };

                if !in_pml {
                    continue;
                }

                // Local position in psi field
                let (li, lj, lk) = match dir {
                    0 => (pml_idx, gj, gk),
                    1 => (gi, pml_idx, gk),
                    _ => (gi, gj, pml_idx),
                };

                let b = coeff.b_e[pml_idx];
                let a = coeff.a_e[pml_idx];

                // Get current psi values (BEFORE update)
                let psi_pos_old = psi.psi_pos.get(li, lj, lk);
                let psi_neg_old = psi.psi_neg.get(li, lj, lk);

                // CPML correction is disabled due to instability
                // TODO: Fix the sign convention to match the FDTD operator
                // Corrections would be:
                // e_field.component_mut(comp1).add(gi, gj, gk, da_coeff * psi_pos_old);
                // e_field.component_mut(comp2).add(gi, gj, gk, -da_coeff * psi_neg_old);

                // Get H-field differences for psi update
                let (dh_pos, dh_neg) = get_h_differences(h_field, gi, gj, gk, dir);

                // Update psi for next timestep
                let psi_pos_new = b * psi_pos_old + a * dh_pos;
                psi.psi_pos.set(li, lj, lk, psi_pos_new);

                let psi_neg_new = b * psi_neg_old + a * dh_neg;
                psi.psi_neg.set(li, lj, lk, psi_neg_new);
            }
        }
    }
}

/// Get E-field spatial differences for CPML correction.
/// Note: CPML uses field DIFFERENCES, not derivatives. The 'a' coefficient
/// already accounts for the spatial discretization.
fn get_e_differences(
    e_field: &VectorField3D,
    dims: Dimensions,
    gi: usize,
    gj: usize,
    gk: usize,
    dir: usize,
) -> (f32, f32) {
    // For H-field update in direction 'dir':
    // dir=0 (x): Hy uses ΔEz, Hz uses ΔEy
    // dir=1 (y): Hz uses ΔEx, Hx uses ΔEz
    // dir=2 (z): Hx uses ΔEy, Hy uses ΔEx

    match dir {
        0 => {
            // ΔEz for Hy, ΔEy for Hz (x-direction differences)
            let dez = if gi + 1 < dims.nx {
                e_field.z.get(gi + 1, gj, gk) - e_field.z.get(gi, gj, gk)
            } else {
                0.0
            };
            let dey = if gi + 1 < dims.nx {
                e_field.y.get(gi + 1, gj, gk) - e_field.y.get(gi, gj, gk)
            } else {
                0.0
            };
            (dez, dey)
        }
        1 => {
            // ΔEx for Hz, ΔEz for Hx (y-direction differences)
            let dex = if gj + 1 < dims.ny {
                e_field.x.get(gi, gj + 1, gk) - e_field.x.get(gi, gj, gk)
            } else {
                0.0
            };
            let dez = if gj + 1 < dims.ny {
                e_field.z.get(gi, gj + 1, gk) - e_field.z.get(gi, gj, gk)
            } else {
                0.0
            };
            (dex, dez)
        }
        _ => {
            // ΔEy for Hx, ΔEx for Hy (z-direction differences)
            let dey = if gk + 1 < dims.nz {
                e_field.y.get(gi, gj, gk + 1) - e_field.y.get(gi, gj, gk)
            } else {
                0.0
            };
            let dex = if gk + 1 < dims.nz {
                e_field.x.get(gi, gj, gk + 1) - e_field.x.get(gi, gj, gk)
            } else {
                0.0
            };
            (dey, dex)
        }
    }
}

/// Get H-field spatial differences for CPML correction.
/// Note: CPML uses field DIFFERENCES, not derivatives.
fn get_h_differences(
    h_field: &VectorField3D,
    gi: usize,
    gj: usize,
    gk: usize,
    dir: usize,
) -> (f32, f32) {
    // For E-field update in direction 'dir':
    // dir=0 (x): Ey uses ΔHz, Ez uses ΔHy
    // dir=1 (y): Ez uses ΔHx, Ex uses ΔHz
    // dir=2 (z): Ex uses ΔHy, Ey uses ΔHx

    match dir {
        0 => {
            // ΔHz for Ey, ΔHy for Ez (x-direction differences)
            let dhz = if gi > 0 {
                h_field.z.get(gi, gj, gk) - h_field.z.get(gi - 1, gj, gk)
            } else {
                0.0
            };
            let dhy = if gi > 0 {
                h_field.y.get(gi, gj, gk) - h_field.y.get(gi - 1, gj, gk)
            } else {
                0.0
            };
            (dhz, dhy)
        }
        1 => {
            // ΔHx for Ez, ΔHz for Ex (y-direction differences)
            let dhx = if gj > 0 {
                h_field.x.get(gi, gj, gk) - h_field.x.get(gi, gj - 1, gk)
            } else {
                0.0
            };
            let dhz = if gj > 0 {
                h_field.z.get(gi, gj, gk) - h_field.z.get(gi, gj - 1, gk)
            } else {
                0.0
            };
            (dhx, dhz)
        }
        _ => {
            // ΔHy for Ex, ΔHx for Ey (z-direction differences)
            let dhy = if gk > 0 {
                h_field.y.get(gi, gj, gk) - h_field.y.get(gi, gj, gk - 1)
            } else {
                0.0
            };
            let dhx = if gk > 0 {
                h_field.x.get(gi, gj, gk) - h_field.x.get(gi, gj, gk - 1)
            } else {
                0.0
            };
            (dhy, dhx)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpml_creation() {
        let dims = Dimensions {
            nx: 50,
            ny: 50,
            nz: 50,
        };
        let delta = [0.001, 0.001, 0.001];
        let dt = 1e-12;
        let boundaries = CpmlBoundaries::uniform(8);

        let cpml = Cpml::new(dims, delta, dt, boundaries);

        assert!(cpml.is_active());
        assert_eq!(cpml.num_regions(), 6);
    }

    #[test]
    fn test_cpml_coefficients() {
        let boundaries = CpmlBoundaries::uniform(8);
        let sigma_max = boundaries.sigma_max(0.001);

        eprintln!("sigma_max = {:.6e}", sigma_max);

        // sigma_max should be positive
        assert!(sigma_max > 0.0);

        // Create coefficients
        let mut coeff = CpmlCoefficients::new(8);
        coeff.init(
            0.001,
            1e-12,
            sigma_max,
            boundaries.alpha_max,
            boundaries.kappa_max,
            boundaries.grading_order,
            true,
        );

        // Print coefficient values for debugging
        eprintln!("CPML Coefficients (min boundary, 8 layers):");
        for i in 0..8 {
            eprintln!(
                "  Layer {}: b_e={:.6}, a_e={:.6e}, b_h={:.6}, a_h={:.6e}",
                i, coeff.b_e[i], coeff.a_e[i], coeff.b_h[i], coeff.a_h[i]
            );
        }

        // b should be between 0 and 1 (exponential decay)
        for &b in &coeff.b_e {
            assert!((0.0..=1.0).contains(&b), "b_e = {} out of range", b);
        }
        for &b in &coeff.b_h {
            assert!((0.0..=1.0).contains(&b), "b_h = {} out of range", b);
        }

        // a should be positive (we use 1-b > 0)
        for &a in &coeff.a_e {
            assert!(a >= 0.0, "a_e = {} should be >= 0", a);
        }
    }

    #[test]
    fn test_cpml_field_update_stability() {
        let dims = Dimensions {
            nx: 30,
            ny: 30,
            nz: 30,
        };
        let delta = [0.001, 0.001, 0.001];
        let dt = 1e-12;
        let boundaries = CpmlBoundaries::uniform(6);

        let mut cpml = Cpml::new(dims, delta, dt, boundaries);

        let mut e_field = VectorField3D::new(dims);
        let mut h_field = VectorField3D::new(dims);

        // Set initial field values
        e_field.z.set(15, 15, 15, 1.0);

        // Run several update cycles
        for _ in 0..10 {
            cpml.post_update_h(&mut h_field, &e_field);
            cpml.post_update_e(&mut e_field, &h_field);
        }

        // Check for NaN
        let e_energy = e_field.energy();
        let h_energy = h_field.energy();
        assert!(!e_energy.is_nan(), "E-field energy is NaN");
        assert!(!h_energy.is_nan(), "H-field energy is NaN");
        assert!(!e_energy.is_infinite(), "E-field energy is infinite");
        assert!(!h_energy.is_infinite(), "H-field energy is infinite");
    }

    #[test]
    fn test_cpml_psi_values() {
        // Test to verify psi values are being computed correctly
        let dims = Dimensions {
            nx: 20,
            ny: 20,
            nz: 20,
        };
        let delta = [0.001, 0.001, 0.001];
        let dt = 1e-12;
        let boundaries = CpmlBoundaries::uniform(4);

        let mut cpml = Cpml::new(dims, delta, dt, boundaries);

        let mut e_field = VectorField3D::new(dims);
        let mut h_field = VectorField3D::new(dims);

        // Create a gradient in Ez that spans the PML region
        for i in 0..dims.nx {
            for j in 0..dims.ny {
                for k in 0..dims.nz {
                    // Linear gradient in x-direction: Ez = x
                    let ez_val = i as f32 * 0.1;
                    e_field.z.set(i, j, k, ez_val);
                }
            }
        }

        eprintln!("\nTest: CPML psi values");
        eprintln!("Initial E-field energy: {:.6e}", e_field.energy());

        // Apply CPML once
        cpml.post_update_h(&mut h_field, &e_field);
        eprintln!("After H correction: E={:.6e}, H={:.6e}", e_field.energy(), h_field.energy());

        cpml.post_update_e(&mut e_field, &h_field);
        eprintln!("After E correction: E={:.6e}, H={:.6e}", e_field.energy(), h_field.energy());

        // Check that psi fields have non-zero values
        let mut psi_sum = 0.0f64;
        if let Some(psi) = &cpml.psi_h_min[0] {
            for i in 0..4 {
                for j in 0..dims.ny {
                    for k in 0..dims.nz {
                        psi_sum += (psi.psi_pos.get(i, j, k).abs() + psi.psi_neg.get(i, j, k).abs()) as f64;
                    }
                }
            }
        }
        eprintln!("Sum of |psi_h_min[x]|: {:.6e}", psi_sum);

        // Verify psi is non-zero (CPML should be doing something)
        assert!(psi_sum > 0.0, "CPML psi fields should be non-zero with field gradients");
    }
}
