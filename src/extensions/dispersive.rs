//! Dispersive Material Models.
//!
//! Implements frequency-dependent material models using Auxiliary Differential
//! Equation (ADE) methods:
//! - Lorentz model: dielectric resonance
//! - Drude model: metals and plasmas
//! - Debye model: polar molecules

use crate::arrays::VectorField3D;
use crate::constants::EPS0;
use std::f64::consts::PI;

/// Lorentz oscillator parameters.
#[derive(Debug, Clone)]
pub struct LorentzParams {
    /// Resonance frequency (rad/s)
    pub omega_0: f64,
    /// Damping frequency (rad/s)
    pub gamma: f64,
    /// Oscillator strength (delta epsilon)
    pub delta_eps: f64,
}

impl LorentzParams {
    /// Create from frequency in Hz.
    pub fn from_hz(f0: f64, gamma: f64, delta_eps: f64) -> Self {
        Self {
            omega_0: 2.0 * PI * f0,
            gamma: 2.0 * PI * gamma,
            delta_eps,
        }
    }
}

/// Drude model parameters.
#[derive(Debug, Clone)]
pub struct DrudeParams {
    /// Plasma frequency (rad/s)
    pub omega_p: f64,
    /// Collision frequency (rad/s)
    pub gamma: f64,
}

impl DrudeParams {
    /// Create from frequency in Hz.
    pub fn from_hz(fp: f64, gamma: f64) -> Self {
        Self {
            omega_p: 2.0 * PI * fp,
            gamma: 2.0 * PI * gamma,
        }
    }

    /// Create for a typical metal (e.g., gold, silver).
    pub fn metal(plasma_freq_hz: f64, damping_freq_hz: f64) -> Self {
        Self::from_hz(plasma_freq_hz, damping_freq_hz)
    }
}

/// Debye relaxation parameters.
#[derive(Debug, Clone)]
pub struct DebyeParams {
    /// Relaxation time (seconds)
    pub tau: f64,
    /// Static permittivity increment
    pub delta_eps: f64,
}

impl DebyeParams {
    /// Create from relaxation time in seconds.
    pub fn new(tau: f64, delta_eps: f64) -> Self {
        Self { tau, delta_eps }
    }
}

/// ADE storage for dispersive materials.
#[derive(Debug)]
struct AdeStorage {
    /// Position indices where this material exists
    positions: Vec<[usize; 3]>,
    /// ADE field components for each position (for each polarization direction)
    j_ade: [Vec<f32>; 3], // Auxiliary current
    /// Update coefficients
    a_coeff: f32, // J_new = a * J_old + b * E
    b_coeff: f32,
    c_coeff: f32, // E contribution from J
}

impl AdeStorage {
    fn new(positions: Vec<[usize; 3]>, a: f32, b: f32, c: f32) -> Self {
        let n = positions.len();
        Self {
            positions,
            j_ade: [vec![0.0; n], vec![0.0; n], vec![0.0; n]],
            a_coeff: a,
            b_coeff: b,
            c_coeff: c,
        }
    }
}

/// Trait for dispersive material models.
pub trait DispersiveMaterial: Send + Sync {
    /// Get the material name.
    fn name(&self) -> &str;

    /// Pre-update hook for E-field.
    fn pre_update_e(&mut self, e_field: &VectorField3D);

    /// Post-update hook for E-field.
    fn post_update_e(&mut self, e_field: &mut VectorField3D);

    /// Reset internal state.
    fn reset(&mut self);
}

/// Lorentz dispersive material.
#[allow(dead_code)]
pub struct LorentzMaterial {
    /// Material parameters
    params: LorentzParams,
    /// Timestep
    dt: f64,
    /// ADE storage
    ade: AdeStorage,
}

impl LorentzMaterial {
    /// Create a new Lorentz material.
    ///
    /// # Arguments
    /// * `params` - Lorentz oscillator parameters
    /// * `dt` - Simulation timestep
    /// * `positions` - Grid positions where material exists
    pub fn new(params: LorentzParams, dt: f64, positions: Vec<[usize; 3]>) -> Self {
        // Lorentz ADE coefficients (second-order oscillator)
        // d²P/dt² + gamma*dP/dt + omega_0²*P = eps0*delta_eps*omega_0²*E
        //
        // Discretized using central differences:
        // P^{n+1} = a*P^n + b*P^{n-1} + c*E^n
        //
        // a = (2 - omega_0²*dt²) / (1 + gamma*dt/2)
        // b = (gamma*dt/2 - 1) / (1 + gamma*dt/2)
        // c = eps0*delta_eps*omega_0²*dt² / (1 + gamma*dt/2)

        let omega_0 = params.omega_0;
        let gamma = params.gamma;
        let delta_eps = params.delta_eps;

        let denom = 1.0 + gamma * dt / 2.0;
        let a = ((2.0 - omega_0 * omega_0 * dt * dt) / denom) as f32;
        let b = ((gamma * dt / 2.0 - 1.0) / denom) as f32;
        let c = (EPS0 * delta_eps * omega_0 * omega_0 * dt * dt / denom) as f32;

        Self {
            params,
            dt,
            ade: AdeStorage::new(positions, a, b, c),
        }
    }
}

impl DispersiveMaterial for LorentzMaterial {
    fn name(&self) -> &str {
        "Lorentz"
    }

    fn pre_update_e(&mut self, _e_field: &VectorField3D) {
        // Lorentz uses post-update only
    }

    fn post_update_e(&mut self, e_field: &mut VectorField3D) {
        let a = self.ade.a_coeff;
        let b = self.ade.b_coeff;
        let c = self.ade.c_coeff;

        for (idx, pos) in self.ade.positions.iter().enumerate() {
            for dir in 0..3 {
                let field = match dir {
                    0 => &mut e_field.x,
                    1 => &mut e_field.y,
                    _ => &mut e_field.z,
                };

                let e = field.get(pos[0], pos[1], pos[2]);
                let j_old = self.ade.j_ade[dir][idx];

                // Update auxiliary current (simplified first-order for now)
                let j_new = a * j_old + c * e;
                self.ade.j_ade[dir][idx] = j_new;

                // Subtract polarization current contribution from E
                let e_new = e - b * j_new;
                field.set(pos[0], pos[1], pos[2], e_new);
            }
        }
    }

    fn reset(&mut self) {
        for dir in 0..3 {
            self.ade.j_ade[dir].fill(0.0);
        }
    }
}

impl LorentzMaterial {
    /// Generate GPU extension data for shader compilation.
    ///
    /// Returns None if no positions are defined.
    pub fn gpu_data(&self) -> Option<crate::extensions::GpuExtensionData> {
        if self.ade.positions.is_empty() {
            return None;
        }

        generate_dispersive_gpu_data(&self.ade, DispersiveType::Lorentz, self.dt as f32)
    }
}

/// Type of dispersive material for GPU shader generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum DispersiveType {
    /// Lorentz oscillator model for dielectric resonance.
    Lorentz = 0,
    /// Drude model for metals and plasmas.
    Drude = 1,
    /// Debye relaxation model for polar molecules.
    Debye = 2,
}

/// Generate GPU extension data for dispersive materials.
fn generate_dispersive_gpu_data(
    ade: &AdeStorage,
    material_type: DispersiveType,
    dt: f32,
) -> Option<crate::extensions::GpuExtensionData> {
    // GPU metadata structure
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct GpuDispersiveMeta {
        // Number of positions with dispersive material
        num_positions: u32,
        // Material type (0=Lorentz, 1=Drude, 2=Debye)
        material_type: u32,
        // Timestep
        dt: f32,
        // Padding
        _padding: u32,
    }

    // GPU ADE coefficients
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct GpuAdeCoeffs {
        a: f32,
        b: f32,
        c: f32,
        _padding: f32,
    }

    let meta = GpuDispersiveMeta {
        num_positions: ade.positions.len() as u32,
        material_type: material_type as u32,
        dt,
        _padding: 0,
    };

    let coeffs = GpuAdeCoeffs {
        a: ade.a_coeff,
        b: ade.b_coeff,
        c: ade.c_coeff,
        _padding: 0.0,
    };

    // Pack positions as [i, j, k, padding] per position
    let mut positions_data: Vec<u8> = Vec::with_capacity(ade.positions.len() * 16);
    for pos in &ade.positions {
        let gpu_pos: [u32; 4] = [pos[0] as u32, pos[1] as u32, pos[2] as u32, 0];
        positions_data.extend_from_slice(bytemuck::bytes_of(&gpu_pos));
    }

    // Pack auxiliary current storage (3 components per position)
    // Layout: [j_x[0], j_x[1], ..., j_y[0], j_y[1], ..., j_z[0], j_z[1], ...]
    let mut j_data: Vec<u8> = Vec::with_capacity(ade.positions.len() * 3 * 4);
    for dir in 0..3 {
        for &j in &ade.j_ade[dir] {
            j_data.extend_from_slice(bytemuck::bytes_of(&j));
        }
    }

    let shader_code = generate_dispersive_shader();

    Some(crate::extensions::GpuExtensionData {
        shader_code,
        buffers: vec![
            crate::extensions::GpuBufferDescriptor {
                label: "Dispersive Metadata".to_string(),
                data: bytemuck::bytes_of(&meta).to_vec(),
                usage: wgpu::BufferUsages::UNIFORM,
                binding: 20,
            },
            crate::extensions::GpuBufferDescriptor {
                label: "Dispersive Coefficients".to_string(),
                data: bytemuck::bytes_of(&coeffs).to_vec(),
                usage: wgpu::BufferUsages::UNIFORM,
                binding: 21,
            },
            crate::extensions::GpuBufferDescriptor {
                label: "Dispersive Positions".to_string(),
                data: positions_data,
                usage: wgpu::BufferUsages::STORAGE,
                binding: 22,
            },
            crate::extensions::GpuBufferDescriptor {
                label: "Dispersive Auxiliary Current".to_string(),
                data: j_data,
                usage: wgpu::BufferUsages::STORAGE,
                binding: 23,
            },
        ],
        bind_group_entries: vec![
            wgpu::BindGroupLayoutEntry {
                binding: 20,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 21,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 22,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 23,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}

/// Generate WGSL shader code for dispersive materials.
fn generate_dispersive_shader() -> String {
    r#"
// Dispersive Material Extension Declarations
struct DispersiveMeta {
    num_positions: u32,
    material_type: u32,  // 0=Lorentz, 1=Drude, 2=Debye
    dt: f32,
    _padding: u32,
}

struct DispersiveCoeffs {
    a: f32,
    b: f32,
    c: f32,
    _padding: f32,
}

@group(0) @binding(20) var<uniform> disp_meta: DispersiveMeta;
@group(0) @binding(21) var<uniform> disp_coeffs: DispersiveCoeffs;
@group(0) @binding(22) var<storage, read> disp_positions: array<vec4<u32>>;  // [i, j, k, pad]
@group(0) @binding(23) var<storage, read_write> disp_j: array<f32>;  // [j_x..., j_y..., j_z...]

// Material type constants
const DISP_LORENTZ: u32 = 0u;
const DISP_DRUDE: u32 = 1u;
const DISP_DEBYE: u32 = 2u;

// Get position for a dispersive material cell
fn disp_get_position(idx: u32) -> vec3<u32> {
    let pos = disp_positions[idx];
    return vec3<u32>(pos.x, pos.y, pos.z);
}

// Get auxiliary current for a position and direction
fn disp_get_j(idx: u32, dir: u32) -> f32 {
    return disp_j[dir * disp_meta.num_positions + idx];
}

// Set auxiliary current for a position and direction
fn disp_set_j(idx: u32, dir: u32, val: f32) {
    disp_j[dir * disp_meta.num_positions + idx] = val;
}

// Check if a global position matches any dispersive material position
fn disp_find_position(gi: u32, gj: u32, gk: u32) -> i32 {
    for (var idx = 0u; idx < disp_meta.num_positions; idx++) {
        let pos = disp_get_position(idx);
        if (pos.x == gi && pos.y == gj && pos.z == gk) {
            return i32(idx);
        }
    }
    return -1;
}

// Apply dispersive material update to E-field
// Returns the modified E-field value
fn disp_apply_update(e: f32, idx: u32, dir: u32) -> f32 {
    let j_old = disp_get_j(idx, dir);
    var j_new: f32;
    var e_new: f32;

    switch (disp_meta.material_type) {
        case DISP_LORENTZ: {
            // J_new = a * J_old + c * E
            j_new = disp_coeffs.a * j_old + disp_coeffs.c * e;
            // E_new = E - b * J_new
            e_new = e - disp_coeffs.b * j_new;
        }
        case DISP_DRUDE: {
            // J_new = a * J_old + b * E
            j_new = disp_coeffs.a * j_old + disp_coeffs.b * e;
            // E_new = E - c * J_new
            e_new = e - disp_coeffs.c * j_new;
        }
        case DISP_DEBYE: {
            // P_new = a * P_old + b * E
            let p_old = j_old;  // Using j storage for polarization
            let p_new = disp_coeffs.a * p_old + disp_coeffs.b * e;
            j_new = p_new;
            // dP/dt = (P_new - P_old) / dt
            let dp_dt = (p_new - p_old) / disp_meta.dt;
            // E_new = E - c * dP/dt
            e_new = e - disp_coeffs.c * dp_dt;
        }
        default: {
            j_new = j_old;
            e_new = e;
        }
    }

    disp_set_j(idx, dir, j_new);
    return e_new;
}
"#
    .to_string()
}

/// Drude dispersive material (metals, plasmas).
#[allow(dead_code)]
pub struct DrudeMaterial {
    /// Material parameters
    params: DrudeParams,
    /// Timestep
    dt: f64,
    /// ADE storage
    ade: AdeStorage,
}

impl DrudeMaterial {
    /// Create a new Drude material.
    pub fn new(params: DrudeParams, dt: f64, positions: Vec<[usize; 3]>) -> Self {
        // Drude model: epsilon(omega) = eps_inf - omega_p² / (omega² + i*gamma*omega)
        //
        // ADE: dJ/dt + gamma*J = eps0*omega_p²*E
        //
        // Discretized: J^{n+1} = a*J^n + b*E^{n+1/2}
        // a = (1 - gamma*dt/2) / (1 + gamma*dt/2)
        // b = eps0*omega_p²*dt / (1 + gamma*dt/2)

        let omega_p = params.omega_p;
        let gamma = params.gamma;

        let denom = 1.0 + gamma * dt / 2.0;
        let a = ((1.0 - gamma * dt / 2.0) / denom) as f32;
        let b = (EPS0 * omega_p * omega_p * dt / denom) as f32;
        let c = (dt / EPS0) as f32; // E correction coefficient

        Self {
            params,
            dt,
            ade: AdeStorage::new(positions, a, b, c),
        }
    }
}

impl DispersiveMaterial for DrudeMaterial {
    fn name(&self) -> &str {
        "Drude"
    }

    fn pre_update_e(&mut self, _e_field: &VectorField3D) {
        // Drude uses post-update only
    }

    fn post_update_e(&mut self, e_field: &mut VectorField3D) {
        let a = self.ade.a_coeff;
        let b = self.ade.b_coeff;
        let c = self.ade.c_coeff;

        for (idx, pos) in self.ade.positions.iter().enumerate() {
            for dir in 0..3 {
                let field = match dir {
                    0 => &mut e_field.x,
                    1 => &mut e_field.y,
                    _ => &mut e_field.z,
                };

                let e = field.get(pos[0], pos[1], pos[2]);
                let j_old = self.ade.j_ade[dir][idx];

                // Update auxiliary current
                let j_new = a * j_old + b * e;
                self.ade.j_ade[dir][idx] = j_new;

                // Subtract current contribution from E
                let e_new = e - c * j_new;
                field.set(pos[0], pos[1], pos[2], e_new);
            }
        }
    }

    fn reset(&mut self) {
        for dir in 0..3 {
            self.ade.j_ade[dir].fill(0.0);
        }
    }
}

impl DrudeMaterial {
    /// Generate GPU extension data for shader compilation.
    ///
    /// Returns None if no positions are defined.
    pub fn gpu_data(&self) -> Option<crate::extensions::GpuExtensionData> {
        if self.ade.positions.is_empty() {
            return None;
        }

        generate_dispersive_gpu_data(&self.ade, DispersiveType::Drude, self.dt as f32)
    }
}

/// Debye dispersive material (polar molecules).
#[allow(dead_code)]
pub struct DebyeMaterial {
    /// Material parameters
    params: DebyeParams,
    /// Timestep
    dt: f64,
    /// ADE storage
    ade: AdeStorage,
}

impl DebyeMaterial {
    /// Create a new Debye material.
    pub fn new(params: DebyeParams, dt: f64, positions: Vec<[usize; 3]>) -> Self {
        // Debye model: epsilon(omega) = eps_inf + delta_eps / (1 + i*omega*tau)
        //
        // ADE: tau*dP/dt + P = eps0*delta_eps*E
        //
        // Discretized: P^{n+1} = a*P^n + b*E^n
        // a = (2*tau - dt) / (2*tau + dt)
        // b = 2*eps0*delta_eps*dt / (2*tau + dt)

        let tau = params.tau;
        let delta_eps = params.delta_eps;

        let denom = 2.0 * tau + dt;
        let a = ((2.0 * tau - dt) / denom) as f32;
        let b = (2.0 * EPS0 * delta_eps * dt / denom) as f32;
        let c = (dt / EPS0) as f32;

        Self {
            params,
            dt,
            ade: AdeStorage::new(positions, a, b, c),
        }
    }
}

impl DispersiveMaterial for DebyeMaterial {
    fn name(&self) -> &str {
        "Debye"
    }

    fn pre_update_e(&mut self, _e_field: &VectorField3D) {
        // Debye uses post-update only
    }

    fn post_update_e(&mut self, e_field: &mut VectorField3D) {
        let a = self.ade.a_coeff;
        let b = self.ade.b_coeff;
        let c = self.ade.c_coeff;

        for (idx, pos) in self.ade.positions.iter().enumerate() {
            for dir in 0..3 {
                let field = match dir {
                    0 => &mut e_field.x,
                    1 => &mut e_field.y,
                    _ => &mut e_field.z,
                };

                let e = field.get(pos[0], pos[1], pos[2]);
                let p_old = self.ade.j_ade[dir][idx];

                // Update polarization
                let p_new = a * p_old + b * e;
                self.ade.j_ade[dir][idx] = p_new;

                // Apply polarization current correction
                let dp_dt = (p_new - p_old) / self.dt as f32;
                let e_new = e - c * dp_dt;
                field.set(pos[0], pos[1], pos[2], e_new);
            }
        }
    }

    fn reset(&mut self) {
        for dir in 0..3 {
            self.ade.j_ade[dir].fill(0.0);
        }
    }
}

impl DebyeMaterial {
    /// Generate GPU extension data for shader compilation.
    ///
    /// Returns None if no positions are defined.
    pub fn gpu_data(&self) -> Option<crate::extensions::GpuExtensionData> {
        if self.ade.positions.is_empty() {
            return None;
        }

        generate_dispersive_gpu_data(&self.ade, DispersiveType::Debye, self.dt as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arrays::Dimensions;

    #[test]
    fn test_lorentz_params() {
        let params = LorentzParams::from_hz(1e9, 1e8, 2.0);
        assert!(params.omega_0 > 0.0);
        assert!(params.gamma > 0.0);
    }

    #[test]
    fn test_drude_params() {
        // Typical gold parameters
        let params = DrudeParams::metal(2.18e15, 6.45e12);
        assert!(params.omega_p > 0.0);
    }

    #[test]
    fn test_lorentz_material() {
        let dims = Dimensions {
            nx: 10,
            ny: 10,
            nz: 10,
        };
        let dt = 1e-15;
        let params = LorentzParams::from_hz(1e14, 1e13, 1.0);

        let positions = vec![[5, 5, 5], [5, 5, 6]];
        let mut material = LorentzMaterial::new(params, dt, positions);

        let mut e_field = VectorField3D::new(dims);
        e_field.x.set(5, 5, 5, 1.0);

        // Run several updates to accumulate effect
        for _ in 0..100 {
            material.post_update_e(&mut e_field);
            e_field.x.set(5, 5, 5, 1.0); // Reset E to simulate continuous excitation
        }

        // After multiple updates with excitation, auxiliary currents should be non-zero
        assert!(
            material.ade.j_ade[0][0] != 0.0,
            "Auxiliary current should build up"
        );
    }

    #[test]
    fn test_drude_material() {
        let dims = Dimensions {
            nx: 10,
            ny: 10,
            nz: 10,
        };
        let dt = 1e-16;
        let params = DrudeParams::metal(2e15, 1e13);

        let positions = vec![[3, 3, 3]];
        let mut material = DrudeMaterial::new(params, dt, positions);

        let mut e_field = VectorField3D::new(dims);
        e_field.z.set(3, 3, 3, 1.0);

        material.post_update_e(&mut e_field);

        // Field should be modified by conduction current
        let e = e_field.z.get(3, 3, 3);
        assert!(e != 1.0);
    }

    #[test]
    fn test_debye_material() {
        let dims = Dimensions {
            nx: 10,
            ny: 10,
            nz: 10,
        };
        let dt = 1e-12;
        let params = DebyeParams::new(1e-11, 10.0); // Water-like

        let positions = vec![[4, 4, 4]];
        let mut material = DebyeMaterial::new(params, dt, positions);

        let mut e_field = VectorField3D::new(dims);
        e_field.y.set(4, 4, 4, 1.0);

        material.post_update_e(&mut e_field);
    }

    #[test]
    fn test_material_reset() {
        let dt = 1e-15;
        let params = LorentzParams::from_hz(1e14, 1e13, 1.0);
        let positions = vec![[0, 0, 0]];
        let mut material = LorentzMaterial::new(params, dt, positions);

        // Modify internal state
        material.ade.j_ade[0][0] = 1.0;

        // Reset should clear it
        material.reset();
        assert_eq!(material.ade.j_ade[0][0], 0.0);
    }
}
