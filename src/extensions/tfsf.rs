//! Total-Field/Scattered-Field (TF/SF) Boundary.
//!
//! Implements the TF/SF technique for injecting plane waves into the
//! FDTD domain while separating total and scattered field regions.

use crate::arrays::{Dimensions, VectorField3D};
use crate::constants::C0;
use std::f64::consts::PI;

/// Direction of wave propagation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PropagationDirection {
    /// Propagating in +x direction
    XPlus,
    /// Propagating in -x direction
    XMinus,
    /// Propagating in +y direction
    YPlus,
    /// Propagating in -y direction
    YMinus,
    /// Propagating in +z direction
    ZPlus,
    /// Propagating in -z direction
    ZMinus,
}

impl PropagationDirection {
    /// Get the axis index (0, 1, 2 for x, y, z).
    pub fn axis(&self) -> usize {
        match self {
            Self::XPlus | Self::XMinus => 0,
            Self::YPlus | Self::YMinus => 1,
            Self::ZPlus | Self::ZMinus => 2,
        }
    }

    /// Check if propagating in positive direction.
    pub fn is_positive(&self) -> bool {
        matches!(self, Self::XPlus | Self::YPlus | Self::ZPlus)
    }
}

/// Polarization of the incident wave.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Polarization {
    /// E-field along first perpendicular axis
    Transverse1,
    /// E-field along second perpendicular axis
    Transverse2,
}

/// Configuration for TF/SF boundary.
#[derive(Debug, Clone)]
pub struct TfsfConfig {
    /// Propagation direction
    pub direction: PropagationDirection,
    /// Polarization
    pub polarization: Polarization,
    /// Center frequency (Hz)
    pub frequency: f64,
    /// Amplitude (V/m)
    pub amplitude: f64,
    /// TF/SF boundary position (layer number from boundary)
    pub boundary_layer: usize,
    /// Phase velocity (defaults to c0)
    pub phase_velocity: Option<f64>,
}

impl TfsfConfig {
    /// Create a TF/SF configuration.
    pub fn new(direction: PropagationDirection, frequency: f64, amplitude: f64) -> Self {
        Self {
            direction,
            polarization: Polarization::Transverse1,
            frequency,
            amplitude,
            boundary_layer: 10,
            phase_velocity: None,
        }
    }

    /// Set polarization.
    pub fn with_polarization(mut self, pol: Polarization) -> Self {
        self.polarization = pol;
        self
    }

    /// Set boundary layer position.
    pub fn with_boundary_layer(mut self, layer: usize) -> Self {
        self.boundary_layer = layer;
        self
    }

    /// Set phase velocity.
    pub fn with_phase_velocity(mut self, v: f64) -> Self {
        self.phase_velocity = Some(v);
        self
    }
}

/// Internal storage for TF/SF field values.
#[derive(Debug)]
struct TfsfStorage {
    /// E-field component on source plane
    e_src: Vec<f32>,
    /// H-field component on source plane
    h_src: Vec<f32>,
    /// 1D auxiliary E-field for plane wave
    e_1d: Vec<f32>,
    /// 1D auxiliary H-field for plane wave
    h_1d: Vec<f32>,
}

/// Total-Field/Scattered-Field boundary.
pub struct TfsfBoundary {
    /// Grid dimensions
    dims: Dimensions,
    /// Cell sizes
    delta: [f64; 3],
    /// Timestep
    dt: f64,
    /// Configuration
    config: TfsfConfig,
    /// Storage for field values
    storage: TfsfStorage,
    /// Propagation axis
    prop_axis: usize,
    /// E-field axis
    e_axis: usize,
    /// H-field axis
    h_axis: usize,
    /// Boundary position
    boundary_pos: usize,
    /// Phase velocity
    phase_velocity: f64,
    /// Current timestep
    timestep: u64,
}

impl TfsfBoundary {
    /// Create a new TF/SF boundary.
    pub fn new(dims: Dimensions, delta: [f64; 3], dt: f64, config: TfsfConfig) -> Self {
        let prop_axis = config.direction.axis();
        let v = config.phase_velocity.unwrap_or(C0);

        // Determine E and H field axes based on polarization
        let (e_axis, h_axis) = match config.polarization {
            Polarization::Transverse1 => ((prop_axis + 1) % 3, (prop_axis + 2) % 3),
            Polarization::Transverse2 => ((prop_axis + 2) % 3, (prop_axis + 1) % 3),
        };

        // Boundary position
        let boundary_pos = config.boundary_layer;

        // Calculate plane size
        let plane_size = match prop_axis {
            0 => dims.ny * dims.nz,
            1 => dims.nx * dims.nz,
            _ => dims.nx * dims.ny,
        };

        // 1D grid size for auxiliary propagation
        let aux_size = match prop_axis {
            0 => dims.nx,
            1 => dims.ny,
            _ => dims.nz,
        };

        let storage = TfsfStorage {
            e_src: vec![0.0; plane_size],
            h_src: vec![0.0; plane_size],
            e_1d: vec![0.0; aux_size + 1],
            h_1d: vec![0.0; aux_size + 1],
        };

        Self {
            dims,
            delta,
            dt,
            config,
            storage,
            prop_axis,
            e_axis,
            h_axis,
            boundary_pos,
            phase_velocity: v,
            timestep: 0,
        }
    }

    /// Check if TF/SF is active.
    pub fn is_active(&self) -> bool {
        true
    }

    /// Calculate incident E-field at given time.
    #[allow(dead_code)]
    fn incident_e(&self, t: f64) -> f64 {
        let omega = 2.0 * PI * self.config.frequency;
        self.config.amplitude * (omega * t).sin()
    }

    /// Calculate incident H-field at given time.
    fn incident_h(&self, t: f64) -> f64 {
        let omega = 2.0 * PI * self.config.frequency;
        // H = E / eta, with 90-degree phase shift for plane wave
        let eta = (crate::constants::MU0 / crate::constants::EPS0).sqrt();
        (self.config.amplitude / eta) * (omega * t).sin()
    }

    /// Update 1D auxiliary grid.
    fn update_1d_grid(&mut self) {
        let dt = self.dt;
        let dx = self.delta[self.prop_axis];
        let c = self.phase_velocity;

        // 1D FDTD update
        // H update
        for i in 0..self.storage.h_1d.len() - 1 {
            let db = -(dt / (crate::constants::MU0 * dx)) as f32;
            self.storage.h_1d[i] += db * (self.storage.e_1d[i + 1] - self.storage.e_1d[i]);
        }

        // Inject source at left boundary
        let t = self.timestep as f64 * dt;
        self.storage.h_1d[0] = self.incident_h(t) as f32;

        // E update
        for i in 1..self.storage.e_1d.len() {
            let cb = (dt / (crate::constants::EPS0 * dx)) as f32;
            self.storage.e_1d[i] += cb * (self.storage.h_1d[i] - self.storage.h_1d[i - 1]);
        }

        // ABC at right boundary (simple first-order Mur)
        let coeff = ((c * dt - dx) / (c * dt + dx)) as f32;
        let n = self.storage.e_1d.len() - 1;
        self.storage.e_1d[n] = coeff * self.storage.e_1d[n - 1];
    }

    /// Pre-update for H-field: Add incident field.
    pub fn pre_update_h(&mut self, h_field: &mut VectorField3D) {
        // Update 1D auxiliary grid
        self.update_1d_grid();

        // Get incident field at boundary
        let e_inc = if self.boundary_pos < self.storage.e_1d.len() {
            self.storage.e_1d[self.boundary_pos]
        } else {
            0.0
        };

        // Add correction to H-field at TF/SF boundary
        // This subtracts the incident field contribution from the scattered field region
        let db = -(self.dt / (crate::constants::MU0 * self.delta[self.prop_axis])) as f32;

        let h_field_comp = match self.h_axis {
            0 => &mut h_field.x,
            1 => &mut h_field.y,
            _ => &mut h_field.z,
        };

        // Apply correction on the boundary plane
        self.apply_h_correction(h_field_comp, e_inc, db);
    }

    /// Apply H-field correction at boundary.
    fn apply_h_correction(&self, h_field: &mut crate::arrays::Field3D, e_inc: f32, db: f32) {
        let pos = self.boundary_pos;

        match self.prop_axis {
            0 => {
                // X propagation: correct H on YZ plane at x=pos
                for j in 0..self.dims.ny {
                    for k in 0..self.dims.nz {
                        let h_old = h_field.get(pos, j, k);
                        h_field.set(pos, j, k, h_old - db * e_inc);
                    }
                }
            }
            1 => {
                // Y propagation: correct H on XZ plane
                for i in 0..self.dims.nx {
                    for k in 0..self.dims.nz {
                        let h_old = h_field.get(i, pos, k);
                        h_field.set(i, pos, k, h_old - db * e_inc);
                    }
                }
            }
            _ => {
                // Z propagation: correct H on XY plane
                for i in 0..self.dims.nx {
                    for j in 0..self.dims.ny {
                        let h_old = h_field.get(i, j, pos);
                        h_field.set(i, j, pos, h_old - db * e_inc);
                    }
                }
            }
        }
    }

    /// Pre-update for E-field: Add incident field.
    pub fn pre_update_e(&mut self, e_field: &mut VectorField3D) {
        // Get incident H-field at boundary
        let h_inc = if self.boundary_pos < self.storage.h_1d.len() {
            self.storage.h_1d[self.boundary_pos]
        } else {
            0.0
        };

        // Add correction to E-field
        let cb = (self.dt / (crate::constants::EPS0 * self.delta[self.prop_axis])) as f32;

        let e_field_comp = match self.e_axis {
            0 => &mut e_field.x,
            1 => &mut e_field.y,
            _ => &mut e_field.z,
        };

        self.apply_e_correction(e_field_comp, h_inc, cb);

        self.timestep += 1;
    }

    /// Apply E-field correction at boundary.
    fn apply_e_correction(&self, e_field: &mut crate::arrays::Field3D, h_inc: f32, cb: f32) {
        let pos = self.boundary_pos;

        match self.prop_axis {
            0 => {
                for j in 0..self.dims.ny {
                    for k in 0..self.dims.nz {
                        let e_old = e_field.get(pos, j, k);
                        e_field.set(pos, j, k, e_old + cb * h_inc);
                    }
                }
            }
            1 => {
                for i in 0..self.dims.nx {
                    for k in 0..self.dims.nz {
                        let e_old = e_field.get(i, pos, k);
                        e_field.set(i, pos, k, e_old + cb * h_inc);
                    }
                }
            }
            _ => {
                for i in 0..self.dims.nx {
                    for j in 0..self.dims.ny {
                        let e_old = e_field.get(i, j, pos);
                        e_field.set(i, j, pos, e_old + cb * h_inc);
                    }
                }
            }
        }
    }

    /// Reset the TF/SF boundary.
    pub fn reset(&mut self) {
        self.storage.e_src.fill(0.0);
        self.storage.h_src.fill(0.0);
        self.storage.e_1d.fill(0.0);
        self.storage.h_1d.fill(0.0);
        self.timestep = 0;
    }

    /// Generate GPU extension data for shader compilation.
    ///
    /// Returns None if TF/SF is not active.
    pub fn gpu_data(&self) -> Option<crate::extensions::GpuExtensionData> {
        if !self.is_active() {
            return None;
        }

        // GPU TF/SF metadata structure
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct GpuTfsfMeta {
            // Grid dimensions for plane calculation
            dims: [u32; 3],
            // Propagation axis (0=x, 1=y, 2=z)
            prop_axis: u32,
            // E-field axis
            e_axis: u32,
            // H-field axis
            h_axis: u32,
            // Boundary position along propagation axis
            boundary_pos: u32,
            // 1D auxiliary grid size
            aux_size: u32,
            // Plane size (ny*nz, nx*nz, or nx*ny)
            plane_size: u32,
            // Is propagating in positive direction
            is_positive: u32,
            // Padding for alignment
            _padding: [u32; 2],
        }

        // GPU TF/SF coefficients structure
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct GpuTfsfCoeffs {
            // H-field update: db = -dt / (mu0 * dx)
            db_coeff: f32,
            // E-field update: cb = dt / (eps0 * dx)
            cb_coeff: f32,
            // 1D FDTD coefficients
            db_1d: f32,
            cb_1d: f32,
            // ABC coefficient for right boundary
            abc_coeff: f32,
            // Source parameters
            omega: f32,     // 2 * pi * frequency
            amplitude: f32, // E-field amplitude
            eta_inv: f32,   // 1/eta for H-field calculation
            // Timestep
            dt: f32,
            _padding: [f32; 3],
        }

        let aux_size = self.storage.e_1d.len();
        let plane_size = self.storage.e_src.len();

        let meta = GpuTfsfMeta {
            dims: [
                self.dims.nx as u32,
                self.dims.ny as u32,
                self.dims.nz as u32,
            ],
            prop_axis: self.prop_axis as u32,
            e_axis: self.e_axis as u32,
            h_axis: self.h_axis as u32,
            boundary_pos: self.boundary_pos as u32,
            aux_size: aux_size as u32,
            plane_size: plane_size as u32,
            is_positive: if self.config.direction.is_positive() {
                1
            } else {
                0
            },
            _padding: [0; 2],
        };

        let dx = self.delta[self.prop_axis];
        let omega = 2.0 * PI * self.config.frequency;
        let eta = (crate::constants::MU0 / crate::constants::EPS0).sqrt();

        let coeffs = GpuTfsfCoeffs {
            db_coeff: -(self.dt / (crate::constants::MU0 * dx)) as f32,
            cb_coeff: (self.dt / (crate::constants::EPS0 * dx)) as f32,
            db_1d: -(self.dt / (crate::constants::MU0 * dx)) as f32,
            cb_1d: (self.dt / (crate::constants::EPS0 * dx)) as f32,
            abc_coeff: ((self.phase_velocity * self.dt - dx) / (self.phase_velocity * self.dt + dx))
                as f32,
            omega: omega as f32,
            amplitude: self.config.amplitude as f32,
            eta_inv: (1.0 / eta) as f32,
            dt: self.dt as f32,
            _padding: [0.0; 3],
        };

        // Pack 1D auxiliary field storage (initialized to zeros)
        // Layout: [e_1d[0], e_1d[1], ..., h_1d[0], h_1d[1], ...]
        let mut aux_data: Vec<u8> = Vec::with_capacity(aux_size * 2 * 4);
        for &e in &self.storage.e_1d {
            aux_data.extend_from_slice(bytemuck::bytes_of(&e));
        }
        for &h in &self.storage.h_1d {
            aux_data.extend_from_slice(bytemuck::bytes_of(&h));
        }

        let shader_code = Self::generate_tfsf_shader();

        Some(crate::extensions::GpuExtensionData {
            shader_code,
            buffers: vec![
                crate::extensions::GpuBufferDescriptor {
                    label: "TF/SF Metadata".to_string(),
                    data: bytemuck::bytes_of(&meta).to_vec(),
                    usage: wgpu::BufferUsages::UNIFORM,
                    binding: 17,
                },
                crate::extensions::GpuBufferDescriptor {
                    label: "TF/SF Coefficients".to_string(),
                    data: bytemuck::bytes_of(&coeffs).to_vec(),
                    usage: wgpu::BufferUsages::UNIFORM,
                    binding: 18,
                },
                crate::extensions::GpuBufferDescriptor {
                    label: "TF/SF 1D Auxiliary Fields".to_string(),
                    data: aux_data,
                    usage: wgpu::BufferUsages::STORAGE,
                    binding: 19,
                },
            ],
            bind_group_entries: vec![
                wgpu::BindGroupLayoutEntry {
                    binding: 17,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 18,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 19,
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

    /// Generate WGSL shader code for TF/SF update.
    fn generate_tfsf_shader() -> String {
        r#"
// TF/SF Extension Declarations
struct TfsfMeta {
    dims: vec3<u32>,
    prop_axis: u32,
    e_axis: u32,
    h_axis: u32,
    boundary_pos: u32,
    aux_size: u32,
    plane_size: u32,
    is_positive: u32,
    _padding: vec2<u32>,
}

struct TfsfCoeffs {
    db_coeff: f32,
    cb_coeff: f32,
    db_1d: f32,
    cb_1d: f32,
    abc_coeff: f32,
    omega: f32,
    amplitude: f32,
    eta_inv: f32,
    dt: f32,
    _padding: vec3<f32>,
}

@group(0) @binding(17) var<uniform> tfsf_meta: TfsfMeta;
@group(0) @binding(18) var<uniform> tfsf_coeffs: TfsfCoeffs;
@group(0) @binding(19) var<storage, read_write> tfsf_aux: array<f32>;  // [e_1d..., h_1d...]

// Get E-field from 1D auxiliary grid
fn tfsf_get_e_1d(idx: u32) -> f32 {
    return tfsf_aux[idx];
}

// Get H-field from 1D auxiliary grid
fn tfsf_get_h_1d(idx: u32) -> f32 {
    return tfsf_aux[tfsf_meta.aux_size + idx];
}

// Set E-field in 1D auxiliary grid
fn tfsf_set_e_1d(idx: u32, val: f32) {
    tfsf_aux[idx] = val;
}

// Set H-field in 1D auxiliary grid
fn tfsf_set_h_1d(idx: u32, val: f32) {
    tfsf_aux[tfsf_meta.aux_size + idx] = val;
}

// Calculate incident H-field at given timestep
fn tfsf_incident_h(timestep: u32) -> f32 {
    let t = f32(timestep) * tfsf_coeffs.dt;
    return tfsf_coeffs.amplitude * tfsf_coeffs.eta_inv * sin(tfsf_coeffs.omega * t);
}

// Update 1D auxiliary FDTD grid (called once per timestep by workgroup 0)
fn tfsf_update_1d_grid(timestep: u32) {
    let aux_size = tfsf_meta.aux_size;

    // H-field update
    for (var i = 0u; i < aux_size - 1u; i++) {
        let e_diff = tfsf_get_e_1d(i + 1u) - tfsf_get_e_1d(i);
        let h_old = tfsf_get_h_1d(i);
        tfsf_set_h_1d(i, h_old + tfsf_coeffs.db_1d * e_diff);
    }

    // Inject source at left boundary
    tfsf_set_h_1d(0u, tfsf_incident_h(timestep));

    // E-field update
    for (var i = 1u; i < aux_size; i++) {
        let h_diff = tfsf_get_h_1d(i) - tfsf_get_h_1d(i - 1u);
        let e_old = tfsf_get_e_1d(i);
        tfsf_set_e_1d(i, e_old + tfsf_coeffs.cb_1d * h_diff);
    }

    // ABC at right boundary
    let n = aux_size - 1u;
    tfsf_set_e_1d(n, tfsf_coeffs.abc_coeff * tfsf_get_e_1d(n - 1u));
}

// Check if position is on TF/SF boundary for H-field correction
fn tfsf_is_on_h_boundary(i: u32, j: u32, k: u32) -> bool {
    var pos = array<u32, 3>(i, j, k);
    return pos[tfsf_meta.prop_axis] == tfsf_meta.boundary_pos;
}

// Check if position is on TF/SF boundary for E-field correction
fn tfsf_is_on_e_boundary(i: u32, j: u32, k: u32) -> bool {
    var pos = array<u32, 3>(i, j, k);
    return pos[tfsf_meta.prop_axis] == tfsf_meta.boundary_pos;
}

// Get incident E-field at boundary position
fn tfsf_get_e_inc() -> f32 {
    let pos = tfsf_meta.boundary_pos;
    if (pos < tfsf_meta.aux_size) {
        return tfsf_get_e_1d(pos);
    }
    return 0.0;
}

// Get incident H-field at boundary position
fn tfsf_get_h_inc() -> f32 {
    let pos = tfsf_meta.boundary_pos;
    if (pos < tfsf_meta.aux_size) {
        return tfsf_get_h_1d(pos);
    }
    return 0.0;
}
"#
        .to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tfsf_config() {
        let config = TfsfConfig::new(PropagationDirection::ZPlus, 1e9, 1.0);
        assert_eq!(config.direction, PropagationDirection::ZPlus);
        assert_eq!(config.frequency, 1e9);
    }

    #[test]
    fn test_propagation_direction() {
        assert_eq!(PropagationDirection::XPlus.axis(), 0);
        assert_eq!(PropagationDirection::YMinus.axis(), 1);
        assert_eq!(PropagationDirection::ZPlus.axis(), 2);

        assert!(PropagationDirection::XPlus.is_positive());
        assert!(!PropagationDirection::XMinus.is_positive());
    }

    #[test]
    fn test_tfsf_creation() {
        let dims = Dimensions {
            nx: 50,
            ny: 50,
            nz: 50,
        };
        let delta = [0.001, 0.001, 0.001];
        let dt = 1e-12;

        let config = TfsfConfig::new(PropagationDirection::ZPlus, 1e9, 1.0);
        let tfsf = TfsfBoundary::new(dims, delta, dt, config);

        assert!(tfsf.is_active());
    }

    #[test]
    fn test_incident_field() {
        let dims = Dimensions {
            nx: 20,
            ny: 20,
            nz: 20,
        };
        let delta = [0.001, 0.001, 0.001];
        let dt = 1e-12;

        let config = TfsfConfig::new(PropagationDirection::ZPlus, 1e9, 1.0);
        let tfsf = TfsfBoundary::new(dims, delta, dt, config);

        let e = tfsf.incident_e(0.0);
        assert_eq!(e, 0.0); // sin(0) = 0

        let t = 0.25e-9; // Quarter period for 1 GHz
        let e = tfsf.incident_e(t);
        assert!((e - 1.0).abs() < 0.01); // Should be near maximum
    }

    #[test]
    fn test_tfsf_update() {
        let dims = Dimensions {
            nx: 20,
            ny: 20,
            nz: 20,
        };
        let delta = [0.001, 0.001, 0.001];
        let dt = 1e-12;

        let config = TfsfConfig::new(PropagationDirection::ZPlus, 1e9, 1.0).with_boundary_layer(5);
        let mut tfsf = TfsfBoundary::new(dims, delta, dt, config);

        let mut e_field = VectorField3D::new(dims);
        let mut h_field = VectorField3D::new(dims);

        // Run a few updates
        for _ in 0..10 {
            tfsf.pre_update_h(&mut h_field);
            tfsf.pre_update_e(&mut e_field);
        }

        // Fields should have been modified
    }

    #[test]
    fn test_tfsf_reset() {
        let dims = Dimensions {
            nx: 20,
            ny: 20,
            nz: 20,
        };
        let delta = [0.001, 0.001, 0.001];
        let dt = 1e-12;

        let config = TfsfConfig::new(PropagationDirection::XPlus, 1e9, 1.0);
        let mut tfsf = TfsfBoundary::new(dims, delta, dt, config);

        // Modify state
        tfsf.storage.e_1d[5] = 1.0;
        tfsf.timestep = 100;

        tfsf.reset();
        assert_eq!(tfsf.storage.e_1d[5], 0.0);
        assert_eq!(tfsf.timestep, 0);
    }
}
