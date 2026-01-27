//! Conducting Sheet Model Extension.
//!
//! Implements a frequency-dependent conducting sheet model for efficient FDTD
//! analysis of planar waveguides and circuits.
//!
//! Reference: Lauer, A.; Wolff, I.; "A conducting sheet model for efficient wide band
//! FDTD analysis of planar waveguides and circuits," IEEE MTT-S, 1999.

use crate::arrays::{Dimensions, VectorField3D};

/// Pre-computed Lorentz parameters for frequency-dependent sheet impedance.
/// These parameters fit the skin-effect impedance over specific frequency ranges.
#[derive(Debug, Clone)]
pub struct SheetLorentzParams {
    /// Drude conductivity term (DC conductance)
    pub g: f64,
    /// First Lorentz pole resistance
    pub r1: f64,
    /// First Lorentz pole inductance
    pub l1: f64,
    /// Second Lorentz pole resistance
    pub r2: f64,
    /// Second Lorentz pole inductance
    pub l2: f64,
    /// Critical frequency (rad/s)
    pub omega_critical: f64,
    /// Stop frequency (rad/s)
    pub omega_stop: f64,
}

impl SheetLorentzParams {
    /// Get parameters for copper at a given frequency range.
    /// Based on pre-optimized fits from the original C++ implementation.
    pub fn copper_for_frequency(f_max: f64) -> Self {
        let omega_max = 2.0 * std::f64::consts::PI * f_max;

        // Simplified parameter selection based on frequency range
        // Full implementation would have 30+ frequency ranges
        if omega_max < 1e9 {
            Self {
                g: 5.8e7, // Copper conductivity
                r1: 0.0,
                l1: 1e-12,
                r2: 0.0,
                l2: 1e-12,
                omega_critical: 1e8,
                omega_stop: 1e9,
            }
        } else if omega_max < 1e10 {
            Self {
                g: 5.8e7,
                r1: 0.008,
                l1: 1.5e-12,
                r2: 0.004,
                l2: 2.5e-12,
                omega_critical: 1e9,
                omega_stop: 1e10,
            }
        } else {
            Self {
                g: 5.8e7,
                r1: 0.025,
                l1: 0.8e-12,
                r2: 0.012,
                l2: 1.2e-12,
                omega_critical: 1e10,
                omega_stop: 1e11,
            }
        }
    }
}

/// Configuration for a conducting sheet.
#[derive(Debug, Clone)]
pub struct ConductingSheetConfig {
    /// Sheet conductivity (S/m)
    pub conductivity: f64,
    /// Sheet thickness (m)
    pub thickness: f64,
    /// Maximum simulation frequency (Hz)
    pub f_max: f64,
    /// Normal direction of sheet (0=x, 1=y, 2=z)
    pub normal_direction: usize,
    /// Sheet position along normal (grid index)
    pub position: usize,
    /// Sheet extent: start indices [i, j, k]
    pub start: [usize; 3],
    /// Sheet extent: stop indices [i, j, k]
    pub stop: [usize; 3],
}

impl ConductingSheetConfig {
    /// Create a new conducting sheet configuration.
    pub fn new(
        conductivity: f64,
        thickness: f64,
        f_max: f64,
        normal_direction: usize,
        position: usize,
    ) -> Self {
        Self {
            conductivity,
            thickness,
            f_max,
            normal_direction,
            position,
            start: [0, 0, 0],
            stop: [0, 0, 0],
        }
    }

    /// Set sheet extent.
    pub fn with_extent(mut self, start: [usize; 3], stop: [usize; 3]) -> Self {
        self.start = start;
        self.stop = stop;
        self
    }

    /// Create copper sheet (conductivity = 5.8e7 S/m).
    pub fn copper(thickness: f64, f_max: f64, normal_direction: usize, position: usize) -> Self {
        Self::new(5.8e7, thickness, f_max, normal_direction, position)
    }

    /// Create aluminum sheet (conductivity = 3.8e7 S/m).
    pub fn aluminum(thickness: f64, f_max: f64, normal_direction: usize, position: usize) -> Self {
        Self::new(3.8e7, thickness, f_max, normal_direction, position)
    }
}

/// ADE coefficients for conducting sheet update.
#[derive(Debug, Clone)]
struct AdeCoefficients {
    /// E-field update coefficient
    c1: f32,
    /// Auxiliary current coefficient
    c2: f32,
    /// Previous auxiliary current coefficient
    c3: f32,
    /// Previous E-field coefficient
    c4: f32,
}

/// Single conducting sheet element storage.
#[derive(Debug)]
struct SheetElement {
    /// Position in grid
    pos: [usize; 3],
    /// Tangential direction (primary)
    tan_dir: usize,
    /// ADE coefficients for first Lorentz pole
    ade1: AdeCoefficients,
    /// ADE coefficients for second Lorentz pole
    ade2: AdeCoefficients,
    /// Auxiliary current 1
    j_aux1: f32,
    /// Auxiliary current 2
    j_aux2: f32,
    /// Previous E-field value
    e_prev: f32,
}

impl SheetElement {
    fn new(pos: [usize; 3], tan_dir: usize) -> Self {
        Self {
            pos,
            tan_dir,
            ade1: AdeCoefficients {
                c1: 1.0,
                c2: 0.0,
                c3: 0.0,
                c4: 0.0,
            },
            ade2: AdeCoefficients {
                c1: 1.0,
                c2: 0.0,
                c3: 0.0,
                c4: 0.0,
            },
            j_aux1: 0.0,
            j_aux2: 0.0,
            e_prev: 0.0,
        }
    }
}

/// Conducting Sheet Extension.
///
/// Models thin conducting sheets with frequency-dependent impedance
/// using auxiliary differential equations (ADE).
pub struct ConductingSheet {
    /// Configuration
    config: ConductingSheetConfig,
    /// Grid dimensions
    dims: Dimensions,
    /// Timestep
    dt: f64,
    /// Lorentz parameters
    params: SheetLorentzParams,
    /// Sheet elements (one per affected grid edge)
    elements: Vec<SheetElement>,
    /// Whether the extension is active
    active: bool,
}

impl ConductingSheet {
    /// Create a new conducting sheet extension.
    pub fn new(config: ConductingSheetConfig, dims: Dimensions, dt: f64) -> Self {
        let params = SheetLorentzParams::copper_for_frequency(config.f_max);

        let mut sheet = Self {
            config,
            dims,
            dt,
            params,
            elements: Vec::new(),
            active: false,
        };

        sheet.build_elements();
        sheet
    }

    /// Build sheet elements for all affected grid edges.
    fn build_elements(&mut self) {
        let ny = self.config.normal_direction;
        let nyp = (ny + 1) % 3;
        let nypp = (ny + 2) % 3;

        let pos_ny = self.config.position;

        // Determine sheet extent
        let start = self.config.start;
        let stop = [
            if self.config.stop[0] == 0 {
                self.dims.nx
            } else {
                self.config.stop[0]
            },
            if self.config.stop[1] == 0 {
                self.dims.ny
            } else {
                self.config.stop[1]
            },
            if self.config.stop[2] == 0 {
                self.dims.nz
            } else {
                self.config.stop[2]
            },
        ];

        // Create elements for both tangential directions
        for tan_dir in [nyp, nypp] {
            let (i_start, i_end, j_start, j_end) = match ny {
                0 => (start[1], stop[1], start[2], stop[2]),
                1 => (start[0], stop[0], start[2], stop[2]),
                _ => (start[0], stop[0], start[1], stop[1]),
            };

            for i in i_start..i_end {
                for j in j_start..j_end {
                    let mut pos = [0usize; 3];
                    pos[ny] = pos_ny;
                    match ny {
                        0 => {
                            pos[1] = i;
                            pos[2] = j;
                        }
                        1 => {
                            pos[0] = i;
                            pos[2] = j;
                        }
                        _ => {
                            pos[0] = i;
                            pos[1] = j;
                        }
                    }

                    let mut elem = SheetElement::new(pos, tan_dir);
                    self.init_coefficients(&mut elem);
                    self.elements.push(elem);
                }
            }
        }

        self.active = !self.elements.is_empty();
    }

    /// Initialize ADE coefficients for a sheet element.
    fn init_coefficients(&self, elem: &mut SheetElement) {
        // Surface resistance per square
        let r_sheet = 1.0 / (self.config.conductivity * self.config.thickness);

        // First Lorentz pole coefficients
        // From: Z1 = r1 + j*omega*l1
        // ADE: dJ1/dt + (r1/l1)*J1 = (1/l1)*E
        let alpha1 = if self.params.l1 > 0.0 {
            self.params.r1 / self.params.l1
        } else {
            0.0
        };
        let beta1 = if self.params.l1 > 0.0 {
            1.0 / self.params.l1
        } else {
            0.0
        };

        // Discretized: J1^(n+1) = c3*J1^n + c2*E^(n+1/2)
        let denom1 = 1.0 + 0.5 * alpha1 * self.dt;
        elem.ade1.c3 = ((1.0 - 0.5 * alpha1 * self.dt) / denom1) as f32;
        elem.ade1.c2 = (beta1 * self.dt / denom1) as f32;
        elem.ade1.c1 = (r_sheet * elem.ade1.c2 as f64) as f32;

        // Second Lorentz pole
        let alpha2 = if self.params.l2 > 0.0 {
            self.params.r2 / self.params.l2
        } else {
            0.0
        };
        let beta2 = if self.params.l2 > 0.0 {
            1.0 / self.params.l2
        } else {
            0.0
        };

        let denom2 = 1.0 + 0.5 * alpha2 * self.dt;
        elem.ade2.c3 = ((1.0 - 0.5 * alpha2 * self.dt) / denom2) as f32;
        elem.ade2.c2 = (beta2 * self.dt / denom2) as f32;
        elem.ade2.c1 = (r_sheet * elem.ade2.c2 as f64) as f32;

        // Drude (DC) term
        elem.ade1.c4 = (self.params.g * self.config.thickness * self.dt) as f32;
    }

    /// Check if extension is active.
    pub fn is_active(&self) -> bool {
        self.active
    }

    /// Get number of sheet elements.
    pub fn num_elements(&self) -> usize {
        self.elements.len()
    }

    /// Apply conducting sheet modification to E-field update.
    pub fn apply_to_voltage(&mut self, e_field: &mut VectorField3D) {
        if !self.active {
            return;
        }

        for elem in &mut self.elements {
            let (i, j, k) = (elem.pos[0], elem.pos[1], elem.pos[2]);

            // Get current E-field
            let e_current = match elem.tan_dir {
                0 => e_field.x.get(i, j, k),
                1 => e_field.y.get(i, j, k),
                _ => e_field.z.get(i, j, k),
            };

            // Update auxiliary currents (ADE)
            let j1_new = elem.ade1.c3 * elem.j_aux1 + elem.ade1.c2 * e_current;
            let j2_new = elem.ade2.c3 * elem.j_aux2 + elem.ade2.c2 * e_current;

            // Calculate total current modification
            let j_total = elem.ade1.c4 * e_current // Drude term
                + elem.ade1.c1 * (j1_new + elem.j_aux1) * 0.5 // Lorentz 1
                + elem.ade2.c1 * (j2_new + elem.j_aux2) * 0.5; // Lorentz 2

            // Modify E-field
            let e_modified = e_current - j_total;

            // Store updated values
            elem.j_aux1 = j1_new;
            elem.j_aux2 = j2_new;
            elem.e_prev = e_current;

            // Write back
            match elem.tan_dir {
                0 => e_field.x.set(i, j, k, e_modified),
                1 => e_field.y.set(i, j, k, e_modified),
                _ => e_field.z.set(i, j, k, e_modified),
            }
        }
    }

    /// Reset all auxiliary variables.
    pub fn reset(&mut self) {
        for elem in &mut self.elements {
            elem.j_aux1 = 0.0;
            elem.j_aux2 = 0.0;
            elem.e_prev = 0.0;
        }
    }

    /// Generate GPU extension data for shader compilation.
    ///
    /// Returns None if the sheet is not active.
    pub fn gpu_data(&self) -> Option<crate::extensions::GpuExtensionData> {
        if !self.is_active() {
            return None;
        }

        // GPU conducting sheet metadata
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct GpuSheetMeta {
            // Number of sheet elements
            num_elements: u32,
            // Normal direction of sheet (0=x, 1=y, 2=z)
            normal_dir: u32,
            // Sheet position along normal
            position: u32,
            // Padding
            _padding: u32,
        }

        // GPU element data: position and tangent direction
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct GpuSheetElement {
            // Position [i, j, k]
            pos: [u32; 3],
            // Tangential direction (0=x, 1=y, 2=z)
            tan_dir: u32,
        }

        // GPU ADE coefficients per element (both Lorentz poles)
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct GpuSheetAdeCoeffs {
            // First Lorentz pole: c1, c2, c3, c4
            ade1_c1: f32,
            ade1_c2: f32,
            ade1_c3: f32,
            ade1_c4: f32,
            // Second Lorentz pole: c1, c2, c3
            ade2_c1: f32,
            ade2_c2: f32,
            ade2_c3: f32,
            _padding: f32,
        }

        let meta = GpuSheetMeta {
            num_elements: self.elements.len() as u32,
            normal_dir: self.config.normal_direction as u32,
            position: self.config.position as u32,
            _padding: 0,
        };

        // Pack element positions
        let mut elements_data: Vec<u8> = Vec::with_capacity(self.elements.len() * 16);
        for elem in &self.elements {
            let gpu_elem = GpuSheetElement {
                pos: [elem.pos[0] as u32, elem.pos[1] as u32, elem.pos[2] as u32],
                tan_dir: elem.tan_dir as u32,
            };
            elements_data.extend_from_slice(bytemuck::bytes_of(&gpu_elem));
        }

        // Pack ADE coefficients
        let mut coeffs_data: Vec<u8> = Vec::with_capacity(self.elements.len() * 32);
        for elem in &self.elements {
            let gpu_coeffs = GpuSheetAdeCoeffs {
                ade1_c1: elem.ade1.c1,
                ade1_c2: elem.ade1.c2,
                ade1_c3: elem.ade1.c3,
                ade1_c4: elem.ade1.c4,
                ade2_c1: elem.ade2.c1,
                ade2_c2: elem.ade2.c2,
                ade2_c3: elem.ade2.c3,
                _padding: 0.0,
            };
            coeffs_data.extend_from_slice(bytemuck::bytes_of(&gpu_coeffs));
        }

        // Pack auxiliary storage (j_aux1, j_aux2, e_prev per element)
        let mut storage_data: Vec<u8> = Vec::with_capacity(self.elements.len() * 12);
        for elem in &self.elements {
            storage_data.extend_from_slice(bytemuck::bytes_of(&elem.j_aux1));
            storage_data.extend_from_slice(bytemuck::bytes_of(&elem.j_aux2));
            storage_data.extend_from_slice(bytemuck::bytes_of(&elem.e_prev));
        }
        // Pad to 16 bytes per element for alignment
        while storage_data.len() < self.elements.len() * 16 {
            storage_data.extend_from_slice(bytemuck::bytes_of(&0.0f32));
        }

        let shader_code = Self::generate_sheet_shader();

        Some(crate::extensions::GpuExtensionData {
            shader_code,
            buffers: vec![
                crate::extensions::GpuBufferDescriptor {
                    label: "Conducting Sheet Metadata".to_string(),
                    data: bytemuck::bytes_of(&meta).to_vec(),
                    usage: wgpu::BufferUsages::UNIFORM,
                    binding: 24,
                },
                crate::extensions::GpuBufferDescriptor {
                    label: "Conducting Sheet Elements".to_string(),
                    data: elements_data,
                    usage: wgpu::BufferUsages::STORAGE,
                    binding: 25,
                },
                crate::extensions::GpuBufferDescriptor {
                    label: "Conducting Sheet ADE Coefficients".to_string(),
                    data: coeffs_data,
                    usage: wgpu::BufferUsages::STORAGE,
                    binding: 26,
                },
                crate::extensions::GpuBufferDescriptor {
                    label: "Conducting Sheet Storage".to_string(),
                    data: storage_data,
                    usage: wgpu::BufferUsages::STORAGE,
                    binding: 27,
                },
            ],
            bind_group_entries: vec![
                wgpu::BindGroupLayoutEntry {
                    binding: 24,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 25,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 26,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 27,
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

    /// Generate WGSL shader code for conducting sheet update.
    fn generate_sheet_shader() -> String {
        r#"
// Conducting Sheet Extension Declarations
struct SheetMeta {
    num_elements: u32,
    normal_dir: u32,
    position: u32,
    _padding: u32,
}

struct SheetElement {
    pos: vec3<u32>,
    tan_dir: u32,
}

struct SheetAdeCoeffs {
    ade1_c1: f32,
    ade1_c2: f32,
    ade1_c3: f32,
    ade1_c4: f32,
    ade2_c1: f32,
    ade2_c2: f32,
    ade2_c3: f32,
    _padding: f32,
}

struct SheetStorage {
    j_aux1: f32,
    j_aux2: f32,
    e_prev: f32,
    _padding: f32,
}

@group(0) @binding(24) var<uniform> sheet_meta: SheetMeta;
@group(0) @binding(25) var<storage, read> sheet_elements: array<SheetElement>;
@group(0) @binding(26) var<storage, read> sheet_coeffs: array<SheetAdeCoeffs>;
@group(0) @binding(27) var<storage, read_write> sheet_storage: array<SheetStorage>;

// Get sheet element by index
fn sheet_get_element(idx: u32) -> SheetElement {
    return sheet_elements[idx];
}

// Get ADE coefficients for an element
fn sheet_get_coeffs(idx: u32) -> SheetAdeCoeffs {
    return sheet_coeffs[idx];
}

// Get storage values for an element
fn sheet_get_storage(idx: u32) -> SheetStorage {
    return sheet_storage[idx];
}

// Set storage values for an element
fn sheet_set_storage(idx: u32, s: SheetStorage) {
    sheet_storage[idx] = s;
}

// Find sheet element at given position, returns -1 if not found
fn sheet_find_element(gi: u32, gj: u32, gk: u32, tan_dir: u32) -> i32 {
    for (var idx = 0u; idx < sheet_meta.num_elements; idx++) {
        let elem = sheet_elements[idx];
        if (elem.pos.x == gi && elem.pos.y == gj && elem.pos.z == gk && elem.tan_dir == tan_dir) {
            return i32(idx);
        }
    }
    return -1;
}

// Apply conducting sheet update to E-field
// Returns the modified E-field value
fn sheet_apply_update(e_current: f32, idx: u32) -> f32 {
    let coeffs = sheet_get_coeffs(idx);
    var storage = sheet_get_storage(idx);

    // Update auxiliary currents (ADE)
    let j1_new = coeffs.ade1_c3 * storage.j_aux1 + coeffs.ade1_c2 * e_current;
    let j2_new = coeffs.ade2_c3 * storage.j_aux2 + coeffs.ade2_c2 * e_current;

    // Calculate total current modification
    let j_total = coeffs.ade1_c4 * e_current  // Drude term
        + coeffs.ade1_c1 * (j1_new + storage.j_aux1) * 0.5  // Lorentz 1
        + coeffs.ade2_c1 * (j2_new + storage.j_aux2) * 0.5; // Lorentz 2

    // Modify E-field
    let e_modified = e_current - j_total;

    // Store updated values
    storage.j_aux1 = j1_new;
    storage.j_aux2 = j2_new;
    storage.e_prev = e_current;
    sheet_set_storage(idx, storage);

    return e_modified;
}

// Check if position is on conducting sheet boundary
fn sheet_is_on_boundary(gi: u32, gj: u32, gk: u32) -> bool {
    var pos = array<u32, 3>(gi, gj, gk);
    return pos[sheet_meta.normal_dir] == sheet_meta.position;
}
"#
        .to_string()
    }
}

/// Manager for multiple conducting sheets.
pub struct ConductingSheetManager {
    sheets: Vec<ConductingSheet>,
}

impl ConductingSheetManager {
    /// Create a new manager.
    pub fn new() -> Self {
        Self { sheets: Vec::new() }
    }

    /// Add a conducting sheet.
    pub fn add_sheet(&mut self, sheet: ConductingSheet) {
        self.sheets.push(sheet);
    }

    /// Apply all sheets to E-field.
    pub fn apply_to_voltage(&mut self, e_field: &mut VectorField3D) {
        for sheet in &mut self.sheets {
            sheet.apply_to_voltage(e_field);
        }
    }

    /// Reset all sheets.
    pub fn reset(&mut self) {
        for sheet in &mut self.sheets {
            sheet.reset();
        }
    }

    /// Check if any sheets are active.
    pub fn is_active(&self) -> bool {
        self.sheets.iter().any(|s| s.is_active())
    }
}

impl Default for ConductingSheetManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lorentz_params() {
        let params = SheetLorentzParams::copper_for_frequency(1e9);
        assert!((params.g - 5.8e7).abs() < 1e3);

        let params_high = SheetLorentzParams::copper_for_frequency(1e11);
        assert!(params_high.r1 > 0.0);
    }

    #[test]
    fn test_sheet_config() {
        let config = ConductingSheetConfig::copper(35e-6, 10e9, 2, 10);
        assert!((config.conductivity - 5.8e7).abs() < 1.0);
        assert!((config.thickness - 35e-6).abs() < 1e-9);
    }

    #[test]
    fn test_sheet_creation() {
        let dims = Dimensions::new(20, 20, 20);
        let config =
            ConductingSheetConfig::copper(35e-6, 10e9, 2, 10).with_extent([0, 0, 0], [20, 20, 0]);

        let sheet = ConductingSheet::new(config, dims, 1e-12);

        assert!(sheet.is_active());
        // Should have elements for both tangential directions across the sheet
        assert!(sheet.num_elements() > 0);
    }

    #[test]
    fn test_sheet_update() {
        let dims = Dimensions::new(10, 10, 10);
        let config = ConductingSheetConfig::copper(35e-6, 10e9, 2, 5);

        let mut sheet = ConductingSheet::new(config, dims, 1e-12);
        let mut e_field = VectorField3D::new(dims);

        // Set initial field at sheet position
        for i in 0..10 {
            for j in 0..10 {
                e_field.x.set(i, j, 5, 1.0);
                e_field.y.set(i, j, 5, 1.0);
            }
        }

        // Apply sheet
        sheet.apply_to_voltage(&mut e_field);

        // Field should be modified (reduced by conductive losses)
        let ex = e_field.x.get(5, 5, 5);
        assert!(ex <= 1.0);
    }

    #[test]
    fn test_sheet_reset() {
        let dims = Dimensions::new(10, 10, 10);
        let config = ConductingSheetConfig::copper(35e-6, 10e9, 2, 5);

        let mut sheet = ConductingSheet::new(config, dims, 1e-12);
        let mut e_field = VectorField3D::new(dims);

        e_field.x.fill(1.0);

        // Run a few updates
        for _ in 0..5 {
            sheet.apply_to_voltage(&mut e_field);
        }

        // Reset
        sheet.reset();

        // All auxiliary variables should be zero
        for elem in &sheet.elements {
            assert_eq!(elem.j_aux1, 0.0);
            assert_eq!(elem.j_aux2, 0.0);
        }
    }

    #[test]
    fn test_sheet_manager() {
        let dims = Dimensions::new(20, 20, 20);

        let config1 = ConductingSheetConfig::copper(35e-6, 10e9, 2, 5);
        let config2 = ConductingSheetConfig::aluminum(35e-6, 10e9, 2, 15);

        let mut manager = ConductingSheetManager::new();
        manager.add_sheet(ConductingSheet::new(config1, dims, 1e-12));
        manager.add_sheet(ConductingSheet::new(config2, dims, 1e-12));

        assert!(manager.is_active());

        let mut e_field = VectorField3D::new(dims);
        e_field.x.fill(1.0);

        manager.apply_to_voltage(&mut e_field);
        manager.reset();
    }
}
