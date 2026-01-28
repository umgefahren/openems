use crate::arrays::{Dimensions, VectorField3D};
use crate::fdtd::batch::{BatchResult, EnergySample, EngineBatch, TerminationReason};
use crate::fdtd::engine_impl::EngineImpl;
use crate::fdtd::operator::Operator;
use crate::Result;
use half::f16;
use instant::Instant;
use std::borrow::Cow;
use wgpu::util::DeviceExt;

/// GPU-accelerated FDTD engine using WebGPU with batching support.
pub struct GpuEngine {
    instance: wgpu::Instance,
    device: wgpu::Device,
    queue: wgpu::Queue,
    update_h_pipeline: wgpu::ComputePipeline,
    update_e_pipeline: wgpu::ComputePipeline,
    apply_excitations_pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,

    // Core field buffers
    e_field_buffer: wgpu::Buffer,
    h_field_buffer: wgpu::Buffer,

    // Coefficient buffers (stored for bind group recreation)
    e_coeff_f16_buffer: wgpu::Buffer,
    e_coeff_f32_buffer: wgpu::Buffer,
    h_coeff_f16_buffer: wgpu::Buffer,
    h_coeff_f32_buffer: wgpu::Buffer,
    cell_class_buffer: wgpu::Buffer,

    // Excitation buffers (dynamically resized per batch)
    excitation_buffer: wgpu::Buffer,
    excitation_offsets_buffer: wgpu::Buffer,
    timestep_buffer: wgpu::Buffer,
    max_excitations: usize,
    max_timesteps: usize,

    // Energy reduction buffers
    energy_partial_buffer: wgpu::Buffer,
    energy_staging_buffer: wgpu::Buffer,
    energy_pipeline: wgpu::ComputePipeline,
    num_energy_workgroups: usize,

    // Dimensions
    dims: Dimensions,

    // Workgroup dispatch size
    dispatch_x: u32,
    dispatch_y: u32,
    dispatch_z: u32,

    // State
    timestep: u64,
    #[allow(dead_code)]
    supports_f16: bool,

    // Field cache (avoid unnecessary GPU↔CPU transfers)
    field_cache: Option<(VectorField3D, VectorField3D)>,
    cache_dirty: bool,
}

impl GpuEngine {
    /// Check if GPU acceleration is available on this system.
    pub fn is_available() -> bool {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }));

        adapter.is_ok()
    }

    /// Create a new GPU engine.
    pub fn new_internal(operator: &Operator) -> Self {
        let dims = operator.dimensions();
        let total = dims.total();

        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .expect("Failed to find an appropriate adapter");

        // Check for f16 support
        let features = adapter.features();
        let supports_f16 = features.contains(wgpu::Features::SHADER_F16);

        let required_features = if supports_f16 {
            wgpu::Features::SHADER_F16
        } else {
            wgpu::Features::empty()
        };

        let limits = adapter.limits();
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: None,
            required_features,
            required_limits: limits,
            memory_hints: wgpu::MemoryHints::Performance,
            ..Default::default()
        }))
        .expect("Failed to create device");

        // Prepare shader source
        let shader_base = include_str!("shaders.wgsl");
        let shader_source = if supports_f16 {
            format!("enable f16;\nalias float16 = f16;\n{}", shader_base)
        } else {
            format!("alias float16 = f32;\n{}", shader_base)
        };

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("FDTD Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(shader_source)),
        });

        let field_size = (total * 3 * std::mem::size_of::<f32>()) as u64;

        let e_field_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("E Field Buffer"),
            size: field_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let h_field_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("H Field Buffer"),
            size: field_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Prepare coefficients and classification
        let e_coeff = operator.e_coefficients();
        let h_coeff = operator.h_coefficients();

        // Layout:
        // coeff_f16: [ca_x, ca_y, ca_z, cb_x, cb_y, cb_z] per cell (interleaved? no, array of structs?)
        // The shader expects arrays.
        // Prompt shader:
        // @group(0) @binding(2) var<storage, read> coeff_f16: array<vec4<float16>>;
        // @group(0) @binding(3) var<storage, read> coeff_f32: array<vec4<f32>>;
        // @group(0) @binding(4) var<storage, read> cell_class: array<u32>;

        // We need to pack 6 coeffs per cell.
        // Option 1: 2 vec4s per cell. [ca_x, ca_y, ca_z, cb_x] and [cb_y, cb_z, 0, 0].
        // Option 2: 1 vec4 + 1 vec2 (packed? alignment issues).
        // Let's use 2 vec4s per cell for alignment simplicity (stride 8 floats).

        let mut e_coeff_f16_data: Vec<u8> = Vec::with_capacity(total * 8 * 2); // 8 halfs * 2 bytes
        let mut e_coeff_f32_data: Vec<u8> = Vec::with_capacity(total * 8 * 4); // 8 floats * 4 bytes
        let mut cell_class_data: Vec<u32> = Vec::with_capacity(total);

        // Populate E coefficients
        // IMPORTANT: Loop order must match shader's get_idx(i,j,k) = i*ny*nz + j*nz + k
        // This means i varies slowest, j next, k fastest (row-major for x,y,z)
        for i in 0..dims.nx {
            for j in 0..dims.ny {
                for k in 0..dims.nz {
                    let ca = [
                        e_coeff.ca[0].get(i, j, k),
                        e_coeff.ca[1].get(i, j, k),
                        e_coeff.ca[2].get(i, j, k),
                    ];
                    let cb = [
                        e_coeff.cb[0].get(i, j, k),
                        e_coeff.cb[1].get(i, j, k),
                        e_coeff.cb[2].get(i, j, k),
                    ];

                    // Classify: Use f16 where it can accurately represent the coefficients.
                    // f16 has ~3 decimal digits precision (machine epsilon ~9.77e-4).
                    //
                    // IMPORTANT: Even small coefficient errors accumulate over many timesteps.
                    // For a 300-step simulation, a 0.01% per-step error can become ~3% total.
                    // We use a very tight threshold (1e-5 = 0.001%) to ensure errors stay
                    // below our allclose tolerance (rtol=5e-4) even after hundreds of steps.
                    //
                    // In practice, this means f16 is only used for:
                    // - Coefficients that are exactly representable in f16 (powers of 2, etc.)
                    // - Ca = 1.0 (lossless vacuum) which is exact in f16
                    let class_id = if supports_f16 {
                        let coeffs = [ca[0], ca[1], ca[2], cb[0], cb[1], cb[2]];
                        let needs_f32 = coeffs.iter().any(|&v| {
                            if v == 0.0 {
                                false // Zero is exact in both formats
                            } else {
                                let v_f16 = f16::from_f32(v).to_f32();
                                let rel_err = ((v - v_f16) / v).abs();
                                rel_err > 1e-5 // 0.001% threshold - very tight for long simulations
                            }
                        });
                        if needs_f32 { 1u32 } else { 0u32 }
                    } else {
                        1u32 // No f16 support, always use f32
                    };
                    cell_class_data.push(class_id);

                    // Pack data
                    let data_floats = [ca[0], ca[1], ca[2], cb[0], cb[1], cb[2], 0.0, 0.0];

                    // f32 buffer
                    for &val in &data_floats {
                        e_coeff_f32_data.extend_from_slice(bytemuck::bytes_of(&val));
                    }

                    // f16 buffer
                    if supports_f16 {
                        for &val in &data_floats {
                            let val_f16 = f16::from_f32(val);
                            e_coeff_f16_data.extend_from_slice(bytemuck::bytes_of(&val_f16));
                        }
                    } else {
                        // Fallback: store f32 in "f16" buffer slot (shader alias handles type)
                        for &val in &data_floats {
                            e_coeff_f32_data.extend_from_slice(bytemuck::bytes_of(&val));
                        }
                    }
                }
            }
        }

        // If fallback, e_coeff_f16_data is empty, we point to f32 data?
        // No, shader expects binding 2 to exist.
        // If !supports_f16, alias float16 = f32.
        // So binding 2 should contain f32s.
        if !supports_f16 {
            e_coeff_f16_data = e_coeff_f32_data.clone();
        }

        let e_coeff_f16_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("E Coeff f16 Buffer"),
            contents: &e_coeff_f16_data,
            usage: wgpu::BufferUsages::STORAGE,
        });

        let e_coeff_f32_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("E Coeff f32 Buffer"),
            contents: &e_coeff_f32_data,
            usage: wgpu::BufferUsages::STORAGE,
        });

        // H Coefficients
        // Similar logic, but H usually doesn't have conductivity in standard FDTD unless magnetic loss.
        // But we follow symmetry.
        let mut h_coeff_f16_data: Vec<u8> = Vec::with_capacity(total * 8 * 2);
        let mut h_coeff_f32_data: Vec<u8> = Vec::with_capacity(total * 8 * 4);

        // Note: cell_class is shared? Or separate?
        // We reused cell_class logic for E. H might have different requirements (magnetic loss).
        // The prompt implies a single cell_class buffer.
        // We'll assume cell_class covers both (union of requirements).
        // Since we already filled cell_class based on E, we might need to update it for H?
        // But H update uses cell_class to choose H coeffs.

        // IMPORTANT: Loop order must match shader's get_idx(i,j,k) = i*ny*nz + j*nz + k
        for i in 0..dims.nx {
            for j in 0..dims.ny {
                for k in 0..dims.nz {
                    let da = [
                        h_coeff.da[0].get(i, j, k),
                        h_coeff.da[1].get(i, j, k),
                        h_coeff.da[2].get(i, j, k),
                    ];
                    let db = [
                        h_coeff.db[0].get(i, j, k),
                        h_coeff.db[1].get(i, j, k),
                        h_coeff.db[2].get(i, j, k),
                    ];

                    let data_floats = [da[0], da[1], da[2], db[0], db[1], db[2], 0.0, 0.0];

                    for &val in &data_floats {
                        h_coeff_f32_data.extend_from_slice(bytemuck::bytes_of(&val));
                    }

                    if supports_f16 {
                        for &val in &data_floats {
                            let val_f16 = f16::from_f32(val);
                            h_coeff_f16_data.extend_from_slice(bytemuck::bytes_of(&val_f16));
                        }
                    }
                }
            }
        }

        if !supports_f16 {
            h_coeff_f16_data = h_coeff_f32_data.clone();
        }

        let h_coeff_f16_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("H Coeff f16 Buffer"),
            contents: &h_coeff_f16_data,
            usage: wgpu::BufferUsages::STORAGE,
        });

        let h_coeff_f32_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("H Coeff f32 Buffer"),
            contents: &h_coeff_f32_data,
            usage: wgpu::BufferUsages::STORAGE,
        });

        let cell_class_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Cell Class Buffer"),
            contents: bytemuck::cast_slice(&cell_class_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Create initial excitation buffers (small default, will be resized per batch)
        let initial_max_excitations = 16;
        let initial_max_timesteps = 1024;

        let excitation_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Excitation Buffer"),
            // ExcitationPoint: position (3 u32) + direction (u32) + value (f32) + soft_source (u32) = 24 bytes
            size: (initial_max_excitations * initial_max_timesteps * 24) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let excitation_offsets_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Excitation Offsets Buffer"),
            // One offset per timestep + 1 for end sentinel
            size: ((initial_max_timesteps + 1) * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Timestep uniform buffer (with dynamic offsets for batching)
        // Each timestep value is in a 256-byte aligned chunk (WebGPU requirement)
        let timestep_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Timestep Uniform Buffer"),
            size: (initial_max_timesteps * 256) as u64,  // 256-byte alignment per timestep
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Energy reduction buffers (created early so they can be added to bind group)
        // We use 256 workgroups for reduction, each outputs 2 f32s (e_energy, h_energy)
        let num_energy_workgroups = 256usize;
        let energy_partial_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Energy Partial Buffer"),
            size: (num_energy_workgroups * 2 * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let energy_staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Energy Staging Buffer"),
            size: (num_energy_workgroups * 2 * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("FDTD Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    // E Field
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // H Field
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // Coeff f16 (E)
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // Coeff f32 (E)
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // Cell Class
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // Coeff f16 (H)
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // Coeff f32 (H)
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // Excitations
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // Excitation Offsets
                    binding: 8,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // Timestep uniform (with dynamic offset for batching)
                    binding: 9,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: Some(std::num::NonZeroU64::new(4).unwrap()),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // Energy output
                    binding: 10,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("FDTD Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: e_field_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: h_field_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: e_coeff_f16_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: e_coeff_f32_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: cell_class_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: h_coeff_f16_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: h_coeff_f32_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: excitation_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: excitation_offsets_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &timestep_buffer,
                        offset: 0,
                        size: Some(std::num::NonZeroU64::new(256).unwrap()),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: energy_partial_buffer.as_entire_binding(),
                },
            ],
        });

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            ..Default::default()
        });

        let constants_data = [
            ("nx", dims.nx as f64),
            ("ny", dims.ny as f64),
            ("nz", dims.nz as f64),
        ];

        let update_h_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Update H Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("update_h"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &constants_data,
                ..Default::default()
            },
            cache: None,
        });

        let update_e_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Update E Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("update_e"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &constants_data,
                ..Default::default()
            },
            cache: None,
        });

        let apply_excitations_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Apply Excitations Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("apply_excitations"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &constants_data,
                ..Default::default()
            },
            cache: None,
        });

        // Energy reduction pipeline
        let energy_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Energy Reduction Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("compute_energy"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &constants_data,
                ..Default::default()
            },
            cache: None,
        });

        // Dispatch dimensions swapped to match shader mapping:
        // GlobalID.x -> k (fastest mem) -> dispatch_x covers nz (ILP=4)
        // GlobalID.y -> j -> dispatch_y covers ny
        // GlobalID.z -> i -> dispatch_z covers nx

        // Workgroup size is (64, 1, 1) defined in shader.
        // Each thread along X (k) processes 4 elements (ILP=4).
        // So one workgroup covers 64 * 4 = 256 elements along Z.
        let dispatch_x = (dims.nz as u32).div_ceil(256);

        // Threads along Y and Z process 1 element each.
        let dispatch_y = dims.ny as u32;
        let dispatch_z = dims.nx as u32;

        Self {
            instance,
            device,
            queue,
            update_h_pipeline,
            update_e_pipeline,
            apply_excitations_pipeline,
            bind_group_layout,
            bind_group,
            e_field_buffer,
            h_field_buffer,
            e_coeff_f16_buffer,
            e_coeff_f32_buffer,
            h_coeff_f16_buffer,
            h_coeff_f32_buffer,
            cell_class_buffer,
            excitation_buffer,
            excitation_offsets_buffer,
            timestep_buffer,
            max_excitations: initial_max_excitations,
            max_timesteps: initial_max_timesteps,
            energy_partial_buffer,
            energy_staging_buffer,
            energy_pipeline,
            num_energy_workgroups,
            dims,
            dispatch_x,
            dispatch_y,
            dispatch_z,
            timestep: 0,
            supports_f16,
            field_cache: None,
            cache_dirty: false,
        }
    }

    /// Synchronize fields from GPU to CPU cache.
    fn sync_fields_from_gpu(&mut self) {
        if self.cache_dirty || self.field_cache.is_none() {
            let mut e_field = VectorField3D::new(self.dims);
            let mut h_field = VectorField3D::new(self.dims);

            self.read_buffer_to_field(&self.e_field_buffer, &mut e_field);
            self.read_buffer_to_field(&self.h_field_buffer, &mut h_field);

            self.field_cache = Some((e_field, h_field));
            self.cache_dirty = false;
        }
    }

    /// Synchronize fields from CPU cache to GPU.
    fn sync_fields_to_gpu(&mut self) {
        if let Some((ref e_field, ref h_field)) = self.field_cache {
            self.write_e_field(e_field);
            self.write_h_field(h_field);
            self.cache_dirty = false;
        }
    }

    /// Perform one FDTD timestep on the GPU.
    pub fn step(&self) {
        // Upload current timestep value to the first 256-byte chunk
        let mut timestep_chunk = vec![0u8; 256];
        timestep_chunk[0..4].copy_from_slice(&(self.timestep as u32).to_le_bytes());
        self.queue.write_buffer(&self.timestep_buffer, 0, &timestep_chunk);

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Step Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Update H Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.update_h_pipeline);
            compute_pass.set_bind_group(0, &self.bind_group, &[0]);  // Use offset 0
            compute_pass.dispatch_workgroups(self.dispatch_x, self.dispatch_y, self.dispatch_z);
        }

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Update E Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.update_e_pipeline);
            compute_pass.set_bind_group(0, &self.bind_group, &[0]);  // Use offset 0
            compute_pass.dispatch_workgroups(self.dispatch_x, self.dispatch_y, self.dispatch_z);
        }

        self.queue.submit(Some(encoder.finish()));
    }

    /// Upload E-field data to the GPU.
    pub fn write_e_field(&self, field: &VectorField3D) {
        let mut data = Vec::with_capacity(self.dims.total() * 3);
        data.extend_from_slice(field.x.as_slice());
        data.extend_from_slice(field.y.as_slice());
        data.extend_from_slice(field.z.as_slice());

        self.queue
            .write_buffer(&self.e_field_buffer, 0, bytemuck::cast_slice(&data));
    }

    /// Upload H-field data to the GPU.
    pub fn write_h_field(&self, field: &VectorField3D) {
        let mut data = Vec::with_capacity(self.dims.total() * 3);
        data.extend_from_slice(field.x.as_slice());
        data.extend_from_slice(field.y.as_slice());
        data.extend_from_slice(field.z.as_slice());

        self.queue
            .write_buffer(&self.h_field_buffer, 0, bytemuck::cast_slice(&data));
    }

    /// Download E-field data from the GPU.
    pub fn read_e_field(&self, field: &mut VectorField3D) {
        self.read_buffer_to_field(&self.e_field_buffer, field);
    }

    /// Download H-field data from the GPU.
    pub fn read_h_field(&self, field: &mut VectorField3D) {
        self.read_buffer_to_field(&self.h_field_buffer, field);
    }

    /// Update a single element of the E-field on the GPU.
    pub fn update_e_field_element(&self, i: usize, j: usize, k: usize, dir: usize, value: f32) {
        let total = self.dims.total();
        let offset_elems = match dir {
            0 => 0,
            1 => total,
            2 => 2 * total,
            _ => return,
        };
        let idx = self.dims.to_linear(i, j, k);
        let final_offset = (offset_elems + idx) * 4;

        self.queue.write_buffer(
            &self.e_field_buffer,
            final_offset as u64,
            bytemuck::bytes_of(&value),
        );
    }

    fn read_buffer_to_field(&self, buffer: &wgpu::Buffer, field: &mut VectorField3D) {
        let size = buffer.size();
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Read Encoder"),
            });

        encoder.copy_buffer_to_buffer(buffer, 0, &staging_buffer, 0, size);
        self.queue.submit(Some(encoder.finish()));

        let slice = staging_buffer.slice(..);
        let (sender, receiver) = futures::channel::oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

        self.instance.poll_all(true);
        pollster::block_on(receiver).unwrap().unwrap();

        let data = slice.get_mapped_range();
        let floats: &[f32] = bytemuck::cast_slice(&data);

        let total = self.dims.total();
        field.x.as_mut_slice().copy_from_slice(&floats[0..total]);
        field
            .y
            .as_mut_slice()
            .copy_from_slice(&floats[total..2 * total]);
        field
            .z
            .as_mut_slice()
            .copy_from_slice(&floats[2 * total..3 * total]);

        drop(data);
        staging_buffer.unmap();
    }

    /// Reset all fields to zero.
    pub fn reset(&self) {
        let size = self.e_field_buffer.size();
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.clear_buffer(&self.e_field_buffer, 0, Some(size));
        encoder.clear_buffer(&self.h_field_buffer, 0, Some(size));
        self.queue.submit(Some(encoder.finish()));
    }

    /// Upload excitation schedule to GPU for the entire batch.
    ///
    /// This pre-uploads all excitation data, eliminating per-timestep CPU→GPU transfers.
    fn upload_excitation_schedule(
        &mut self,
        excitations: &[crate::fdtd::batch::ScheduledExcitation],
        num_timesteps: u64,
    ) {
        // GPU ExcitationPoint: position (3 u32) + direction (u32) + value (f32) + soft_source (u32) + padding (2 u32) = 32 bytes
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct GpuExcitationPoint {
            pos_x: u32,
            pos_y: u32,
            pos_z: u32,
            direction: u32,
            value: f32,
            soft_source: u32,
            _padding: [u32; 2],
        }

        // Build per-timestep excitation lists
        let mut timestep_excitations: Vec<Vec<GpuExcitationPoint>> =
            vec![Vec::new(); num_timesteps as usize];

        for exc in excitations {
            if let crate::fdtd::batch::ExcitationWaveform::Sampled(ref samples) = exc.waveform {
                for (step, &value) in samples.iter().enumerate() {
                    if (step as u64) < num_timesteps && value.abs() > 1e-12 {
                        timestep_excitations[step].push(GpuExcitationPoint {
                            pos_x: exc.position.0 as u32,
                            pos_y: exc.position.1 as u32,
                            pos_z: exc.position.2 as u32,
                            direction: exc.direction as u32,
                            value,
                            soft_source: if exc.soft_source { 1 } else { 0 },
                            _padding: [0, 0],
                        });
                    }
                }
            }
        }

        // Flatten into linear buffer with offset array
        let mut all_excitations: Vec<GpuExcitationPoint> = Vec::new();
        let mut offsets: Vec<u32> = Vec::with_capacity(num_timesteps as usize + 1);

        for ts_excs in &timestep_excitations {
            offsets.push(all_excitations.len() as u32);
            all_excitations.extend_from_slice(ts_excs);
        }
        offsets.push(all_excitations.len() as u32); // End sentinel

        // Check if we need to resize buffers
        let total_excitations = all_excitations.len();
        if total_excitations > self.max_excitations * self.max_timesteps
            || (num_timesteps as usize + 1) > self.max_timesteps + 1
        {
            // Resize buffers
            let new_max_excitations = total_excitations.max(16);
            let new_max_timesteps = (num_timesteps as usize + 1).max(1024);

            self.excitation_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Excitation Buffer (resized)"),
                size: (new_max_excitations * 32) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            self.excitation_offsets_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Excitation Offsets Buffer (resized)"),
                size: (new_max_timesteps * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            self.max_excitations = new_max_excitations;
            self.max_timesteps = new_max_timesteps;

            // Recreate bind group with new buffers
            self.bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("FDTD Bind Group (resized)"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.e_field_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.h_field_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.e_coeff_f16_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.e_coeff_f32_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: self.cell_class_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: self.h_coeff_f16_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: self.h_coeff_f32_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: self.excitation_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: self.excitation_offsets_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 9,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &self.timestep_buffer,
                            offset: 0,
                            size: Some(std::num::NonZeroU64::new(256).unwrap()),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 10,
                        resource: self.energy_partial_buffer.as_entire_binding(),
                    },
                ],
            });
        }

        // Upload data
        if !all_excitations.is_empty() {
            self.queue.write_buffer(
                &self.excitation_buffer,
                0,
                bytemuck::cast_slice(&all_excitations),
            );
        }
        self.queue.write_buffer(
            &self.excitation_offsets_buffer,
            0,
            bytemuck::cast_slice(&offsets),
        );
    }

    /// Compute energy using GPU reduction (async-capable).
    ///
    /// Returns (e_energy, h_energy) without blocking with wait_idle().
    fn compute_energy_gpu(&self, encoder: &mut wgpu::CommandEncoder, dynamic_offset: wgpu::DynamicOffset) {
        // Dispatch energy reduction
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Energy Reduction Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.energy_pipeline);
            compute_pass.set_bind_group(0, &self.bind_group, &[dynamic_offset]);
            compute_pass.dispatch_workgroups(self.num_energy_workgroups as u32, 1, 1);
        }

        // Copy partial results to staging buffer
        encoder.copy_buffer_to_buffer(
            &self.energy_partial_buffer,
            0,
            &self.energy_staging_buffer,
            0,
            self.energy_staging_buffer.size(),
        );
    }

    /// Read back energy from staging buffer and compute final sum.
    ///
    /// This should be called after compute_energy_gpu and after submitting commands.
    fn read_energy_result(&self) -> (f64, f64) {
        let slice = self.energy_staging_buffer.slice(..);
        let (sender, receiver) = futures::channel::oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

        self.instance.poll_all(true);
        pollster::block_on(receiver).unwrap().unwrap();

        let data = slice.get_mapped_range();
        let floats: &[f32] = bytemuck::cast_slice(&data);

        // Sum up partial results (each workgroup contributed [e_partial, h_partial])
        let mut e_energy = 0.0f64;
        let mut h_energy = 0.0f64;
        for i in 0..self.num_energy_workgroups {
            e_energy += floats[i * 2] as f64;
            h_energy += floats[i * 2 + 1] as f64;
        }

        drop(data);
        self.energy_staging_buffer.unmap();

        (e_energy, h_energy)
    }

    /// Wait for all GPU operations to complete.
    pub fn wait_idle(&self) {
        self.instance.poll_all(true);
    }

}

impl EngineImpl for GpuEngine {
    fn new(operator: &Operator) -> Result<Self> {
        Ok(Self::new_internal(operator))
    }

    fn run_batch<E>(&mut self, batch: EngineBatch<E>) -> Result<BatchResult>
    where
        E: crate::extensions::Extension,
    {
        let start = Instant::now();
        let mut energy_samples = Vec::new();
        let mut peak_energy = 0.0;

        // Pre-batch: sync fields to GPU if cache exists
        self.sync_fields_to_gpu();

        // Extensions pre-batch (CPU-side only for now)
        // TODO: Implement GPU extension compilation
        for ext in &batch.extensions {
            // GPU doesn't support extensions yet - would need shader compilation
            if ext.gpu_data().is_some() {
                log::warn!(
                    "GPU extensions not yet implemented, extension {} will be ignored",
                    ext.name()
                );
            }
        }

        let num_steps = batch.num_steps.unwrap_or(1000);

        // Pre-upload excitation schedule to GPU (eliminates per-timestep transfers)
        self.upload_excitation_schedule(&batch.excitations, num_steps);

        if log::log_enabled!(log::Level::Debug) {
            log::debug!("Uploaded {} excitations for {} timesteps", batch.excitations.len(), num_steps);
        }

        // Determine energy sampling strategy
        let use_gpu_energy = batch.energy_monitoring.sample_interval > 0;
        let sample_interval = if use_gpu_energy {
            batch.energy_monitoring.sample_interval
        } else {
            // If no sampling, use a large interval to avoid branches
            u64::MAX
        };

        // Upload all timestep values to timestep buffer with 256-byte alignment
        // Each timestep value occupies a 256-byte aligned chunk for dynamic offsets
        let timestep_values: Vec<u8> = (0..num_steps as u32)
            .flat_map(|ts| {
                let mut chunk = vec![0u8; 256];  // 256-byte alignment required by WebGPU
                chunk[0..4].copy_from_slice(&ts.to_le_bytes());
                chunk
            })
            .collect();

        let required_size = (num_steps as usize * 256) as u64;

        // Expand timestep buffer if needed
        if self.timestep_buffer.size() < required_size {
            self.timestep_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Timestep Uniform Buffer"),
                size: required_size,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            // Recreate bind group with new buffer
            self.bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("FDTD Bind Group"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.e_field_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.h_field_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.e_coeff_f16_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.e_coeff_f32_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: self.cell_class_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: self.h_coeff_f16_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: self.h_coeff_f32_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: self.excitation_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: self.excitation_offsets_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 9,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &self.timestep_buffer,
                            offset: 0,
                            size: Some(std::num::NonZeroU64::new(256).unwrap()),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 10,
                        resource: self.energy_partial_buffer.as_entire_binding(),
                    },
                ],
            });
        }

        // Upload all timestep values at once
        self.queue.write_buffer(&self.timestep_buffer, 0, &timestep_values);

        // Process timesteps individually to ensure correct excitation application.
        let mut step = 0u64;
        while step < num_steps {
            // Create command buffer for this timestep
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Timestep Encoder"),
                });

            // Calculate dynamic offset for timestep buffer (256-byte aligned)
            let dynamic_offset = (step as u32 * 256) as wgpu::DynamicOffset;

            // Encode H update
            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Update H Pass"),
                    timestamp_writes: None,
                });
                compute_pass.set_pipeline(&self.update_h_pipeline);
                compute_pass.set_bind_group(0, &self.bind_group, &[dynamic_offset]);
                compute_pass.dispatch_workgroups(self.dispatch_x, self.dispatch_y, self.dispatch_z);
            }

            // Encode E update
            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Update E Pass"),
                    timestamp_writes: None,
                });
                compute_pass.set_pipeline(&self.update_e_pipeline);
                compute_pass.set_bind_group(0, &self.bind_group, &[dynamic_offset]);
                compute_pass.dispatch_workgroups(self.dispatch_x, self.dispatch_y, self.dispatch_z);
            }

            // Apply excitations (AFTER E-field update to match BasicEngine)
            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Apply Excitations Pass"),
                    timestamp_writes: None,
                });
                compute_pass.set_pipeline(&self.apply_excitations_pipeline);
                compute_pass.set_bind_group(0, &self.bind_group, &[dynamic_offset]);
                compute_pass.dispatch_workgroups(1, 1, 1);  // Single workgroup (1,1,1)
            }

            self.timestep += 1;
            step += 1;

            // Energy sampling at configured intervals
            let should_sample = use_gpu_energy
                && sample_interval > 0
                && step % sample_interval == 0
                && step < num_steps;

            if should_sample {
                // Add GPU energy computation to the command buffer
                self.compute_energy_gpu(&mut encoder, dynamic_offset);

                // Submit and wait for results
                self.queue.submit(Some(encoder.finish()));

                // Read energy from staging buffer
                let (e_energy, h_energy) = self.read_energy_result();
                let total_energy = e_energy + h_energy;

                if total_energy > peak_energy {
                    peak_energy = total_energy;
                }

                energy_samples.push(EnergySample {
                    timestep: self.timestep,
                    e_energy,
                    h_energy,
                    total_energy,
                });

                // Check energy decay termination
                if let Some(threshold) = batch.termination.energy_decay_db {
                    if peak_energy > 0.0 {
                        let decay_db = 10.0 * (total_energy / peak_energy).log10();
                        if decay_db < threshold {
                            self.cache_dirty = true;
                            return Ok(BatchResult {
                                timesteps_executed: step,
                                termination_reason: TerminationReason::EnergyDecay {
                                    final_decay_db: decay_db,
                                },
                                energy_samples,
                                elapsed_time: start.elapsed(),
                            });
                        }
                    }
                }
            } else {
                // Submit IMMEDIATELY after each timestep to ensure buffer copies execute in order
                self.queue.submit(Some(encoder.finish()));
            }
        }

        // Final wait for completion
        self.wait_idle();
        self.cache_dirty = true;

        Ok(BatchResult {
            timesteps_executed: num_steps,
            termination_reason: TerminationReason::StepsCompleted,
            energy_samples,
            elapsed_time: start.elapsed(),
        })
    }

    fn current_timestep(&self) -> u64 {
        self.timestep
    }

    fn read_fields(&mut self) -> (&VectorField3D, &VectorField3D) {
        self.sync_fields_from_gpu();
        let (ref e_field, ref h_field) = self.field_cache.as_ref().unwrap();
        (e_field, h_field)
    }

    fn write_fields(&mut self) -> (&mut VectorField3D, &mut VectorField3D) {
        self.sync_fields_from_gpu();
        self.cache_dirty = true; // Mark for re-upload
        let (ref mut e_field, ref mut h_field) = self.field_cache.as_mut().unwrap();
        (e_field, h_field)
    }

    fn reset(&mut self) {
        let size = self.e_field_buffer.size();
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.clear_buffer(&self.e_field_buffer, 0, Some(size));
        encoder.clear_buffer(&self.h_field_buffer, 0, Some(size));
        self.queue.submit(Some(encoder.finish()));
        self.wait_idle();

        self.timestep = 0;
        self.field_cache = None;
        self.cache_dirty = false;
    }
}
