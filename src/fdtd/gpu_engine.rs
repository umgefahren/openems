use crate::arrays::{Dimensions, VectorField3D};
use crate::fdtd::operator::Operator;
use half::f16;
use std::borrow::Cow;
use wgpu::util::DeviceExt;

/// GPU-accelerated FDTD engine using WebGPU.
pub struct GpuEngine {
    instance: wgpu::Instance,
    device: wgpu::Device,
    queue: wgpu::Queue,
    update_h_pipeline: wgpu::ComputePipeline,
    update_e_pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,

    // Buffers
    e_field_buffer: wgpu::Buffer,
    h_field_buffer: wgpu::Buffer,

    // Dimensions
    dims: Dimensions,

    // Workgroup dispatch size
    dispatch_x: u32,
    dispatch_y: u32,
    dispatch_z: u32,

    // State
    #[allow(dead_code)]
    supports_f16: bool,
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
    pub fn new(operator: &Operator) -> Self {
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
        for k in 0..dims.nz {
            for j in 0..dims.ny {
                for i in 0..dims.nx {
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

                    // Classify
                    // Heuristic: Use f32 if Ca != 1.0 (lossy) and Ca != 0.0 (PEC).
                    // Or if we implement PML later, we mark those regions.
                    // For now: Standard = 0, HighPrecision = 1.
                    let is_lossy = (ca[0] - 1.0).abs() > 1e-5 || (ca[0].abs() > 1e-5 && ca[0] < 0.99);
                    let class_id = if is_lossy { 1u32 } else { 0u32 };
                    cell_class_data.push(class_id);

                    // Pack data
                    let data_floats = [
                        ca[0], ca[1], ca[2], cb[0],
                        cb[1], cb[2], 0.0, 0.0
                    ];

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
        
        for k in 0..dims.nz {
            for j in 0..dims.ny {
                for i in 0..dims.nx {
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
                    
                    let data_floats = [
                        da[0], da[1], da[2], db[0],
                        db[1], db[2], 0.0, 0.0
                    ];

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

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("FDTD Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry { // E Field
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry { // H Field
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry { // Coeff f16 (E)
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry { // Coeff f32 (E)
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry { // Cell Class
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry { // Coeff f16 (H)
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry { // Coeff f32 (H)
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
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
            ],
        });

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

        // Dispatch dimensions swapped to match shader mapping:
        let dispatch_x = (dims.nz as u32).div_ceil(64); // Workgroup size 64 along X
        let dispatch_y = (dims.ny as u32).div_ceil(2); // Workgroup size 2 along Y
        let dispatch_z = (dims.nx as u32).div_ceil(2); // Workgroup size 2 along Z

        Self {
            instance,
            device,
            queue,
            update_h_pipeline,
            update_e_pipeline,
            bind_group,
            e_field_buffer,
            h_field_buffer,
            dims,
            dispatch_x,
            dispatch_y,
            dispatch_z,
            supports_f16,
        }
    }

    /// Perform one FDTD timestep on the GPU.
    pub fn step(&self) {
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
            compute_pass.set_bind_group(0, &self.bind_group, &[]);
            compute_pass.dispatch_workgroups(self.dispatch_x, self.dispatch_y, self.dispatch_z);
        }

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Update E Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.update_e_pipeline);
            compute_pass.set_bind_group(0, &self.bind_group, &[]);
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

    /// Wait for all GPU operations to complete.
    pub fn wait_idle(&self) {
        self.instance.poll_all(true);
    }
}
