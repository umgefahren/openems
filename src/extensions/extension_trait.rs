//! Extension trait for batching-capable FDTD extensions.
//!
//! This module provides a generic extension system that works across all
//! engine types (CPU and GPU). Extensions hook into the FDTD update cycle
//! at specific points to modify field values.
//!
//! # FDTD Update Cycle
//!
//! The standard FDTD update cycle is:
//! 1. `pre_update_h` - Before H-field update (e.g., PML flux swap)
//! 2. H-field update: H = Da*H + Db*curl(E)
//! 3. `post_update_h` - After H-field update (e.g., PML correction)
//! 4. `pre_update_e` - Before E-field update (e.g., PML flux swap)
//! 5. E-field update: E = Ca*E + Cb*curl(H)
//! 6. `post_update_e` - After E-field update (e.g., dispersive material ADE, PML correction)
//! 7. Apply excitations
//!
//! # GPU Support
//!
//! For GPU engines, extensions can either:
//! - Provide `gpu_data()` with WGSL shader code that gets compiled into the main shader
//! - Fall back to CPU-side processing (engine syncs fields to CPU, applies extension, syncs back)

use crate::arrays::VectorField3D;
use crate::Result;

/// Core extension trait for FDTD simulations.
///
/// Extensions hook into the FDTD update cycle to modify field values.
/// Each hook has a default no-op implementation, so extensions only need
/// to implement the hooks they require.
pub trait Extension: Sized + Send {
    /// Extension name for logging and debugging.
    fn name(&self) -> &str;

    /// Pre-batch initialization hook.
    ///
    /// Called once before a batch begins. Use this to:
    /// - Initialize internal state
    /// - Validate configuration
    /// - Allocate temporary buffers
    fn pre_batch<E>(&mut self, _engine: &mut E) -> Result<()>
    where
        E: crate::fdtd::EngineImpl,
    {
        Ok(())
    }

    /// Post-batch cleanup hook.
    ///
    /// Called once after a batch completes. Use this to:
    /// - Write output data
    /// - Deallocate temporary buffers
    /// - Finalize internal state
    fn post_batch<E>(&mut self, _engine: &mut E) -> Result<()>
    where
        E: crate::fdtd::EngineImpl,
    {
        Ok(())
    }

    /// Pre-H-update hook.
    ///
    /// Called before the H-field update. Used by PML to swap fields with flux
    /// and compute intermediate values.
    ///
    /// # Arguments
    /// * `h_field` - Mutable reference to H-field (can be modified)
    /// * `e_field` - Reference to E-field (read-only)
    /// * `step` - Current timestep number
    fn pre_update_h(
        &mut self,
        _h_field: &mut VectorField3D,
        _e_field: &VectorField3D,
        _step: u64,
    ) -> Result<()> {
        Ok(())
    }

    /// Post-H-update hook.
    ///
    /// Called after the H-field update. Used by PML to complete the
    /// split-field correction.
    ///
    /// # Arguments
    /// * `h_field` - Mutable reference to H-field (can be modified)
    /// * `e_field` - Reference to E-field (read-only)
    /// * `step` - Current timestep number
    fn post_update_h(
        &mut self,
        _h_field: &mut VectorField3D,
        _e_field: &VectorField3D,
        _step: u64,
    ) -> Result<()> {
        Ok(())
    }

    /// Pre-E-update hook.
    ///
    /// Called before the E-field update. Used by PML to swap fields with flux
    /// and compute intermediate values.
    ///
    /// # Arguments
    /// * `e_field` - Mutable reference to E-field (can be modified)
    /// * `h_field` - Reference to H-field (read-only)
    /// * `step` - Current timestep number
    fn pre_update_e(
        &mut self,
        _e_field: &mut VectorField3D,
        _h_field: &VectorField3D,
        _step: u64,
    ) -> Result<()> {
        Ok(())
    }

    /// Post-E-update hook.
    ///
    /// Called after the E-field update. Used by:
    /// - PML to complete the split-field correction
    /// - Dispersive materials (Lorentz/Drude/Debye) for ADE update
    ///
    /// # Arguments
    /// * `e_field` - Mutable reference to E-field (can be modified)
    /// * `h_field` - Reference to H-field (read-only)
    /// * `step` - Current timestep number
    fn post_update_e(
        &mut self,
        _e_field: &mut VectorField3D,
        _h_field: &VectorField3D,
        _step: u64,
    ) -> Result<()> {
        Ok(())
    }

    /// Check for early termination.
    ///
    /// Called periodically during batch execution. Return Some(reason)
    /// to request early termination.
    fn check_termination(&self) -> Option<String> {
        None
    }

    /// Provide GPU-specific data for shader compilation.
    ///
    /// GPU engines query this method to get shader code and buffer data.
    /// Return None if this extension doesn't support GPU execution
    /// (the engine will fall back to CPU-side processing).
    fn gpu_data(&self) -> Option<GpuExtensionData> {
        None
    }

    /// Provide CPU-specific data.
    ///
    /// CPU engines can query this for optimized coefficient arrays or
    /// other CPU-specific metadata. Most extensions don't need this.
    fn cpu_data(&self) -> Option<CpuExtensionData> {
        None
    }

    /// Check if this extension requires CPU fallback on GPU.
    ///
    /// Returns true if the extension needs CPU-side processing even when
    /// running on GPU (i.e., gpu_data() returns None or is incomplete).
    fn requires_cpu_fallback(&self) -> bool {
        self.gpu_data().is_none()
    }
}

/// Data that GPU extensions provide for shader compilation.
#[derive(Debug, Clone)]
pub struct GpuExtensionData {
    /// WGSL shader code to inject into the main shader.
    ///
    /// This should define functions that will be called at appropriate
    /// points in the FDTD update:
    /// - `extension_pre_update_h(pos: vec3<u32>)` - before H update
    /// - `extension_post_update_h(pos: vec3<u32>)` - after H update
    /// - `extension_pre_update_e(pos: vec3<u32>)` - before E update
    /// - `extension_post_update_e(pos: vec3<u32>)` - after E update
    pub shader_code: String,

    /// Buffer descriptors for GPU data upload.
    pub buffers: Vec<GpuBufferDescriptor>,

    /// Bind group layout entries for the shader bindings.
    pub bind_group_entries: Vec<wgpu::BindGroupLayoutEntry>,
}

/// Descriptor for a GPU buffer to upload.
#[derive(Debug, Clone)]
pub struct GpuBufferDescriptor {
    /// Debug label for the buffer.
    pub label: String,

    /// Raw buffer data (as bytes).
    pub data: Vec<u8>,

    /// Buffer usage flags.
    pub usage: wgpu::BufferUsages,

    /// Binding index in the shader.
    pub binding: u32,
}

/// CPU-specific extension data (if needed).
#[derive(Debug, Clone)]
pub struct CpuExtensionData {
    /// Pre-computed coefficient arrays.
    pub coefficients: Option<Vec<f32>>,
}

/// Heterogeneous extension enum for storing multiple extension types.
///
/// Use this when you need to store extensions of different concrete types
/// in the same collection.
#[derive(Debug)]
pub enum AnyExtension {
    /// Placeholder variant
    Placeholder(std::marker::PhantomData<()>),
}

impl Extension for AnyExtension {
    fn name(&self) -> &str {
        match self {
            AnyExtension::Placeholder(_) => "placeholder",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mock extension for testing
    struct MockExtension {
        name: String,
        pre_h_count: usize,
        post_h_count: usize,
        pre_e_count: usize,
        post_e_count: usize,
    }

    impl Extension for MockExtension {
        fn name(&self) -> &str {
            &self.name
        }

        fn pre_update_h(
            &mut self,
            _h_field: &mut VectorField3D,
            _e_field: &VectorField3D,
            _step: u64,
        ) -> Result<()> {
            self.pre_h_count += 1;
            Ok(())
        }

        fn post_update_h(
            &mut self,
            _h_field: &mut VectorField3D,
            _e_field: &VectorField3D,
            _step: u64,
        ) -> Result<()> {
            self.post_h_count += 1;
            Ok(())
        }

        fn pre_update_e(
            &mut self,
            _e_field: &mut VectorField3D,
            _h_field: &VectorField3D,
            _step: u64,
        ) -> Result<()> {
            self.pre_e_count += 1;
            Ok(())
        }

        fn post_update_e(
            &mut self,
            _e_field: &mut VectorField3D,
            _h_field: &VectorField3D,
            _step: u64,
        ) -> Result<()> {
            self.post_e_count += 1;
            Ok(())
        }
    }

    #[test]
    fn test_extension_trait_name() {
        let ext = MockExtension {
            name: "TestExt".to_string(),
            pre_h_count: 0,
            post_h_count: 0,
            pre_e_count: 0,
            post_e_count: 0,
        };
        assert_eq!(ext.name(), "TestExt");
    }

    #[test]
    fn test_extension_default_hooks() {
        struct TestEngine;
        impl crate::fdtd::EngineImpl for TestEngine {
            fn new(_: &crate::fdtd::Operator) -> Result<Self> {
                Ok(TestEngine)
            }
            fn run_batch<E>(
                &mut self,
                _: crate::fdtd::batch::EngineBatch<E>,
            ) -> Result<crate::fdtd::batch::BatchResult>
            where
                E: Extension,
            {
                unimplemented!()
            }
            fn current_timestep(&self) -> u64 {
                0
            }
            fn read_fields(
                &mut self,
            ) -> (&crate::arrays::VectorField3D, &crate::arrays::VectorField3D) {
                unimplemented!()
            }
            fn write_fields(
                &mut self,
            ) -> (
                &mut crate::arrays::VectorField3D,
                &mut crate::arrays::VectorField3D,
            ) {
                unimplemented!()
            }
            fn reset(&mut self) {}
        }

        let mut ext = MockExtension {
            name: "Test".to_string(),
            pre_h_count: 0,
            post_h_count: 0,
            pre_e_count: 0,
            post_e_count: 0,
        };
        let mut engine = TestEngine;

        // Default pre_batch and post_batch should succeed
        ext.pre_batch(&mut engine).unwrap();
        ext.post_batch(&mut engine).unwrap();

        // Default check_termination returns None
        assert_eq!(ext.check_termination(), None);

        // Default gpu_data returns None
        assert!(ext.gpu_data().is_none());

        // Default cpu_data returns None
        assert!(ext.cpu_data().is_none());

        // requires_cpu_fallback should be true when gpu_data is None
        assert!(ext.requires_cpu_fallback());
    }
}
