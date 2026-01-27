//! Extension trait for batching-capable FDTD extensions.
//!
//! This module provides a generic extension system that works across all
//! engine types (CPU and GPU) without requiring trait objects. Extensions
//! provide data that engines can query and use as appropriate.

use crate::Result;

/// Core extension trait for FDTD simulations.
///
/// Extensions provide data and functionality that engines can incorporate
/// into their execution. The trait is designed to be engine-agnostic:
/// - CPU engines call apply_step() for each timestep
/// - GPU engines query gpu_data() to compile shaders once per batch
///
/// # Design Philosophy
/// Extensions provide "what" (data, shader code), engines decide "how" (execution strategy).
pub trait Extension: Sized + Send {
    /// Extension name for logging and debugging.
    fn name(&self) -> &str;

    /// Pre-batch initialization hook.
    ///
    /// Called once before a batch begins. Use this to:
    /// - Initialize internal state
    /// - Validate configuration
    /// - Allocate temporary buffers
    ///
    /// # Type Parameters
    /// * `E` - Engine implementation type
    fn pre_batch<E>(&mut self, _engine: &mut E) -> Result<()>
    where
        E: crate::fdtd::EngineImpl,
    {
        Ok(()) // Default: no-op
    }

    /// Apply extension for a single timestep.
    ///
    /// Called by CPU engines for each timestep. GPU engines typically
    /// don't call this (they compile extensions into shaders instead).
    ///
    /// # Type Parameters
    /// * `E` - Engine implementation type
    ///
    /// # Arguments
    /// * `engine` - Mutable reference to engine (for field access)
    /// * `step` - Current timestep number
    fn apply_step<E>(&mut self, engine: &mut E, step: u64) -> Result<()>
    where
        E: crate::fdtd::EngineImpl;

    /// Post-batch cleanup hook.
    ///
    /// Called once after a batch completes. Use this to:
    /// - Write output data
    /// - Deallocate temporary buffers
    /// - Finalize internal state
    ///
    /// # Type Parameters
    /// * `E` - Engine implementation type
    fn post_batch<E>(&mut self, _engine: &mut E) -> Result<()>
    where
        E: crate::fdtd::EngineImpl,
    {
        Ok(()) // Default: no-op
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
    /// Return None if this extension doesn't support GPU execution.
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
}

/// Data that GPU extensions provide for shader compilation.
#[derive(Debug, Clone)]
pub struct GpuExtensionData {
    /// WGSL shader code to inject into the main shader.
    ///
    /// This should define functions that will be called at appropriate
    /// points in the FDTD update (e.g., apply_pml_e, apply_pml_h).
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
    // Can be extended with other CPU-specific metadata
}

/// Heterogeneous extension enum for storing multiple extension types.
///
/// Use this when you need to store extensions of different concrete types
/// in the same collection. The enum uses macro-based dispatch to forward
/// method calls to the appropriate implementation.
///
/// NOTE: Individual extension variants should be added here once they
/// implement the Extension trait.
#[derive(Debug)]
pub enum AnyExtension {
    /// Placeholder variant - individual extensions will be added as they're ported
    /// to the new Extension trait
    Placeholder(std::marker::PhantomData<()>),
}

impl Extension for AnyExtension {
    fn name(&self) -> &str {
        match self {
            AnyExtension::Placeholder(_) => "placeholder",
        }
    }

    fn apply_step<E>(&mut self, _engine: &mut E, _step: u64) -> Result<()>
    where
        E: crate::fdtd::EngineImpl,
    {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mock extension for testing
    struct MockExtension {
        name: String,
        apply_count: usize,
    }

    impl Extension for MockExtension {
        fn name(&self) -> &str {
            &self.name
        }

        fn apply_step<E>(&mut self, _engine: &mut E, _step: u64) -> Result<()>
        where
            E: crate::fdtd::EngineImpl,
        {
            self.apply_count += 1;
            Ok(())
        }
    }

    #[test]
    fn test_extension_trait_name() {
        let ext = MockExtension {
            name: "TestExt".to_string(),
            apply_count: 0,
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
            apply_count: 0,
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
    }
}
