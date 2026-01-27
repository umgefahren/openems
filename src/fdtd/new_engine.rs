//! Unified FDTD Engine with enum-based dispatch.
//!
//! This module provides the main `Engine` enum that dispatches to specific
//! engine implementations (Basic, SIMD, Parallel, GPU) using compile-time
//! macro-based dispatch instead of trait objects.

use crate::arrays::VectorField3D;
use crate::extensions::Extension;
use crate::fdtd::{
    BasicEngine, BatchResult, EngineBatch, EngineImpl, EngineType, GpuEngine, Operator,
    ParallelEngine, SimdEngine,
};
use crate::Result;

/// Main FDTD engine enum with compile-time dispatch.
///
/// This enum provides a unified interface to different engine implementations
/// without using trait objects. Method calls are dispatched using macros that
/// expand to match statements, enabling full compiler optimization.
pub enum Engine {
    /// Single-threaded reference implementation
    Basic(BasicEngine),
    /// SIMD-accelerated implementation
    Simd(SimdEngine),
    /// Multi-threaded SIMD implementation
    Parallel(ParallelEngine),
    /// GPU-accelerated implementation (WebGPU)
    Gpu(GpuEngine),
}

/// Macro for dispatching engine methods to the appropriate implementation.
///
/// This expands to a match statement that calls the method on the specific
/// engine variant, enabling compile-time dispatch without vtables.
macro_rules! dispatch_engine {
    ($self:expr, $method:ident($($args:expr),*)) => {
        match $self {
            Engine::Basic(e) => e.$method($($args),*),
            Engine::Simd(e) => e.$method($($args),*),
            Engine::Parallel(e) => e.$method($($args),*),
            Engine::Gpu(e) => e.$method($($args),*),
        }
    };
}

impl Engine {
    /// Create a new engine from an operator.
    ///
    /// # Arguments
    /// * `operator` - The FDTD operator containing grid and coefficients
    /// * `engine_type` - Which engine implementation to use
    ///
    /// # Returns
    /// A new engine instance ready to execute timesteps.
    pub fn new(operator: &Operator, engine_type: EngineType) -> Result<Self> {
        Ok(match engine_type {
            EngineType::Basic => Engine::Basic(BasicEngine::new(operator)?),
            EngineType::Simd => Engine::Simd(SimdEngine::new(operator)?),
            EngineType::Parallel => Engine::Parallel(ParallelEngine::new(operator)?),
            EngineType::Gpu => Engine::Gpu(GpuEngine::new(operator)?),
        })
    }

    /// Execute a batch of timesteps with extensions.
    ///
    /// This is the main method for running simulations. It dispatches to the
    /// appropriate engine implementation based on the engine type.
    ///
    /// # Type Parameters
    /// * `E` - Extension type implementing the Extension trait
    ///
    /// # Arguments
    /// * `batch` - Batch configuration including timesteps, excitations, extensions
    ///
    /// # Returns
    /// Results including timesteps executed, termination reason, energy samples.
    pub fn run_batch<E>(&mut self, batch: EngineBatch<E>) -> Result<BatchResult>
    where
        E: Extension,
    {
        dispatch_engine!(self, run_batch(batch))
    }

    /// Get the current timestep number.
    #[inline]
    pub fn current_timestep(&self) -> u64 {
        dispatch_engine!(self, current_timestep())
    }

    /// Read access to electromagnetic fields.
    ///
    /// Returns references to the E and H field data.
    /// For GPU engines, this may trigger a GPU→CPU transfer.
    #[inline]
    pub fn read_fields(&mut self) -> (&VectorField3D, &VectorField3D) {
        dispatch_engine!(self, read_fields())
    }

    /// Write access to electromagnetic fields.
    ///
    /// Returns mutable references to the E and H field data.
    /// For GPU engines, this marks the CPU cache as dirty.
    #[inline]
    pub fn write_fields(&mut self) -> (&mut VectorField3D, &mut VectorField3D) {
        dispatch_engine!(self, write_fields())
    }

    /// Reset the engine to initial state.
    ///
    /// Clears all fields and resets the timestep counter.
    #[inline]
    pub fn reset(&mut self) {
        dispatch_engine!(self, reset())
    }

    /// Get the engine type.
    pub fn engine_type(&self) -> EngineType {
        match self {
            Engine::Basic(_) => EngineType::Basic,
            Engine::Simd(_) => EngineType::Simd,
            Engine::Parallel(_) => EngineType::Parallel,
            Engine::Gpu(_) => EngineType::Gpu,
        }
    }

    // Compatibility methods for old Engine API

    /// Get reference to E-field (compatibility method).
    #[inline]
    pub fn e_field(&mut self) -> &VectorField3D {
        let (e, _) = self.read_fields();
        e
    }

    /// Get reference to H-field (compatibility method).
    #[inline]
    pub fn h_field(&mut self) -> &VectorField3D {
        let (_, h) = self.read_fields();
        h
    }

    /// Get references to both fields (compatibility method).
    #[inline]
    pub fn fields(&mut self) -> (&VectorField3D, &VectorField3D) {
        self.read_fields()
    }

    /// Get current timestep (compatibility method).
    #[inline]
    pub fn timestep(&self) -> u64 {
        self.current_timestep()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fdtd::{BoundaryConditions, EnergyMonitorConfig, EngineBatch, TerminationConfig};
    use crate::geometry::Grid;

    #[test]
    fn test_engine_creation_basic() {
        let grid = Grid::uniform(10, 10, 10, 1e-3);
        let operator = Operator::new(grid, BoundaryConditions::all_pec()).unwrap();
        let engine = Engine::new(&operator, EngineType::Basic).unwrap();
        assert_eq!(engine.engine_type(), EngineType::Basic);
    }

    #[test]
    fn test_engine_creation_simd() {
        let grid = Grid::uniform(10, 10, 10, 1e-3);
        let operator = Operator::new(grid, BoundaryConditions::all_pec()).unwrap();
        let engine = Engine::new(&operator, EngineType::Simd).unwrap();
        assert_eq!(engine.engine_type(), EngineType::Simd);
    }

    #[test]
    fn test_engine_creation_parallel() {
        let grid = Grid::uniform(10, 10, 10, 1e-3);
        let operator = Operator::new(grid, BoundaryConditions::all_pec()).unwrap();
        let engine = Engine::new(&operator, EngineType::Parallel).unwrap();
        assert_eq!(engine.engine_type(), EngineType::Parallel);
    }

    #[test]
    fn test_engine_timestep() {
        let grid = Grid::uniform(10, 10, 10, 1e-3);
        let operator = Operator::new(grid, BoundaryConditions::all_pec()).unwrap();
        let mut engine = Engine::new(&operator, EngineType::Basic).unwrap();

        assert_eq!(engine.current_timestep(), 0);

        // Run a small batch
        #[derive(Debug)]
        struct NoOpExtension;
        impl crate::extensions::Extension for NoOpExtension {
            fn name(&self) -> &str {
                "noop"
            }
            fn apply_step<E>(&mut self, _engine: &mut E, _step: u64) -> Result<()>
            where
                E: EngineImpl,
            {
                Ok(())
            }
        }

        let batch = EngineBatch {
            num_steps: Some(10),
            excitations: vec![],
            extensions: vec![NoOpExtension],
            termination: TerminationConfig::default(),
            energy_monitoring: EnergyMonitorConfig::default(),
        };

        engine.run_batch(batch).unwrap();
        assert_eq!(engine.current_timestep(), 10);
    }

    #[test]
    fn test_engine_reset() {
        let grid = Grid::uniform(10, 10, 10, 1e-3);
        let operator = Operator::new(grid, BoundaryConditions::all_pec()).unwrap();
        let mut engine = Engine::new(&operator, EngineType::Basic).unwrap();

        // Run some timesteps
        #[derive(Debug)]
        struct NoOpExtension;
        impl crate::extensions::Extension for NoOpExtension {
            fn name(&self) -> &str {
                "noop"
            }
            fn apply_step<E>(&mut self, _engine: &mut E, _step: u64) -> Result<()>
            where
                E: EngineImpl,
            {
                Ok(())
            }
        }

        let batch = EngineBatch {
            num_steps: Some(5),
            excitations: vec![],
            extensions: vec![NoOpExtension],
            termination: TerminationConfig::default(),
            energy_monitoring: EnergyMonitorConfig::default(),
        };

        engine.run_batch(batch).unwrap();
        assert_eq!(engine.current_timestep(), 5);

        // Reset and verify
        engine.reset();
        assert_eq!(engine.current_timestep(), 0);
    }

    #[test]
    fn test_engine_field_access() {
        let grid = Grid::uniform(10, 10, 10, 1e-3);
        let operator = Operator::new(grid, BoundaryConditions::all_pec()).unwrap();
        let mut engine = Engine::new(&operator, EngineType::Basic).unwrap();

        {
            let (e_field, h_field) = engine.read_fields();
            assert_eq!(e_field.energy(), 0.0);
            assert_eq!(h_field.energy(), 0.0);
        }

        {
            let (e_field, h_field) = engine.write_fields();
            e_field.x.set(5, 5, 5, 1.0);
            h_field.y.set(5, 5, 5, 2.0);
        }

        {
            let (e_field, h_field) = engine.read_fields();
            assert!(e_field.energy() > 0.0);
            assert!(h_field.energy() > 0.0);
        }
    }
}
