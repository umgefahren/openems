//! Engine implementation trait.
//!
//! This module defines the core trait that all FDTD engine implementations
//! must satisfy. This trait is used internally for compile-time dispatch
//! via the Engine enum.

use crate::arrays::VectorField3D;
use crate::fdtd::batch::{BatchResult, EngineBatch};
use crate::fdtd::Operator;
use crate::Result;

/// Core trait that all engine implementations must satisfy.
///
/// This trait defines the interface for FDTD engines with batching support.
/// It is generic over the extension type `E`, enabling zero-cost abstraction
/// without trait objects.
///
/// # Type Parameters
/// * `E` - Extension type (implements Extension trait)
///
/// # Implementation Note
/// This trait is used for internal abstraction. External code should use
/// the `Engine` enum which provides compile-time dispatch to implementations.
pub trait EngineImpl: Sized + Send {
    /// Create a new engine from an operator.
    ///
    /// # Arguments
    /// * `operator` - The FDTD operator containing coefficients and grid info
    ///
    /// # Returns
    /// A new engine instance ready to execute timesteps.
    fn new(operator: &Operator) -> Result<Self>;

    /// Execute a batch of timesteps with extensions.
    ///
    /// This is the primary method for running simulations. It executes
    /// multiple timesteps as a batch, applying extensions and monitoring
    /// termination conditions.
    ///
    /// # Type Parameters
    /// * `E` - Extension type implementing the Extension trait
    ///
    /// # Arguments
    /// * `batch` - Configuration for the batch including timesteps, excitations, extensions
    ///
    /// # Returns
    /// Results including timesteps executed, termination reason, and energy samples.
    fn run_batch<E>(&mut self, batch: EngineBatch<E>) -> Result<BatchResult>
    where
        E: crate::extensions::Extension;

    /// Get the current timestep number.
    fn current_timestep(&self) -> u64;

    /// Read access to electromagnetic fields.
    ///
    /// Returns references to the E and H field data.
    /// For GPU engines, this may trigger a GPU→CPU transfer.
    fn read_fields(&mut self) -> (&VectorField3D, &VectorField3D);

    /// Write access to electromagnetic fields.
    ///
    /// Returns mutable references to the E and H field data.
    /// For GPU engines, this marks the CPU cache as dirty.
    fn write_fields(&mut self) -> (&mut VectorField3D, &mut VectorField3D);

    /// Reset the engine to initial state.
    ///
    /// Clears all fields and resets the timestep counter.
    fn reset(&mut self);
}
