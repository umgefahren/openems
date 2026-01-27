//! Batching support for FDTD engines.
//!
//! This module provides types for batched execution of FDTD timesteps,
//! enabling significant performance improvements by reducing CPU↔GPU
//! synchronization overhead and allowing engines to optimize across
//! multiple timesteps.

use crate::arrays::VectorField3D;
use crate::fdtd::{Excitation, ExcitationType};
use instant::Duration;

/// Configuration for a batch of timesteps.
///
/// This structure is generic over the extension type `E`, allowing
/// engines to work with different extension implementations without
/// requiring trait objects.
pub struct EngineBatch<E> {
    /// Number of timesteps to execute (None = run until termination)
    pub num_steps: Option<u64>,

    /// Pre-scheduled excitations with sampled waveforms
    pub excitations: Vec<ScheduledExcitation>,

    /// Extensions to apply during batch execution
    pub extensions: Vec<E>,

    /// Termination conditions
    pub termination: TerminationConfig,

    /// Energy monitoring settings
    pub energy_monitoring: EnergyMonitorConfig,
}

/// Result from executing a batch of timesteps.
#[derive(Debug, Clone)]
pub struct BatchResult {
    /// Number of timesteps actually executed
    pub timesteps_executed: u64,

    /// Reason for termination
    pub termination_reason: TerminationReason,

    /// Energy samples collected during execution
    pub energy_samples: Vec<EnergySample>,

    /// Wall clock time elapsed
    pub elapsed_time: Duration,
}

/// Reason why a batch terminated.
#[derive(Debug, Clone, PartialEq)]
pub enum TerminationReason {
    /// Completed requested number of steps
    StepsCompleted,

    /// Energy decay threshold reached
    EnergyDecay {
        /// Final decay in dB relative to peak
        final_decay_db: f64,
    },

    /// Extension requested early termination
    ExtensionStop {
        /// Reason provided by extension
        reason: String,
    },
}

/// Excitation with pre-sampled waveform for efficient batching.
#[derive(Debug, Clone)]
pub struct ScheduledExcitation {
    /// Grid position (i, j, k)
    pub position: (usize, usize, usize),

    /// Field component direction (0=x, 1=y, 2=z)
    pub direction: usize,

    /// Is this a soft source? (adds to field instead of replacing)
    pub soft_source: bool,

    /// Waveform data
    pub waveform: ExcitationWaveform,
}

/// Excitation waveform representation.
#[derive(Debug, Clone)]
pub enum ExcitationWaveform {
    /// Pre-computed samples (one per timestep)
    Sampled(Vec<f32>),

    /// Analytical function (evaluated on-the-fly)
    Analytical {
        /// Original excitation type
        excitation_type: ExcitationType,
        /// Amplitude scaling
        amplitude: f64,
        /// Starting timestep
        start_timestep: u64,
    },
}

impl ScheduledExcitation {
    /// Create a scheduled excitation by pre-sampling an excitation.
    ///
    /// # Arguments
    /// * `exc` - The excitation to sample
    /// * `start_timestep` - First timestep in the batch
    /// * `num_timesteps` - Number of timesteps in the batch
    /// * `dt` - Timestep size in seconds
    pub fn from_excitation(
        exc: &Excitation,
        start_timestep: u64,
        num_timesteps: u64,
        dt: f64,
    ) -> Self {
        // Pre-sample the waveform for all timesteps in this batch
        let mut samples = Vec::with_capacity(num_timesteps as usize);
        for step in 0..num_timesteps {
            let t = (start_timestep + step) as f64 * dt;
            samples.push(exc.evaluate(t) as f32);
        }

        Self {
            position: exc.position,
            direction: exc.direction,
            soft_source: exc.soft_source,
            waveform: ExcitationWaveform::Sampled(samples),
        }
    }

    /// Create an analytical scheduled excitation (not pre-sampled).
    ///
    /// This is useful for GPU engines that can evaluate the excitation
    /// function directly in shaders.
    pub fn from_excitation_analytical(exc: &Excitation, start_timestep: u64) -> Self {
        Self {
            position: exc.position,
            direction: exc.direction,
            soft_source: exc.soft_source,
            waveform: ExcitationWaveform::Analytical {
                excitation_type: exc.excitation_type.clone(),
                amplitude: exc.amplitude,
                start_timestep,
            },
        }
    }
}

/// Configuration for batch termination conditions.
#[derive(Debug, Clone)]
pub struct TerminationConfig {
    /// Maximum number of timesteps (enforced even if other conditions not met)
    pub max_timesteps: Option<u64>,

    /// Energy decay threshold in dB (relative to peak)
    pub energy_decay_db: Option<f64>,

    /// How often to check termination conditions (in timesteps)
    pub check_interval: u64,
}

impl Default for TerminationConfig {
    fn default() -> Self {
        Self {
            max_timesteps: None,
            energy_decay_db: None,
            check_interval: 100,
        }
    }
}

/// Configuration for energy monitoring during batch execution.
#[derive(Debug, Clone)]
pub struct EnergyMonitorConfig {
    /// Sample energy every N timesteps (0 = disabled)
    pub sample_interval: u64,

    /// Track peak energy for decay calculations
    pub track_peak: bool,

    /// Decay threshold for early termination (in dB)
    pub decay_threshold: Option<f64>,
}

impl Default for EnergyMonitorConfig {
    fn default() -> Self {
        Self {
            sample_interval: 0,
            track_peak: false,
            decay_threshold: None,
        }
    }
}

/// Energy sample at a specific timestep.
#[derive(Debug, Clone, Copy)]
pub struct EnergySample {
    /// Timestep when sample was taken
    pub timestep: u64,

    /// Electric field energy
    pub e_energy: f64,

    /// Magnetic field energy
    pub h_energy: f64,

    /// Total energy (E + H)
    pub total_energy: f64,
}

impl EnergySample {
    /// Create a new energy sample.
    pub fn new(timestep: u64, e_energy: f64, h_energy: f64) -> Self {
        Self {
            timestep,
            e_energy,
            h_energy,
            total_energy: e_energy + h_energy,
        }
    }

    /// Compute energy from field data.
    pub fn from_fields(timestep: u64, e_field: &VectorField3D, h_field: &VectorField3D) -> Self {
        let e_energy = e_field.energy();
        let h_energy = h_field.energy();
        Self::new(timestep, e_energy, h_energy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scheduled_excitation_from_excitation() {
        let exc = Excitation::gaussian(1e9, 0.5, 2, (10, 10, 10));
        let dt = 1e-12;
        let num_steps = 100;

        let scheduled = ScheduledExcitation::from_excitation(&exc, 0, num_steps, dt);

        assert_eq!(scheduled.position, (10, 10, 10));
        assert_eq!(scheduled.direction, 2);
        assert!(scheduled.soft_source);

        if let ExcitationWaveform::Sampled(ref samples) = scheduled.waveform {
            assert_eq!(samples.len(), 100);

            // Check that samples match original excitation
            for (i, &sample) in samples.iter().enumerate() {
                let t = i as f64 * dt;
                let expected = exc.evaluate(t) as f32;
                assert!((sample - expected).abs() < 1e-6);
            }
        } else {
            panic!("Expected Sampled waveform");
        }
    }

    #[test]
    fn test_energy_sample_from_fields() {
        use crate::arrays::Dimensions;

        let dims = Dimensions::new(10, 10, 10);
        let mut e_field = VectorField3D::new(dims);
        let mut h_field = VectorField3D::new(dims);

        // Set some field values
        e_field.x.set(5, 5, 5, 1.0);
        h_field.y.set(5, 5, 5, 2.0);

        let sample = EnergySample::from_fields(42, &e_field, &h_field);

        assert_eq!(sample.timestep, 42);
        assert!(sample.e_energy > 0.0);
        assert!(sample.h_energy > 0.0);
        assert!((sample.total_energy - (sample.e_energy + sample.h_energy)).abs() < 1e-10);
    }

    #[test]
    fn test_termination_config_default() {
        let config = TerminationConfig::default();
        assert_eq!(config.max_timesteps, None);
        assert_eq!(config.energy_decay_db, None);
        assert_eq!(config.check_interval, 100);
    }

    #[test]
    fn test_energy_monitor_config_default() {
        let config = EnergyMonitorConfig::default();
        assert_eq!(config.sample_interval, 0);
        assert!(!config.track_peak);
        assert_eq!(config.decay_threshold, None);
    }
}
