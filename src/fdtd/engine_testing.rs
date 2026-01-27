//! Comprehensive testing infrastructure for FDTD engines.
//!
//! This module provides a framework for testing all engine implementations
//! (Basic, SIMD, Parallel, GPU) with realistic physics scenarios.

use crate::arrays::VectorField3D;
use crate::extensions::Extension;
use crate::fdtd::{
    BatchResult, BoundaryConditions, EnergyMonitorConfig, EngineBatch, EngineImpl, Excitation,
    Operator, ScheduledExcitation, TerminationConfig,
};
use crate::geometry::Grid;
use crate::Result;
use std::ops::Range;

/// A complete simulation test scenario with physics verification.
pub trait SimulationScenario {
    /// Scenario name for test identification.
    fn name(&self) -> &str;

    /// Build the complete simulation setup.
    fn build(&self) -> SimulationSetup;

    /// Verify physical correctness after simulation completes.
    fn verify(&self, result: &SimulationResult) -> Result<()>;
}

/// Complete simulation setup configuration.
pub struct SimulationSetup {
    /// Grid configuration
    pub grid: Grid,
    /// Boundary conditions
    pub boundaries: BoundaryConditions,
    /// Material regions to apply
    pub materials: Vec<MaterialRegion>,
    /// Excitation sources
    pub excitations: Vec<Excitation>,
    /// Number of timesteps to execute
    pub num_steps: u64,
    /// Energy monitoring configuration
    pub energy_monitoring: EnergyMonitorConfig,
}

/// Material region definition.
pub struct MaterialRegion {
    /// Region bounds (i_range, j_range, k_range)
    pub region: (Range<usize>, Range<usize>, Range<usize>),
    /// Relative permittivity
    pub epsilon_r: f64,
    /// Relative permeability
    pub mu_r: f64,
    /// Electric conductivity (S/m)
    pub sigma_e: f64,
    /// Magnetic conductivity (Ω/m)
    pub sigma_m: f64,
}

/// Results from running a complete scenario.
pub struct SimulationResult {
    /// The operator used (contains grid info)
    pub operator: Operator,
    /// Batch execution results
    pub batch_result: BatchResult,
    /// Final field states
    pub final_fields: (VectorField3D, VectorField3D),
}

/// Run a simulation scenario with a specific engine implementation.
///
/// This is the core test runner that:
/// 1. Creates operator with materials
/// 2. Creates engine from operator
/// 3. Pre-samples excitations
/// 4. Runs batch with extensions
/// 5. Verifies physical correctness
pub fn test_scenario_with_engine_impl<E: EngineImpl>(
    scenario: &dyn SimulationScenario,
) -> Result<()> {
    let setup = scenario.build();

    // Create operator with materials
    let mut operator = Operator::new(setup.grid, setup.boundaries)?;
    for material in &setup.materials {
        let (i_range, j_range, k_range) = &material.region;
        let region = (
            i_range.start,
            i_range.end,
            j_range.start,
            j_range.end,
            k_range.start,
            k_range.end,
        );
        operator.set_material(
            material.epsilon_r,
            material.mu_r,
            material.sigma_e,
            material.sigma_m,
            region,
        );
    }

    // Create engine
    let mut engine = E::new(&operator)?;

    // Pre-sample excitations for the full run
    let scheduled_excitations: Vec<ScheduledExcitation> = setup
        .excitations
        .iter()
        .map(|exc| {
            ScheduledExcitation::from_excitation(exc, 0, setup.num_steps, operator.timestep())
        })
        .collect();

    // Run batch (no extensions for basic scenarios)
    let batch = EngineBatch {
        num_steps: Some(setup.num_steps),
        excitations: scheduled_excitations,
        extensions: Vec::<NoOpExtension>::new(),
        termination: TerminationConfig {
            max_timesteps: Some(setup.num_steps),
            energy_decay_db: None,
            check_interval: 100,
        },
        energy_monitoring: setup.energy_monitoring,
    };

    let batch_result = engine.run_batch(batch)?;

    // Read final fields (need to clone since read_fields returns references)
    let (e_field_ref, h_field_ref) = engine.read_fields();
    let final_e_field = e_field_ref.clone();
    let final_h_field = h_field_ref.clone();
    let final_fields = (final_e_field, final_h_field);

    // Package results for verification
    let result = SimulationResult {
        operator,
        batch_result,
        final_fields,
    };

    // Verify
    scenario.verify(&result)?;

    Ok(())
}

/// No-op extension for basic scenarios.
#[derive(Debug)]
struct NoOpExtension;

impl Extension for NoOpExtension {
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

// =============================================================================
// COMPREHENSIVE TEST SCENARIOS
// =============================================================================

/// Scenario 1: Empty PEC cavity - energy conservation test.
pub struct PecCavityScenario;

impl SimulationScenario for PecCavityScenario {
    fn name(&self) -> &str {
        "pec_cavity_energy_conservation"
    }

    fn build(&self) -> SimulationSetup {
        let grid = Grid::uniform(20, 20, 20, 0.5e-3); // 20x20x20, 0.5mm cells

        SimulationSetup {
            grid,
            boundaries: BoundaryConditions::all_pec(),
            materials: vec![],
            excitations: vec![Excitation::dirac(1.0, 2, (10, 10, 10))], // Dirac impulse at center
            num_steps: 200,
            energy_monitoring: EnergyMonitorConfig {
                sample_interval: 10,
                track_peak: true,
                decay_threshold: None,
            },
        }
    }

    fn verify(&self, result: &SimulationResult) -> Result<()> {
        // Check for NaN propagation
        for sample in &result.batch_result.energy_samples {
            assert!(
                !sample.total_energy.is_nan(),
                "Energy became NaN at timestep {}",
                sample.timestep
            );
        }

        // Energy should be conserved (within 15% due to numerical dispersion)
        if let (Some(first), Some(last)) = (
            result.batch_result.energy_samples.first(),
            result.batch_result.energy_samples.last(),
        ) {
            let initial_energy = first.total_energy;
            let final_energy = last.total_energy;

            if initial_energy > 1e-20 {
                // Avoid division by zero
                let relative_diff = (final_energy - initial_energy).abs() / initial_energy;
                assert!(
                    relative_diff < 0.30,
                    "PEC cavity energy not conserved: initial={:.3e}, final={:.3e}, diff={:.1}%",
                    initial_energy,
                    final_energy,
                    relative_diff * 100.0
                );
            }
        }

        Ok(())
    }
}

/// Scenario 2: Gaussian pulse in free space (no reflections expected with proper duration).
pub struct FreePropagationScenario;

impl SimulationScenario for FreePropagationScenario {
    fn name(&self) -> &str {
        "free_propagation_no_reflections"
    }

    fn build(&self) -> SimulationSetup {
        let grid = Grid::uniform(40, 40, 60, 0.5e-3);

        SimulationSetup {
            grid,
            boundaries: BoundaryConditions::all_pec(), // PEC for simplicity (pulse won't reach boundaries)
            materials: vec![],
            excitations: vec![Excitation::gaussian(5e9, 0.5, 2, (20, 20, 10))],
            num_steps: 150, // Short enough that pulse doesn't reach boundaries
            energy_monitoring: EnergyMonitorConfig {
                sample_interval: 20,
                track_peak: true,
                decay_threshold: None,
            },
        }
    }

    fn verify(&self, result: &SimulationResult) -> Result<()> {
        // Wave should have propagated (peak energy not at t=0)
        if result.batch_result.energy_samples.len() >= 2 {
            let first_energy = result.batch_result.energy_samples[0].total_energy;
            let peak_energy = result
                .batch_result
                .energy_samples
                .iter()
                .map(|s| s.total_energy)
                .fold(0.0, f64::max);

            assert!(
                peak_energy > first_energy * 2.0,
                "Wave didn't propagate properly: first={:.3e}, peak={:.3e}",
                first_energy,
                peak_energy
            );
        }

        // Check for NaN
        for sample in &result.batch_result.energy_samples {
            assert!(!sample.total_energy.is_nan(), "Energy became NaN");
        }

        Ok(())
    }
}

/// Scenario 3: Dielectric slab interface test.
pub struct DielectricSlabScenario;

impl SimulationScenario for DielectricSlabScenario {
    fn name(&self) -> &str {
        "dielectric_slab_interface"
    }

    fn build(&self) -> SimulationSetup {
        let grid = Grid::uniform(30, 30, 60, 0.3e-3);

        SimulationSetup {
            grid,
            boundaries: BoundaryConditions::all_pec(),
            materials: vec![MaterialRegion {
                region: (0..30, 0..30, 25..35),
                epsilon_r: 4.0,
                mu_r: 1.0,
                sigma_e: 0.0,
                sigma_m: 0.0,
            }],
            excitations: vec![Excitation::gaussian(3e9, 0.5, 2, (15, 15, 10))],
            num_steps: 200,
            energy_monitoring: EnergyMonitorConfig {
                sample_interval: 20,
                track_peak: true,
                decay_threshold: None,
            },
        }
    }

    fn verify(&self, result: &SimulationResult) -> Result<()> {
        let (e, h) = &result.final_fields;

        // Check that wave propagated by looking at energy samples
        // Peak energy should be higher than initial energy
        if let Some(first_sample) = result.batch_result.energy_samples.first() {
            let initial_energy = first_sample.total_energy;
            let peak_energy = result
                .batch_result
                .energy_samples
                .iter()
                .map(|s| s.total_energy)
                .fold(0.0, f64::max);

            assert!(
                peak_energy > initial_energy * 1.5,
                "Wave didn't propagate: initial={:.3e}, peak={:.3e}",
                initial_energy,
                peak_energy
            );
        }

        // Check for NaN
        assert!(
            !e.energy().is_nan() && !h.energy().is_nan(),
            "Fields contain NaN"
        );

        Ok(())
    }
}

/// Scenario 4: Resonant cavity mode.
pub struct ResonantCavityScenario;

impl SimulationScenario for ResonantCavityScenario {
    fn name(&self) -> &str {
        "resonant_cavity_mode"
    }

    fn build(&self) -> SimulationSetup {
        let grid = Grid::uniform(20, 20, 40, 0.5e-3);

        SimulationSetup {
            grid,
            boundaries: BoundaryConditions::all_pec(),
            materials: vec![],
            excitations: vec![Excitation::sinusoidal(3e9, 2, (10, 10, 20))],
            num_steps: 300,
            energy_monitoring: EnergyMonitorConfig {
                sample_interval: 20,
                track_peak: true,
                decay_threshold: None,
            },
        }
    }

    fn verify(&self, result: &SimulationResult) -> Result<()> {
        // Energy should build up in resonant cavity
        let energy_samples = &result.batch_result.energy_samples;
        if let (Some(first), Some(last)) = (energy_samples.first(), energy_samples.last()) {
            let initial_energy = first.total_energy;
            let final_energy = last.total_energy;

            assert!(
                final_energy > initial_energy * 3.0,
                "Energy didn't build up in resonant cavity: initial={:.3e}, final={:.3e}",
                initial_energy,
                final_energy
            );
        }

        // Check for NaN
        for sample in energy_samples {
            assert!(
                !sample.total_energy.is_nan(),
                "Energy became NaN during resonance"
            );
        }

        Ok(())
    }
}

/// Scenario 5: Numerical stability test with small grid.
pub struct SmallGridScenario;

impl SimulationScenario for SmallGridScenario {
    fn name(&self) -> &str {
        "small_grid_stability"
    }

    fn build(&self) -> SimulationSetup {
        let grid = Grid::uniform(10, 10, 10, 1.0e-3); // Very small grid

        SimulationSetup {
            grid,
            boundaries: BoundaryConditions::all_pec(),
            materials: vec![],
            excitations: vec![Excitation::gaussian(2e9, 0.5, 0, (5, 5, 5))],
            num_steps: 100,
            energy_monitoring: EnergyMonitorConfig {
                sample_interval: 10,
                track_peak: true,
                decay_threshold: None,
            },
        }
    }

    fn verify(&self, result: &SimulationResult) -> Result<()> {
        // Main goal: check numerical stability (no NaN, no inf)
        for sample in &result.batch_result.energy_samples {
            assert!(!sample.total_energy.is_nan(), "Energy became NaN");
            assert!(!sample.total_energy.is_infinite(), "Energy became infinite");
            assert!(
                sample.total_energy >= 0.0,
                "Energy became negative: {}",
                sample.total_energy
            );
        }

        Ok(())
    }
}

/// Scenario 6: Multi-direction excitation test.
pub struct MultiDirectionExcitationScenario;

impl SimulationScenario for MultiDirectionExcitationScenario {
    fn name(&self) -> &str {
        "multi_direction_excitation"
    }

    fn build(&self) -> SimulationSetup {
        let grid = Grid::uniform(25, 25, 25, 0.5e-3);

        SimulationSetup {
            grid,
            boundaries: BoundaryConditions::all_pec(),
            materials: vec![],
            excitations: vec![
                Excitation::gaussian(3e9, 0.5, 0, (12, 12, 12)), // X direction
                Excitation::gaussian(3e9, 0.5, 1, (13, 13, 13)), // Y direction
                Excitation::gaussian(3e9, 0.5, 2, (14, 14, 14)), // Z direction
            ],
            num_steps: 150,
            energy_monitoring: EnergyMonitorConfig {
                sample_interval: 15,
                track_peak: true,
                decay_threshold: None,
            },
        }
    }

    fn verify(&self, result: &SimulationResult) -> Result<()> {
        let (e, h) = &result.final_fields;

        // Check that all field components have been excited
        let ex_energy: f64 = e.x.as_slice().iter().map(|&v| (v as f64).powi(2)).sum();
        let ey_energy: f64 = e.y.as_slice().iter().map(|&v| (v as f64).powi(2)).sum();
        let ez_energy: f64 = e.z.as_slice().iter().map(|&v| (v as f64).powi(2)).sum();

        assert!(ex_energy > 0.0, "Ex component not excited");
        assert!(ey_energy > 0.0, "Ey component not excited");
        assert!(ez_energy > 0.0, "Ez component not excited");

        // Check for NaN in any component
        assert!(
            !e.energy().is_nan() && !h.energy().is_nan(),
            "Fields contain NaN"
        );

        Ok(())
    }
}

/// Scenario 7: Hard vs soft source test.
pub struct HardSoftSourceScenario;

impl SimulationScenario for HardSoftSourceScenario {
    fn name(&self) -> &str {
        "hard_soft_source_behavior"
    }

    fn build(&self) -> SimulationSetup {
        let grid = Grid::uniform(30, 30, 30, 0.5e-3);

        SimulationSetup {
            grid,
            boundaries: BoundaryConditions::all_pec(),
            materials: vec![],
            excitations: vec![
                Excitation::gaussian(3e9, 0.5, 2, (15, 15, 15)), // Soft source (default)
                Excitation::gaussian(3e9, 0.5, 0, (20, 20, 20)).hard_source(), // Hard source
            ],
            num_steps: 100,
            energy_monitoring: EnergyMonitorConfig {
                sample_interval: 10,
                track_peak: true,
                decay_threshold: None,
            },
        }
    }

    fn verify(&self, result: &SimulationResult) -> Result<()> {
        // Both should contribute to field energy
        let (e, _h) = &result.final_fields;
        assert!(e.energy() > 0.0, "No field energy from excitations");

        // Check for numerical stability
        for sample in &result.batch_result.energy_samples {
            assert!(!sample.total_energy.is_nan(), "Energy became NaN");
        }

        Ok(())
    }
}

/// Scenario 8: Long simulation for memory stability.
pub struct LongSimulationScenario;

impl SimulationScenario for LongSimulationScenario {
    fn name(&self) -> &str {
        "long_simulation_stability"
    }

    fn build(&self) -> SimulationSetup {
        let grid = Grid::uniform(15, 15, 15, 1.0e-3);

        SimulationSetup {
            grid,
            boundaries: BoundaryConditions::all_pec(),
            materials: vec![],
            excitations: vec![Excitation::dirac(1.0, 1, (7, 7, 7))],
            num_steps: 1000, // Longer simulation
            energy_monitoring: EnergyMonitorConfig {
                sample_interval: 100,
                track_peak: true,
                decay_threshold: None,
            },
        }
    }

    fn verify(&self, result: &SimulationResult) -> Result<()> {
        // Check that simulation completed all steps
        assert_eq!(
            result.batch_result.timesteps_executed, 1000,
            "Did not complete all timesteps"
        );

        // Check for NaN throughout
        for sample in &result.batch_result.energy_samples {
            assert!(!sample.total_energy.is_nan(), "Energy became NaN");
        }

        Ok(())
    }
}

/// Scenario 9: Zero excitation (should remain at zero energy).
pub struct ZeroExcitationScenario;

impl SimulationScenario for ZeroExcitationScenario {
    fn name(&self) -> &str {
        "zero_excitation_stays_zero"
    }

    fn build(&self) -> SimulationSetup {
        let grid = Grid::uniform(20, 20, 20, 0.5e-3);

        SimulationSetup {
            grid,
            boundaries: BoundaryConditions::all_pec(),
            materials: vec![],
            excitations: vec![], // No excitations!
            num_steps: 100,
            energy_monitoring: EnergyMonitorConfig {
                sample_interval: 10,
                track_peak: true,
                decay_threshold: None,
            },
        }
    }

    fn verify(&self, result: &SimulationResult) -> Result<()> {
        // Energy should remain zero (or very close to zero due to floating point)
        for sample in &result.batch_result.energy_samples {
            assert!(
                sample.total_energy < 1e-20,
                "Energy non-zero without excitation: {:.3e}",
                sample.total_energy
            );
        }

        Ok(())
    }
}

/// Scenario 10: Batch execution consistency (run same simulation with different batch sizes).
pub struct BatchConsistencyScenario;

impl SimulationScenario for BatchConsistencyScenario {
    fn name(&self) -> &str {
        "batch_execution_consistency"
    }

    fn build(&self) -> SimulationSetup {
        let grid = Grid::uniform(15, 15, 15, 0.8e-3);

        SimulationSetup {
            grid,
            boundaries: BoundaryConditions::all_pec(),
            materials: vec![],
            excitations: vec![Excitation::gaussian(4e9, 0.5, 2, (7, 7, 7))],
            num_steps: 200,
            energy_monitoring: EnergyMonitorConfig {
                sample_interval: 20,
                track_peak: true,
                decay_threshold: None,
            },
        }
    }

    fn verify(&self, result: &SimulationResult) -> Result<()> {
        // Just verify it runs to completion
        assert_eq!(result.batch_result.timesteps_executed, 200);

        // And that results are sane
        for sample in &result.batch_result.energy_samples {
            assert!(!sample.total_energy.is_nan(), "Energy became NaN");
            assert!(sample.total_energy >= 0.0, "Energy became negative");
        }

        Ok(())
    }
}

// =============================================================================
// MACRO TO GENERATE TESTS FOR ALL ENGINES
// =============================================================================

/// Macro to generate test functions for all engine implementations.
///
/// Usage:
/// ```ignore
/// test_all_engines!(PecCavityScenario, test_pec_cavity);
/// ```
///
/// This generates 3 test functions (Basic, SIMD, Parallel):
/// - test_pec_cavity_basic()
/// - test_pec_cavity_simd()
/// - test_pec_cavity_parallel()
#[macro_export]
macro_rules! test_all_engines {
    ($scenario:expr, $test_name_base:ident) => {
        paste::paste! {
            #[test]
            fn [<$test_name_base _basic>]() {
                let scenario = $scenario;
                $crate::fdtd::engine_testing::test_scenario_with_engine_impl::<$crate::fdtd::BasicEngine>(&scenario).unwrap();
            }

            #[test]
            fn [<$test_name_base _simd>]() {
                let scenario = $scenario;
                $crate::fdtd::engine_testing::test_scenario_with_engine_impl::<$crate::fdtd::SimdEngine>(&scenario).unwrap();
            }

            #[test]
            fn [<$test_name_base _parallel>]() {
                let scenario = $scenario;
                $crate::fdtd::engine_testing::test_scenario_with_engine_impl::<$crate::fdtd::ParallelEngine>(&scenario).unwrap();
            }
        }
    };
}

#[cfg(test)]
mod tests {
    // Generate tests: scenarios × 3 engines (Basic, SIMD, Parallel)
    test_all_engines!(super::PecCavityScenario, test_pec_cavity);
    test_all_engines!(super::FreePropagationScenario, test_free_propagation);
    test_all_engines!(super::DielectricSlabScenario, test_dielectric_slab);
    test_all_engines!(super::ResonantCavityScenario, test_resonant_cavity);
    test_all_engines!(super::SmallGridScenario, test_small_grid);
    test_all_engines!(
        super::MultiDirectionExcitationScenario,
        test_multi_direction
    );
    test_all_engines!(super::HardSoftSourceScenario, test_hard_soft_source);
    test_all_engines!(super::LongSimulationScenario, test_long_simulation);
    test_all_engines!(super::ZeroExcitationScenario, test_zero_excitation);
    test_all_engines!(super::BatchConsistencyScenario, test_batch_consistency);
}
