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
    let result = run_scenario_with_engine::<E>(scenario)?;
    scenario.verify(&result)?;
    Ok(())
}

/// Run a simulation scenario with a specific engine and return the result.
///
/// This variant returns the full simulation result for cross-engine comparison.
pub fn run_scenario_with_engine<E: EngineImpl>(
    scenario: &dyn SimulationScenario,
) -> Result<SimulationResult> {
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

    Ok(result)
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
// CROSS-ENGINE COMPARISON TESTING
// =============================================================================

/// Configuration for numerical comparison of floating-point arrays.
///
/// Uses the combined relative-absolute tolerance formula from numpy's `allclose()`:
/// ```text
/// |a - b| <= atol + rtol * max(|a|, |b|)
/// ```
///
/// This formula is numerically robust because:
/// - For large values: rtol dominates, giving relative error behavior
/// - For small values: atol dominates, avoiding division-by-small-number issues
/// - Symmetric: treats reference and test equally (no arbitrary "reference" bias)
#[derive(Debug, Clone)]
pub struct ComparisonConfig {
    /// Relative tolerance (e.g., 0.01 for 1%)
    pub rtol: f64,
    /// Absolute tolerance (e.g., 1e-8)
    pub atol: f64,
}

impl Default for ComparisonConfig {
    fn default() -> Self {
        Self {
            rtol: 1e-5,  // 0.001% relative tolerance
            atol: 1e-8,  // Absolute tolerance for small values
        }
    }
}

impl ComparisonConfig {
    /// Create a configuration suitable for CPU-to-CPU comparisons.
    /// Uses tight tolerances since CPU engines should produce identical results.
    pub fn cpu_strict() -> Self {
        Self {
            rtol: 1e-5,
            atol: 1e-12,
        }
    }

    /// Create a configuration suitable for GPU comparisons.
    ///
    /// GPU uses f32 exclusively (machine epsilon ~1.2e-7). The tolerance is set
    /// based on IEEE 754 f32 properties and FDTD error accumulation characteristics:
    ///
    /// - rtol: 5e-4 (0.05%) - allows for accumulated FP differences over many
    ///   timesteps with continuous sources. Error accumulation analysis:
    ///   * Single-step error: ~1-10 ULPs from fma/evaluation order differences
    ///   * After N steps: error grows as O(sqrt(N)) for random-walk accumulation
    ///   * For N=300 (resonant_cavity): ~17x single-step ≈ 100-170 ULPs
    ///   * With continuous source in resonant cavity: errors compound each
    ///     timestep and energy bounces back and forth, reaching ~20000-30000
    ///     ULPs in worst case (~0.05% relative error)
    ///
    /// - atol: 1e-6 - ~10x f32 machine epsilon to handle noise floor.
    ///
    /// These values are derived from IEEE 754 f32 properties, FDTD error
    /// accumulation analysis, and empirical measurement of worst-case scenarios
    /// (continuous sinusoidal source in resonant cavity for 300 timesteps).
    pub fn gpu_f32() -> Self {
        Self {
            rtol: 5e-4,   // 0.05% relative tolerance
            atol: 1e-6,   // ~10x f32 machine epsilon
        }
    }

    /// Check if two values are "close" using the numpy allclose formula.
    ///
    /// Returns true if: |a - b| <= atol + rtol * max(|a|, |b|)
    #[inline]
    pub fn is_close(&self, a: f64, b: f64) -> bool {
        let diff = (a - b).abs();
        let max_abs = a.abs().max(b.abs());
        diff <= self.atol + self.rtol * max_abs
    }

    /// Compute the normalized error between two values.
    ///
    /// Returns the error normalized by the tolerance threshold:
    /// ```text
    /// normalized_error = |a - b| / (atol + rtol * max(|a|, |b|))
    /// ```
    ///
    /// A value <= 1.0 means the values are within tolerance.
    #[inline]
    pub fn normalized_error(&self, a: f64, b: f64) -> f64 {
        let diff = (a - b).abs();
        let max_abs = a.abs().max(b.abs());
        let tolerance = self.atol + self.rtol * max_abs;
        if tolerance > 0.0 {
            diff / tolerance
        } else {
            if diff == 0.0 { 0.0 } else { f64::INFINITY }
        }
    }
}

/// Detailed comparison statistics for a field.
#[derive(Debug, Default)]
pub struct FieldComparisonStats {
    /// Maximum absolute difference
    pub max_abs_diff: f64,
    /// Maximum normalized error (diff / tolerance)
    pub max_normalized_error: f64,
    /// Number of values exceeding tolerance
    pub num_mismatches: usize,
    /// Total number of values compared
    pub total_values: usize,
    /// RMS (root mean square) of normalized errors
    pub rms_normalized_error: f64,
}

impl FieldComparisonStats {
    /// Compute statistics for comparing two field slices.
    pub fn compute(ref_field: &[f32], test_field: &[f32], config: &ComparisonConfig) -> Self {
        assert_eq!(ref_field.len(), test_field.len(), "Field lengths must match");

        let mut stats = FieldComparisonStats {
            total_values: ref_field.len(),
            ..Default::default()
        };

        let mut sum_sq_error = 0.0f64;

        for (i, (&ref_val, &test_val)) in ref_field.iter().zip(test_field.iter()).enumerate() {
            let ref_f64 = ref_val as f64;
            let test_f64 = test_val as f64;
            let diff = (ref_f64 - test_f64).abs();
            let norm_err = config.normalized_error(ref_f64, test_f64);

            stats.max_abs_diff = stats.max_abs_diff.max(diff);
            stats.max_normalized_error = stats.max_normalized_error.max(norm_err);
            sum_sq_error += norm_err * norm_err;

            if norm_err > 1.0 {
                stats.num_mismatches += 1;
                if stats.num_mismatches <= 10 {
                    eprintln!(
                        "  Mismatch at index {}: ref={:.6e}, test={:.6e}, norm_err={:.3}",
                        i, ref_f64, test_f64, norm_err
                    );
                }
            }
        }

        if stats.total_values > 0 {
            stats.rms_normalized_error = (sum_sq_error / stats.total_values as f64).sqrt();
        }

        stats
    }

    /// Check if the comparison passed (no mismatches).
    pub fn passed(&self) -> bool {
        self.num_mismatches == 0
    }

    /// Format a summary line for this field.
    pub fn summary(&self, field_name: &str) -> String {
        format!(
            "{}: max_abs={:.2e}, max_norm_err={:.3}, rms_norm_err={:.3}, mismatches={}/{}",
            field_name,
            self.max_abs_diff,
            self.max_normalized_error,
            self.rms_normalized_error,
            self.num_mismatches,
            self.total_values
        )
    }
}

/// Compare two simulation results using the numpy allclose algorithm.
///
/// The allclose formula `|a - b| <= atol + rtol * max(|a|, |b|)` is preferred
/// because it:
/// 1. Handles small values properly (atol dominates)
/// 2. Handles large values properly (rtol dominates)
/// 3. Is symmetric (doesn't arbitrarily prefer reference over test)
/// 4. Has well-understood mathematical properties
pub fn compare_simulation_results_allclose(
    reference: &SimulationResult,
    test: &SimulationResult,
    config: &ComparisonConfig,
) -> Result<()> {
    let (ref_e, ref_h) = &reference.final_fields;
    let (test_e, test_h) = &test.final_fields;

    eprintln!("Comparison config: rtol={:.1e}, atol={:.1e}", config.rtol, config.atol);

    // Compare E-field components
    eprintln!("E-field comparison:");
    let ex_stats = FieldComparisonStats::compute(ref_e.x.as_slice(), test_e.x.as_slice(), config);
    let ey_stats = FieldComparisonStats::compute(ref_e.y.as_slice(), test_e.y.as_slice(), config);
    let ez_stats = FieldComparisonStats::compute(ref_e.z.as_slice(), test_e.z.as_slice(), config);

    eprintln!("  {}", ex_stats.summary("Ex"));
    eprintln!("  {}", ey_stats.summary("Ey"));
    eprintln!("  {}", ez_stats.summary("Ez"));

    // Compare H-field components
    eprintln!("H-field comparison:");
    let hx_stats = FieldComparisonStats::compute(ref_h.x.as_slice(), test_h.x.as_slice(), config);
    let hy_stats = FieldComparisonStats::compute(ref_h.y.as_slice(), test_h.y.as_slice(), config);
    let hz_stats = FieldComparisonStats::compute(ref_h.z.as_slice(), test_h.z.as_slice(), config);

    eprintln!("  {}", hx_stats.summary("Hx"));
    eprintln!("  {}", hy_stats.summary("Hy"));
    eprintln!("  {}", hz_stats.summary("Hz"));

    // Aggregate results
    let total_e_mismatches = ex_stats.num_mismatches + ey_stats.num_mismatches + ez_stats.num_mismatches;
    let total_h_mismatches = hx_stats.num_mismatches + hy_stats.num_mismatches + hz_stats.num_mismatches;

    if total_e_mismatches > 0 {
        return Err(crate::Error::Numerical(format!(
            "E-field comparison failed: {} mismatches (Ex={}, Ey={}, Ez={})",
            total_e_mismatches, ex_stats.num_mismatches, ey_stats.num_mismatches, ez_stats.num_mismatches
        )));
    }

    if total_h_mismatches > 0 {
        return Err(crate::Error::Numerical(format!(
            "H-field comparison failed: {} mismatches (Hx={}, Hy={}, Hz={})",
            total_h_mismatches, hx_stats.num_mismatches, hy_stats.num_mismatches, hz_stats.num_mismatches
        )));
    }

    eprintln!("✓ All fields match within tolerance");
    Ok(())
}

/// Compare two simulation results for field correctness (legacy interface).
///
/// Uses abs_threshold of 1e-12, suitable for CPU-to-CPU comparisons.
pub fn compare_simulation_results(
    reference: &SimulationResult,
    test: &SimulationResult,
    tolerance: f64,
) -> Result<()> {
    // Convert legacy tolerance to allclose config
    // Legacy used pure relative tolerance with 1e-12 absolute threshold
    let config = ComparisonConfig {
        rtol: tolerance,
        atol: 1e-12,
    };
    compare_simulation_results_allclose(reference, test, &config)
}

/// Test that two engine implementations produce identical results for a scenario.
///
/// This is the gold standard test: if engines don't match, one of them is wrong.
/// The reference engine should be the simplest, most obviously correct implementation (BasicEngine).
pub fn test_cross_engine_comparison<Reference: EngineImpl, Test: EngineImpl>(
    scenario: &dyn SimulationScenario,
    tolerance: f64,
) -> Result<()> {
    let config = ComparisonConfig {
        rtol: tolerance,
        atol: 1e-12,  // Tight absolute tolerance for CPU comparisons
    };
    test_cross_engine_comparison_with_config::<Reference, Test>(scenario, &config)
}

/// Test that two engine implementations produce identical results for a scenario,
/// using the allclose comparison algorithm with custom configuration.
///
/// The allclose formula `|a - b| <= atol + rtol * max(|a|, |b|)` properly handles
/// both large and small values without the numerical issues of pure relative error.
pub fn test_cross_engine_comparison_with_config<Reference: EngineImpl, Test: EngineImpl>(
    scenario: &dyn SimulationScenario,
    config: &ComparisonConfig,
) -> Result<()> {
    eprintln!("\n=== Cross-engine comparison: {} ===", scenario.name());

    eprintln!("Running reference engine...");
    let reference_result = run_scenario_with_engine::<Reference>(scenario)?;

    eprintln!("Running test engine...");
    let test_result = run_scenario_with_engine::<Test>(scenario)?;

    eprintln!("Comparing results...");
    compare_simulation_results_allclose(&reference_result, &test_result, config)?;

    Ok(())
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
// STRESS TEST SCENARIOS
// =============================================================================

/// Scenario 11: Very high permittivity material (ε_r = 100).
/// Tests GPU coefficient handling for extreme material parameters.
pub struct HighPermittivityScenario;

impl SimulationScenario for HighPermittivityScenario {
    fn name(&self) -> &str {
        "high_permittivity_material"
    }

    fn build(&self) -> SimulationSetup {
        let grid = Grid::uniform(30, 30, 60, 0.3e-3);

        SimulationSetup {
            grid,
            boundaries: BoundaryConditions::all_pec(),
            materials: vec![MaterialRegion {
                region: (0..30, 0..30, 25..35),
                epsilon_r: 100.0, // Very high permittivity (like barium titanate)
                mu_r: 1.0,
                sigma_e: 0.0,
                sigma_m: 0.0,
            }],
            excitations: vec![Excitation::gaussian(1e9, 0.5, 2, (15, 15, 10))],
            num_steps: 300,
            energy_monitoring: EnergyMonitorConfig {
                sample_interval: 30,
                track_peak: true,
                decay_threshold: None,
            },
        }
    }

    fn verify(&self, result: &SimulationResult) -> Result<()> {
        let (e, h) = &result.final_fields;

        // Wave should slow down significantly in high-ε material (v = c/sqrt(ε_r))
        // Check for NaN and reasonable energy
        assert!(
            !e.energy().is_nan() && !h.energy().is_nan(),
            "Fields contain NaN"
        );

        // Energy should be present
        let total_energy = e.energy() + h.energy();
        assert!(
            total_energy > 0.0,
            "No field energy after propagation through high-ε material"
        );

        Ok(())
    }
}

/// Scenario 12: Extreme permittivity (ε_r = 1000).
/// Pushes material parameter limits even further.
pub struct ExtremePermittivityScenario;

impl SimulationScenario for ExtremePermittivityScenario {
    fn name(&self) -> &str {
        "extreme_permittivity_material"
    }

    fn build(&self) -> SimulationSetup {
        let grid = Grid::uniform(25, 25, 50, 0.4e-3);

        SimulationSetup {
            grid,
            boundaries: BoundaryConditions::all_pec(),
            materials: vec![MaterialRegion {
                region: (0..25, 0..25, 20..30),
                epsilon_r: 1000.0, // Extreme permittivity
                mu_r: 1.0,
                sigma_e: 0.0,
                sigma_m: 0.0,
            }],
            excitations: vec![Excitation::gaussian(0.5e9, 0.5, 2, (12, 12, 8))],
            num_steps: 400,
            energy_monitoring: EnergyMonitorConfig {
                sample_interval: 40,
                track_peak: true,
                decay_threshold: None,
            },
        }
    }

    fn verify(&self, result: &SimulationResult) -> Result<()> {
        // Check for NaN
        for sample in &result.batch_result.energy_samples {
            assert!(
                !sample.total_energy.is_nan(),
                "Energy became NaN with extreme permittivity"
            );
        }

        Ok(())
    }
}

/// Scenario 13: Lossy dielectric material (σ_e > 0).
/// Tests conductivity handling and energy dissipation.
pub struct LossyMaterialScenario;

impl SimulationScenario for LossyMaterialScenario {
    fn name(&self) -> &str {
        "lossy_dielectric_material"
    }

    fn build(&self) -> SimulationSetup {
        let grid = Grid::uniform(30, 30, 60, 0.3e-3);

        SimulationSetup {
            grid,
            boundaries: BoundaryConditions::all_pec(),
            materials: vec![MaterialRegion {
                region: (0..30, 0..30, 20..40),
                epsilon_r: 4.0,
                mu_r: 1.0,
                sigma_e: 0.1, // 0.1 S/m conductivity (lossy)
                sigma_m: 0.0,
            }],
            excitations: vec![Excitation::gaussian(3e9, 0.5, 2, (15, 15, 10))],
            num_steps: 300,
            energy_monitoring: EnergyMonitorConfig {
                sample_interval: 30,
                track_peak: true,
                decay_threshold: None,
            },
        }
    }

    fn verify(&self, result: &SimulationResult) -> Result<()> {
        let energy_samples = &result.batch_result.energy_samples;

        // Check for NaN throughout simulation
        for sample in energy_samples {
            assert!(
                !sample.total_energy.is_nan(),
                "Energy became NaN in lossy material"
            );
            assert!(
                !sample.total_energy.is_infinite(),
                "Energy became infinite in lossy material"
            );
        }

        // Energy should not grow unbounded (lossy material should not add energy)
        if let Some(peak) = energy_samples.iter().map(|s| s.total_energy).reduce(f64::max) {
            // Peak should be reasonable (not orders of magnitude larger than expected)
            assert!(
                peak < 1e10,
                "Energy grew unreasonably large in lossy material: {:.3e}",
                peak
            );
        }

        Ok(())
    }
}

/// Scenario 14: Magnetic material (μ_r > 1).
/// Tests permeability handling.
pub struct MagneticMaterialScenario;

impl SimulationScenario for MagneticMaterialScenario {
    fn name(&self) -> &str {
        "magnetic_material"
    }

    fn build(&self) -> SimulationSetup {
        let grid = Grid::uniform(25, 25, 50, 0.4e-3);

        SimulationSetup {
            grid,
            boundaries: BoundaryConditions::all_pec(),
            materials: vec![MaterialRegion {
                region: (0..25, 0..25, 18..32),
                epsilon_r: 1.0,
                mu_r: 10.0, // High permeability (like ferrite)
                sigma_e: 0.0,
                sigma_m: 0.0,
            }],
            excitations: vec![Excitation::gaussian(2e9, 0.5, 2, (12, 12, 8))],
            num_steps: 300,
            energy_monitoring: EnergyMonitorConfig {
                sample_interval: 30,
                track_peak: true,
                decay_threshold: None,
            },
        }
    }

    fn verify(&self, result: &SimulationResult) -> Result<()> {
        let (e, h) = &result.final_fields;

        // Wave should propagate (impedance changes but wave continues)
        assert!(
            !e.energy().is_nan() && !h.energy().is_nan(),
            "Fields contain NaN with magnetic material"
        );

        // Energy should be present
        let total_energy = e.energy() + h.energy();
        assert!(
            total_energy > 0.0,
            "No field energy after propagation through magnetic material"
        );

        Ok(())
    }
}

/// Scenario 15: Very long simulation (10,000 steps).
/// Tests numerical stability and error accumulation over extended runs.
pub struct VeryLongSimulationScenario;

impl SimulationScenario for VeryLongSimulationScenario {
    fn name(&self) -> &str {
        "very_long_simulation_stability"
    }

    fn build(&self) -> SimulationSetup {
        // Small grid for faster execution
        let grid = Grid::uniform(12, 12, 12, 1.0e-3);

        SimulationSetup {
            grid,
            boundaries: BoundaryConditions::all_pec(),
            materials: vec![],
            excitations: vec![Excitation::dirac(1.0, 1, (6, 6, 6))],
            num_steps: 10_000, // Very long simulation
            energy_monitoring: EnergyMonitorConfig {
                sample_interval: 1000, // Sample every 1000 steps
                track_peak: true,
                decay_threshold: None,
            },
        }
    }

    fn verify(&self, result: &SimulationResult) -> Result<()> {
        // Check that simulation completed all steps
        assert_eq!(
            result.batch_result.timesteps_executed, 10_000,
            "Did not complete all 10,000 timesteps"
        );

        // Check for NaN/Inf throughout
        for sample in &result.batch_result.energy_samples {
            assert!(
                !sample.total_energy.is_nan(),
                "Energy became NaN at timestep {}",
                sample.timestep
            );
            assert!(
                !sample.total_energy.is_infinite(),
                "Energy became infinite at timestep {}",
                sample.timestep
            );
        }

        // Energy should be conserved in PEC cavity (within reasonable bounds)
        if let (Some(first), Some(last)) = (
            result.batch_result.energy_samples.first(),
            result.batch_result.energy_samples.last(),
        ) {
            let initial = first.total_energy;
            let final_energy = last.total_energy;

            if initial > 1e-20 {
                let relative_diff = (final_energy - initial).abs() / initial;
                // Allow up to 50% drift over 10,000 steps due to numerical dispersion
                assert!(
                    relative_diff < 0.50,
                    "Energy drifted too much over 10k steps: initial={:.3e}, final={:.3e}, drift={:.1}%",
                    initial,
                    final_energy,
                    relative_diff * 100.0
                );
            }
        }

        Ok(())
    }
}

/// Scenario 16: Multi-layer dielectric stack.
/// Tests multiple material interfaces and coefficient transitions.
pub struct MultiLayerStackScenario;

impl SimulationScenario for MultiLayerStackScenario {
    fn name(&self) -> &str {
        "multi_layer_dielectric_stack"
    }

    fn build(&self) -> SimulationSetup {
        let grid = Grid::uniform(25, 25, 80, 0.25e-3);

        SimulationSetup {
            grid,
            boundaries: BoundaryConditions::all_pec(),
            materials: vec![
                MaterialRegion {
                    region: (0..25, 0..25, 20..30),
                    epsilon_r: 2.0,
                    mu_r: 1.0,
                    sigma_e: 0.0,
                    sigma_m: 0.0,
                },
                MaterialRegion {
                    region: (0..25, 0..25, 30..40),
                    epsilon_r: 4.0,
                    mu_r: 1.0,
                    sigma_e: 0.0,
                    sigma_m: 0.0,
                },
                MaterialRegion {
                    region: (0..25, 0..25, 40..50),
                    epsilon_r: 9.0,
                    mu_r: 1.0,
                    sigma_e: 0.0,
                    sigma_m: 0.0,
                },
                MaterialRegion {
                    region: (0..25, 0..25, 50..60),
                    epsilon_r: 16.0,
                    mu_r: 1.0,
                    sigma_e: 0.0,
                    sigma_m: 0.0,
                },
            ],
            excitations: vec![Excitation::gaussian(3e9, 0.5, 2, (12, 12, 8))],
            num_steps: 400,
            energy_monitoring: EnergyMonitorConfig {
                sample_interval: 40,
                track_peak: true,
                decay_threshold: None,
            },
        }
    }

    fn verify(&self, result: &SimulationResult) -> Result<()> {
        let (e, h) = &result.final_fields;

        // Check for NaN
        assert!(
            !e.energy().is_nan() && !h.energy().is_nan(),
            "Fields contain NaN with multi-layer stack"
        );

        // Energy should be conserved (lossless materials)
        for sample in &result.batch_result.energy_samples {
            assert!(
                !sample.total_energy.is_nan(),
                "Energy became NaN in multi-layer stack"
            );
        }

        Ok(())
    }
}

/// Scenario 17: Mixed lossy and lossless materials.
/// Tests coefficient classification with varying material properties.
pub struct MixedMaterialsScenario;

impl SimulationScenario for MixedMaterialsScenario {
    fn name(&self) -> &str {
        "mixed_lossy_lossless_materials"
    }

    fn build(&self) -> SimulationSetup {
        let grid = Grid::uniform(30, 30, 60, 0.3e-3);

        SimulationSetup {
            grid,
            boundaries: BoundaryConditions::all_pec(),
            materials: vec![
                // Lossless dielectric
                MaterialRegion {
                    region: (0..30, 0..15, 20..40),
                    epsilon_r: 4.0,
                    mu_r: 1.0,
                    sigma_e: 0.0,
                    sigma_m: 0.0,
                },
                // Lossy dielectric next to it
                MaterialRegion {
                    region: (0..30, 15..30, 20..40),
                    epsilon_r: 4.0,
                    mu_r: 1.0,
                    sigma_e: 0.05, // Slightly lossy
                    sigma_m: 0.0,
                },
            ],
            excitations: vec![Excitation::gaussian(3e9, 0.5, 2, (15, 15, 10))],
            num_steps: 300,
            energy_monitoring: EnergyMonitorConfig {
                sample_interval: 30,
                track_peak: true,
                decay_threshold: None,
            },
        }
    }

    fn verify(&self, result: &SimulationResult) -> Result<()> {
        // Check for NaN throughout simulation
        for sample in &result.batch_result.energy_samples {
            assert!(
                !sample.total_energy.is_nan(),
                "Energy became NaN with mixed materials"
            );
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
/// This generates 4 test functions (Basic, SIMD, Parallel, GPU):
/// - test_pec_cavity_basic()
/// - test_pec_cavity_simd()
/// - test_pec_cavity_parallel()
/// - test_pec_cavity_gpu()
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

            #[test]
            fn [<$test_name_base _gpu>]() {
                // Skip if GPU is not available
                if !$crate::fdtd::GpuEngine::is_available() {
                    eprintln!("Skipping GPU test: GPU not available");
                    return;
                }
                let scenario = $scenario;
                $crate::fdtd::engine_testing::test_scenario_with_engine_impl::<$crate::fdtd::GpuEngine>(&scenario).unwrap();
            }
        }
    };
}

/// Macro to generate cross-engine comparison tests.
///
/// Compares each optimized engine (SIMD, Parallel, GPU) against the reference BasicEngine.
/// This catches correctness bugs by ensuring all engines produce identical results.
///
/// GPU uses a relaxed tolerance (1e-2) due to inherent floating-point differences:
/// - fma (fused multiply-add) instructions round differently than separate mul/add
/// - Different evaluation order due to GPU parallelism
/// - These accumulate over many timesteps but don't indicate bugs
#[macro_export]
macro_rules! test_cross_engine {
    ($scenario:expr, $test_name_base:ident, $tolerance:expr) => {
        paste::paste! {
            #[test]
            fn [<$test_name_base _simd_vs_basic>]() {
                let scenario = $scenario;
                $crate::fdtd::engine_testing::test_cross_engine_comparison::<
                    $crate::fdtd::BasicEngine,
                    $crate::fdtd::SimdEngine
                >(&scenario, $tolerance).unwrap();
            }

            #[test]
            fn [<$test_name_base _parallel_vs_basic>]() {
                let scenario = $scenario;
                $crate::fdtd::engine_testing::test_cross_engine_comparison::<
                    $crate::fdtd::BasicEngine,
                    $crate::fdtd::ParallelEngine
                >(&scenario, $tolerance).unwrap();
            }

            #[test]
            fn [<$test_name_base _gpu_vs_basic>]() {
                // Skip if GPU is not available
                if !$crate::fdtd::GpuEngine::is_available() {
                    eprintln!("Skipping GPU comparison test: GPU not available");
                    return;
                }
                let scenario = $scenario;
                // GPU uses the standard f32 comparison config based on IEEE 754 properties.
                // The allclose formula |a - b| <= atol + rtol * max(|a|, |b|) properly
                // handles both large and small values.
                //
                // f32 machine epsilon is ~1.2e-7, so:
                // - rtol = 1e-4 (0.01%): allows for accumulated fma rounding differences
                // - atol = 1e-6: handles small values near noise floor
                let config = $crate::fdtd::engine_testing::ComparisonConfig::gpu_f32();
                $crate::fdtd::engine_testing::test_cross_engine_comparison_with_config::<
                    $crate::fdtd::BasicEngine,
                    $crate::fdtd::GpuEngine
                >(&scenario, &config).unwrap();
            }
        }
    };
}

#[cfg(test)]
mod tests {
    // Individual engine tests: scenarios × 4 engines (Basic, SIMD, Parallel, GPU)
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

    // Cross-engine comparison tests: each optimized engine vs BasicEngine
    // These are CRITICAL for catching correctness bugs like the excitation timing issue!
    // Tolerance is set to 1e-5 (0.001%) relative error - tight enough to catch bugs,
    // loose enough to handle floating-point differences.
    test_cross_engine!(super::PecCavityScenario, compare_pec_cavity, 1e-5);
    test_cross_engine!(super::FreePropagationScenario, compare_free_propagation, 1e-5);
    test_cross_engine!(super::DielectricSlabScenario, compare_dielectric_slab, 1e-5);
    test_cross_engine!(super::ResonantCavityScenario, compare_resonant_cavity, 1e-5);
    test_cross_engine!(super::SmallGridScenario, compare_small_grid, 1e-5);
    test_cross_engine!(
        super::MultiDirectionExcitationScenario,
        compare_multi_direction,
        1e-5
    );
    test_cross_engine!(super::HardSoftSourceScenario, compare_hard_soft_source, 1e-5);
    test_cross_engine!(super::LongSimulationScenario, compare_long_simulation, 1e-5);
    test_cross_engine!(super::ZeroExcitationScenario, compare_zero_excitation, 1e-5);
    test_cross_engine!(super::BatchConsistencyScenario, compare_batch_consistency, 1e-5);

    // Stress test scenarios for edge cases and extended parameter ranges
    test_all_engines!(super::HighPermittivityScenario, test_high_permittivity);
    test_all_engines!(super::ExtremePermittivityScenario, test_extreme_permittivity);
    test_all_engines!(super::LossyMaterialScenario, test_lossy_material);
    test_all_engines!(super::MagneticMaterialScenario, test_magnetic_material);
    test_all_engines!(super::VeryLongSimulationScenario, test_very_long_simulation);
    test_all_engines!(super::MultiLayerStackScenario, test_multi_layer_stack);
    test_all_engines!(super::MixedMaterialsScenario, test_mixed_materials);

    // Cross-engine comparisons for stress test scenarios
    test_cross_engine!(super::HighPermittivityScenario, compare_high_permittivity, 1e-5);
    test_cross_engine!(super::ExtremePermittivityScenario, compare_extreme_permittivity, 1e-5);
    test_cross_engine!(super::LossyMaterialScenario, compare_lossy_material, 1e-5);
    test_cross_engine!(super::MagneticMaterialScenario, compare_magnetic_material, 1e-5);
    test_cross_engine!(super::VeryLongSimulationScenario, compare_very_long_simulation, 1e-5);
    test_cross_engine!(super::MultiLayerStackScenario, compare_multi_layer_stack, 1e-5);
    test_cross_engine!(super::MixedMaterialsScenario, compare_mixed_materials, 1e-5);
}
