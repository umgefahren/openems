//! Holistic benchmarks for realistic FDTD simulations.
//!
//! These benchmarks test complete simulation scenarios across all engine types
//! to provide meaningful performance comparisons.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use openems::arrays::Dimensions;
use openems::extensions::{Extension, PmlBoundaries, Upml};
use openems::fdtd::{
    BoundaryConditions, EnergyMonitorConfig, Engine, EngineBatch, EngineType, Excitation, Operator,
    TerminationConfig,
};
use openems::geometry::Grid;

/// NoOp extension for baseline benchmarks.
#[derive(Debug)]
struct NoOpExtension;

impl Extension for NoOpExtension {
    fn name(&self) -> &str {
        "noop"
    }

    fn apply_step<E>(&mut self, _engine: &mut E, _step: u64) -> openems::Result<()>
    where
        E: openems::fdtd::EngineImpl,
    {
        Ok(())
    }
}

/// PML extension wrapper that implements the Extension trait.
struct PmlExtension {
    pml: Upml,
}

impl std::fmt::Debug for PmlExtension {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PmlExtension").finish()
    }
}

impl PmlExtension {
    fn new(dims: Dimensions, delta: [f64; 3], dt: f64, layers: usize) -> Self {
        let boundaries = PmlBoundaries::uniform(layers);
        Self {
            pml: Upml::new(dims, delta, dt, boundaries),
        }
    }
}

impl Extension for PmlExtension {
    fn name(&self) -> &str {
        "pml"
    }

    fn apply_step<E>(&mut self, engine: &mut E, _step: u64) -> openems::Result<()>
    where
        E: openems::fdtd::EngineImpl,
    {
        let (e_field, h_field) = engine.write_fields();

        // Pre-current update
        self.pml.pre_current_update(h_field);
        // (H update happens in engine)

        // Post-current update
        self.pml.post_current_update(h_field);

        // Pre-voltage update
        self.pml.pre_voltage_update(e_field);
        // (E update happens in engine)

        // Post-voltage update
        self.pml.post_voltage_update(e_field);

        Ok(())
    }
}

/// Scenario 1: Waveguide Propagation (baseline)
/// 200x50x50 grid, 1000 steps, Gaussian pulse, PEC boundaries
fn scenario_waveguide(
    engine_type: EngineType,
    num_steps: u64,
) -> (Engine, EngineBatch<NoOpExtension>) {
    let grid = Grid::uniform(200, 50, 50, 1e-3);
    let op = Operator::new(grid.clone(), BoundaryConditions::all_pec()).unwrap();
    let dt = op.timestep();

    let engine = Engine::new(&op, engine_type).unwrap();

    // Gaussian pulse excitation at input face
    let excitation = Excitation::gaussian(5e9, 0.5, 2, (5, 25, 25));
    let scheduled =
        openems::fdtd::batch::ScheduledExcitation::from_excitation(&excitation, 0, num_steps, dt);

    let batch = EngineBatch {
        num_steps: Some(num_steps),
        excitations: vec![scheduled],
        extensions: vec![NoOpExtension],
        termination: TerminationConfig::default(),
        energy_monitoring: EnergyMonitorConfig::default(),
    };

    (engine, batch)
}

/// Scenario 2: Resonant Cavity with PML
/// 100x100x100 grid, configurable steps, 8-layer PML, sinusoidal excitation
fn scenario_cavity_pml(
    engine_type: EngineType,
    num_steps: u64,
    pml_layers: usize,
) -> (Engine, EngineBatch<PmlExtension>) {
    let nx = 100 + 2 * pml_layers;
    let ny = 100 + 2 * pml_layers;
    let nz = 100 + 2 * pml_layers;
    let delta = 1e-3;

    let grid = Grid::uniform(nx, ny, nz, delta);
    let op = Operator::new(grid.clone(), BoundaryConditions::all_pec()).unwrap();
    let dt = op.timestep();
    let dims = op.dimensions();

    let engine = Engine::new(&op, engine_type).unwrap();

    // Sinusoidal excitation at center
    let center = (nx / 2, ny / 2, nz / 2);
    let excitation = Excitation::sinusoidal(1e9, 2, center);
    let scheduled =
        openems::fdtd::batch::ScheduledExcitation::from_excitation(&excitation, 0, num_steps, dt);

    // PML extension
    let pml_ext = PmlExtension::new(dims, [delta, delta, delta], dt, pml_layers);

    let batch = EngineBatch {
        num_steps: Some(num_steps),
        excitations: vec![scheduled],
        extensions: vec![pml_ext],
        termination: TerminationConfig::default(),
        energy_monitoring: EnergyMonitorConfig::default(),
    };

    (engine, batch)
}

/// Scenario 3: Large Domain (stress test)
/// 200x200x200 grid, configurable steps
fn scenario_large_domain(
    engine_type: EngineType,
    num_steps: u64,
) -> (Engine, EngineBatch<NoOpExtension>) {
    let grid = Grid::uniform(200, 200, 200, 1e-3);
    let op = Operator::new(grid.clone(), BoundaryConditions::all_pec()).unwrap();
    let dt = op.timestep();

    let engine = Engine::new(&op, engine_type).unwrap();

    // Central Gaussian pulse
    let excitation = Excitation::gaussian(5e9, 0.5, 2, (100, 100, 100));
    let scheduled =
        openems::fdtd::batch::ScheduledExcitation::from_excitation(&excitation, 0, num_steps, dt);

    let batch = EngineBatch {
        num_steps: Some(num_steps),
        excitations: vec![scheduled],
        extensions: vec![NoOpExtension],
        termination: TerminationConfig::default(),
        energy_monitoring: EnergyMonitorConfig::default(),
    };

    (engine, batch)
}

/// Scenario 4: Multiple Excitations
/// Tests excitation upload overhead
fn scenario_multi_excitation(
    engine_type: EngineType,
    num_steps: u64,
    num_excitations: usize,
) -> (Engine, EngineBatch<NoOpExtension>) {
    let grid = Grid::uniform(100, 100, 100, 1e-3);
    let op = Operator::new(grid.clone(), BoundaryConditions::all_pec()).unwrap();
    let dt = op.timestep();

    let engine = Engine::new(&op, engine_type).unwrap();

    // Create multiple excitations spread across the grid
    let mut scheduled_excitations = Vec::with_capacity(num_excitations);
    for i in 0..num_excitations {
        let x = 10 + (i * 80 / num_excitations) % 80;
        let y = 10 + ((i * 3) * 80 / num_excitations) % 80;
        let z = 10 + ((i * 7) * 80 / num_excitations) % 80;

        let excitation = Excitation::gaussian(5e9, 0.5, i % 3, (x, y, z));
        scheduled_excitations.push(openems::fdtd::batch::ScheduledExcitation::from_excitation(
            &excitation,
            0,
            num_steps,
            dt,
        ));
    }

    let batch = EngineBatch {
        num_steps: Some(num_steps),
        excitations: scheduled_excitations,
        extensions: vec![NoOpExtension],
        termination: TerminationConfig::default(),
        energy_monitoring: EnergyMonitorConfig::default(),
    };

    (engine, batch)
}

/// Scenario 5: Energy Monitoring
/// Tests GPU energy reduction overhead
fn scenario_energy_monitoring(
    engine_type: EngineType,
    num_steps: u64,
    sample_interval: u64,
) -> (Engine, EngineBatch<NoOpExtension>) {
    let grid = Grid::uniform(100, 100, 100, 1e-3);
    let op = Operator::new(grid.clone(), BoundaryConditions::all_pec()).unwrap();
    let dt = op.timestep();

    let engine = Engine::new(&op, engine_type).unwrap();

    let excitation = Excitation::gaussian(5e9, 0.5, 2, (50, 50, 50));
    let scheduled =
        openems::fdtd::batch::ScheduledExcitation::from_excitation(&excitation, 0, num_steps, dt);

    let batch = EngineBatch {
        num_steps: Some(num_steps),
        excitations: vec![scheduled],
        extensions: vec![NoOpExtension],
        termination: TerminationConfig::default(),
        energy_monitoring: EnergyMonitorConfig {
            sample_interval,
            track_peak: true,
            decay_threshold: None,
        },
    };

    (engine, batch)
}

/// Benchmark group: Waveguide Propagation
fn bench_waveguide(c: &mut Criterion) {
    let mut group = c.benchmark_group("waveguide_200x50x50");
    let total_cells = 200 * 50 * 50;
    let num_steps = 100u64;

    group.throughput(Throughput::Elements(total_cells * num_steps));
    group.sample_size(10);

    for engine_type in [EngineType::Basic, EngineType::Simd, EngineType::Parallel] {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{:?}", engine_type)),
            &engine_type,
            |b, &et| {
                b.iter_batched(
                    || scenario_waveguide(et, num_steps),
                    |(mut engine, batch)| {
                        let result = engine.run_batch(batch).unwrap();
                        black_box(result)
                    },
                    criterion::BatchSize::LargeInput,
                );
            },
        );
    }

    // GPU benchmark (if available)
    if openems::fdtd::GpuEngine::is_available() {
        group.bench_with_input(
            BenchmarkId::from_parameter("Gpu"),
            &EngineType::Gpu,
            |b, &et| {
                b.iter_batched(
                    || scenario_waveguide(et, num_steps),
                    |(mut engine, batch)| {
                        let result = engine.run_batch(batch).unwrap();
                        black_box(result)
                    },
                    criterion::BatchSize::LargeInput,
                );
            },
        );
    }

    group.finish();
}

/// Benchmark group: Large Domain Performance
fn bench_large_domain(c: &mut Criterion) {
    let mut group = c.benchmark_group("large_domain_200x200x200");
    let total_cells = 200 * 200 * 200;
    let num_steps = 10u64; // Fewer steps for large domain

    group.throughput(Throughput::Elements(total_cells * num_steps));
    group.sample_size(10);

    for engine_type in [EngineType::Simd, EngineType::Parallel] {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{:?}", engine_type)),
            &engine_type,
            |b, &et| {
                b.iter_batched(
                    || scenario_large_domain(et, num_steps),
                    |(mut engine, batch)| {
                        let result = engine.run_batch(batch).unwrap();
                        black_box(result)
                    },
                    criterion::BatchSize::LargeInput,
                );
            },
        );
    }

    // GPU benchmark
    if openems::fdtd::GpuEngine::is_available() {
        group.bench_with_input(
            BenchmarkId::from_parameter("Gpu"),
            &EngineType::Gpu,
            |b, &et| {
                b.iter_batched(
                    || scenario_large_domain(et, num_steps),
                    |(mut engine, batch)| {
                        let result = engine.run_batch(batch).unwrap();
                        black_box(result)
                    },
                    criterion::BatchSize::LargeInput,
                );
            },
        );
    }

    group.finish();
}

/// Benchmark group: Excitation Scaling
fn bench_excitation_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("excitation_scaling");
    let total_cells = 100 * 100 * 100;
    let num_steps = 100u64;

    group.sample_size(10);

    for num_excitations in [1, 10, 100, 1000] {
        group.throughput(Throughput::Elements(total_cells * num_steps));

        // Test Parallel engine
        group.bench_with_input(
            BenchmarkId::new("Parallel", num_excitations),
            &num_excitations,
            |b, &n| {
                b.iter_batched(
                    || scenario_multi_excitation(EngineType::Parallel, num_steps, n),
                    |(mut engine, batch)| {
                        let result = engine.run_batch(batch).unwrap();
                        black_box(result)
                    },
                    criterion::BatchSize::LargeInput,
                );
            },
        );

        // Test GPU engine
        if openems::fdtd::GpuEngine::is_available() {
            group.bench_with_input(
                BenchmarkId::new("Gpu", num_excitations),
                &num_excitations,
                |b, &n| {
                    b.iter_batched(
                        || scenario_multi_excitation(EngineType::Gpu, num_steps, n),
                        |(mut engine, batch)| {
                            let result = engine.run_batch(batch).unwrap();
                            black_box(result)
                        },
                        criterion::BatchSize::LargeInput,
                    );
                },
            );
        }
    }

    group.finish();
}

/// Benchmark group: Energy Monitoring Overhead
fn bench_energy_monitoring(c: &mut Criterion) {
    let mut group = c.benchmark_group("energy_monitoring");
    let total_cells = 100 * 100 * 100;
    let num_steps = 1000u64;

    group.sample_size(10);

    // Baseline: no energy monitoring
    group.throughput(Throughput::Elements(total_cells * num_steps));

    for (label, sample_interval) in [("none", 0u64), ("every_100", 100), ("every_10", 10)] {
        // Parallel engine
        group.bench_with_input(
            BenchmarkId::new("Parallel", label),
            &sample_interval,
            |b, &interval| {
                b.iter_batched(
                    || scenario_energy_monitoring(EngineType::Parallel, num_steps, interval),
                    |(mut engine, batch)| {
                        let result = engine.run_batch(batch).unwrap();
                        black_box(result)
                    },
                    criterion::BatchSize::LargeInput,
                );
            },
        );

        // GPU engine
        if openems::fdtd::GpuEngine::is_available() {
            group.bench_with_input(
                BenchmarkId::new("Gpu", label),
                &sample_interval,
                |b, &interval| {
                    b.iter_batched(
                        || scenario_energy_monitoring(EngineType::Gpu, num_steps, interval),
                        |(mut engine, batch)| {
                            let result = engine.run_batch(batch).unwrap();
                            black_box(result)
                        },
                        criterion::BatchSize::LargeInput,
                    );
                },
            );
        }
    }

    group.finish();
}

/// Benchmark group: PML Layer Scaling
/// Tests overhead of PML with different layer counts (4, 8, 12, 16 layers)
fn bench_pml_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("pml_layer_scaling");
    let num_steps = 100u64;

    group.sample_size(10);

    for pml_layers in [4usize, 8, 12, 16] {
        // Grid size adjusted for PML layers
        let base_size = 100usize;
        let total_size = base_size + 2 * pml_layers;
        let total_cells = (total_size * total_size * total_size) as u64;

        group.throughput(Throughput::Elements(total_cells * num_steps));

        // Parallel engine benchmark
        group.bench_with_input(
            BenchmarkId::new("Parallel", pml_layers),
            &pml_layers,
            |b, &layers| {
                b.iter_batched(
                    || scenario_cavity_pml(EngineType::Parallel, num_steps, layers),
                    |(mut engine, batch)| {
                        let result = engine.run_batch(batch).unwrap();
                        black_box(result)
                    },
                    criterion::BatchSize::LargeInput,
                );
            },
        );

        // GPU engine benchmark
        if openems::fdtd::GpuEngine::is_available() {
            group.bench_with_input(
                BenchmarkId::new("Gpu", pml_layers),
                &pml_layers,
                |b, &layers| {
                    b.iter_batched(
                        || scenario_cavity_pml(EngineType::Gpu, num_steps, layers),
                        |(mut engine, batch)| {
                            let result = engine.run_batch(batch).unwrap();
                            black_box(result)
                        },
                        criterion::BatchSize::LargeInput,
                    );
                },
            );
        }
    }

    group.finish();
}

/// Benchmark group: Batch Size Optimization
fn bench_batch_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_size_scaling");
    let total_cells = 100 * 100 * 100;

    group.sample_size(10);

    for num_steps in [10u64, 100, 500, 1000, 5000] {
        group.throughput(Throughput::Elements(total_cells * num_steps));

        // Parallel engine
        group.bench_with_input(
            BenchmarkId::new("Parallel", num_steps),
            &num_steps,
            |b, &steps| {
                b.iter_batched(
                    || scenario_waveguide(EngineType::Parallel, steps),
                    |(mut engine, batch)| {
                        let result = engine.run_batch(batch).unwrap();
                        black_box(result)
                    },
                    criterion::BatchSize::LargeInput,
                );
            },
        );

        // GPU engine
        if openems::fdtd::GpuEngine::is_available() {
            group.bench_with_input(
                BenchmarkId::new("Gpu", num_steps),
                &num_steps,
                |b, &steps| {
                    b.iter_batched(
                        || scenario_waveguide(EngineType::Gpu, steps),
                        |(mut engine, batch)| {
                            let result = engine.run_batch(batch).unwrap();
                            black_box(result)
                        },
                        criterion::BatchSize::LargeInput,
                    );
                },
            );
        }
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_waveguide,
    bench_large_domain,
    bench_excitation_scaling,
    bench_energy_monitoring,
    bench_pml_scaling,
    bench_batch_size,
);

criterion_main!(benches);
