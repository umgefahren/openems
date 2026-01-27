//! High-level simulation control.
//!
//! The Simulation struct provides a convenient interface for setting up
//! and running FDTD simulations.

use super::{
    BoundaryConditions, EnergyMonitorConfig, EngineBatch, EngineType, Excitation, Operator,
    ScheduledExcitation, TerminationConfig, TerminationReason, TimestepInfo,
};
use crate::extensions::AnyExtension;
use crate::fdtd::Engine;
use crate::geometry::Grid;
use crate::{Error, Result};

use indicatif::{ProgressBar, ProgressStyle};
use instant::Instant;
use log::info;

/// Simulation state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimulationState {
    /// Initial state, not yet set up
    Created,
    /// Set up and ready to run
    Ready,
    /// Currently running
    Running,
    /// Completed
    Finished,
    /// Stopped early (by user or convergence)
    Stopped,
}

/// End condition for simulation
#[derive(Debug, Clone)]
pub enum EndCondition {
    /// Run for fixed number of timesteps
    Timesteps(u64),
    /// Run until energy decays below threshold (in dB relative to peak)
    EnergyDecay(f64),
    /// Run for fixed simulation time
    SimulationTime(f64),
}

impl Default for EndCondition {
    fn default() -> Self {
        Self::Timesteps(10000)
    }
}

/// Statistics from a simulation run
#[derive(Debug, Clone)]
pub struct SimulationStats {
    /// Total timesteps executed
    pub timesteps: u64,
    /// Total simulation time (seconds)
    pub sim_time: f64,
    /// Wall clock time (seconds)
    pub wall_time: f64,
    /// Peak energy during simulation
    pub peak_energy: f64,
    /// Final energy
    pub final_energy: f64,
    /// Average speed (cells/second)
    pub speed_mcells_per_sec: f64,
}

/// Main simulation controller.
pub struct Simulation {
    /// Grid definition
    grid: Grid,
    /// Boundary conditions
    boundaries: BoundaryConditions,
    /// Engine type selection
    engine_type: EngineType,
    /// Excitation sources
    excitations: Vec<Excitation>,
    /// End condition
    end_condition: EndCondition,
    /// Current state
    state: SimulationState,
    /// Operator (created during setup)
    operator: Option<Operator>,
    /// Engine (created during setup)
    engine: Option<Engine>,
    /// Verbosity level
    verbose: u8,
    /// Show progress bar
    show_progress: bool,
    /// Batch size for timestep execution (None = auto-select based on engine type)
    batch_size: Option<u64>,
}

impl Simulation {
    /// Create a new simulation with the given grid.
    pub fn new(grid: Grid) -> Self {
        Self {
            grid,
            boundaries: BoundaryConditions::default(),
            engine_type: EngineType::default(),
            excitations: Vec::new(),
            end_condition: EndCondition::default(),
            state: SimulationState::Created,
            operator: None,
            engine: None,
            verbose: 1,
            show_progress: true,
            batch_size: None, // Auto-select based on engine type
        }
    }

    /// Set boundary conditions.
    pub fn set_boundaries(&mut self, boundaries: BoundaryConditions) -> &mut Self {
        self.boundaries = boundaries;
        self
    }

    /// Set engine type.
    pub fn set_engine_type(&mut self, engine_type: EngineType) -> &mut Self {
        self.engine_type = engine_type;
        self
    }

    /// Add an excitation source.
    pub fn add_excitation(&mut self, excitation: Excitation) -> &mut Self {
        self.excitations.push(excitation);
        self
    }

    /// Set end condition.
    pub fn set_end_condition(&mut self, condition: EndCondition) -> &mut Self {
        self.end_condition = condition;
        self
    }

    /// Set verbosity level (0=quiet, 1=normal, 2=verbose).
    pub fn set_verbose(&mut self, level: u8) -> &mut Self {
        self.verbose = level;
        self
    }

    /// Enable/disable progress bar.
    pub fn set_show_progress(&mut self, show: bool) -> &mut Self {
        self.show_progress = show;
        self
    }

    /// Set batch size for timestep execution.
    ///
    /// Default (None) auto-selects based on engine type:
    /// - Basic/SIMD/Parallel: 100 timesteps per batch
    /// - GPU: 1000 timesteps per batch (minimizes CPU↔GPU transfers)
    pub fn set_batch_size(&mut self, batch_size: Option<u64>) -> &mut Self {
        self.batch_size = batch_size;
        self
    }

    /// Get timestep information.
    pub fn timestep_info(&self) -> TimestepInfo {
        TimestepInfo::calculate(&self.grid)
    }

    /// Set up the simulation (create operator and engine).
    pub fn setup(&mut self) -> Result<()> {
        if self.state != SimulationState::Created {
            return Err(Error::Config("Simulation already set up".into()));
        }

        let info = self.timestep_info();

        if self.verbose >= 1 {
            info!(
                "FDTD simulation size: {}x{}x{} -> {} cells",
                self.grid.dimensions().nx,
                self.grid.dimensions().ny,
                self.grid.dimensions().nz,
                info.num_cells
            );
            info!(
                "FDTD timestep: {:.6e} s, Nyquist: {:.3e} Hz",
                info.dt, info.nyquist_freq
            );
            info!("Estimated memory: {}", info.memory_display());
        }

        // Create operator
        let operator = Operator::new(self.grid.clone(), self.boundaries.clone())?;

        // Create engine
        let engine = Engine::new(&operator, self.engine_type)?;

        self.operator = Some(operator);
        self.engine = Some(engine);
        self.state = SimulationState::Ready;

        Ok(())
    }

    /// Run the simulation using batched execution.
    pub fn run(&mut self) -> Result<SimulationStats> {
        // Set up if not already done
        if self.state == SimulationState::Created {
            self.setup()?;
        }

        if self.state != SimulationState::Ready {
            return Err(Error::Config("Simulation not ready to run".into()));
        }

        self.state = SimulationState::Running;

        let operator = self.operator.as_ref().unwrap();
        let engine = self.engine.as_mut().unwrap();
        let dt = operator.timestep();

        // Determine batch size based on engine type
        let batch_size = self.batch_size.unwrap_or_else(|| {
            match self.engine_type {
                EngineType::Gpu => 1000, // Large batches for GPU to minimize transfers
                _ => 100,                // Smaller batches for CPU engines
            }
        });

        // Determine total timesteps to execute
        let max_timesteps = match &self.end_condition {
            EndCondition::Timesteps(n) => *n,
            EndCondition::SimulationTime(t) => (t / dt).ceil() as u64,
            EndCondition::EnergyDecay(_) => 1_000_000, // Upper limit for energy decay
        };

        let energy_threshold = match &self.end_condition {
            EndCondition::EnergyDecay(db) => Some(*db),
            _ => None,
        };

        // Progress bar
        let progress = if self.show_progress {
            let pb = ProgressBar::new(max_timesteps);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} ({per_sec})")
                    .unwrap()
                    .progress_chars("##-"),
            );
            Some(pb)
        } else {
            None
        };

        let start_time = Instant::now();
        let mut peak_energy = 0.0f64;
        let mut timesteps_run = 0u64;

        // Main batching loop
        while timesteps_run < max_timesteps {
            let remaining = max_timesteps - timesteps_run;
            let this_batch_size = remaining.min(batch_size);

            // Pre-sample excitations for this batch
            let scheduled_excitations: Vec<ScheduledExcitation> = self
                .excitations
                .iter()
                .map(|exc| {
                    ScheduledExcitation::from_excitation(exc, timesteps_run, this_batch_size, dt)
                })
                .collect();

            // Configure batch execution
            // Note: Using empty extension list for now - extensions will be added later
            let batch = EngineBatch {
                num_steps: Some(this_batch_size),
                excitations: scheduled_excitations,
                extensions: Vec::<AnyExtension>::new(), // No extensions yet
                termination: TerminationConfig {
                    max_timesteps: Some(this_batch_size),
                    energy_decay_db: energy_threshold,
                    check_interval: 100,
                },
                energy_monitoring: EnergyMonitorConfig {
                    sample_interval: 100, // Sample energy every 100 steps
                    track_peak: true,
                    decay_threshold: energy_threshold,
                },
            };

            // Execute batch
            let result = engine.run_batch(batch)?;
            timesteps_run += result.timesteps_executed;

            // Update peak energy from batch samples
            for sample in &result.energy_samples {
                if sample.total_energy > peak_energy {
                    peak_energy = sample.total_energy;
                }
            }

            // Update progress bar
            if let Some(ref pb) = progress {
                pb.set_position(timesteps_run);
            }

            // Check batch termination reason
            match result.termination_reason {
                TerminationReason::EnergyDecay { final_decay_db } => {
                    if self.verbose >= 1 {
                        info!(
                            "Energy decay reached: {:.1} dB at timestep {}",
                            final_decay_db, timesteps_run
                        );
                    }
                    break;
                }
                TerminationReason::ExtensionStop { ref reason } => {
                    if self.verbose >= 1 {
                        info!(
                            "Extension stopped simulation: {} at timestep {}",
                            reason, timesteps_run
                        );
                    }
                    break;
                }
                TerminationReason::StepsCompleted => {
                    // Continue to next batch
                }
            }
        }

        if let Some(pb) = progress {
            pb.finish_with_message("Simulation complete");
        }

        let wall_time = start_time.elapsed().as_secs_f64();

        // Calculate final energy
        let (e_field, h_field) = engine.read_fields();
        let final_energy = e_field.energy() + h_field.energy();

        let num_cells = operator.dimensions().total();
        let speed = (timesteps_run as f64 * num_cells as f64) / wall_time / 1e6;

        self.state = SimulationState::Finished;

        let stats = SimulationStats {
            timesteps: timesteps_run,
            sim_time: timesteps_run as f64 * dt,
            wall_time,
            peak_energy,
            final_energy,
            speed_mcells_per_sec: speed,
        };

        if self.verbose >= 1 {
            info!(
                "Completed {} timesteps in {:.2}s ({:.2} MC/s)",
                stats.timesteps, stats.wall_time, stats.speed_mcells_per_sec
            );
        }

        Ok(stats)
    }

    /// Get reference to the engine (if set up).
    pub fn engine(&self) -> Option<&Engine> {
        self.engine.as_ref()
    }

    /// Get mutable reference to the engine (if set up).
    pub fn engine_mut(&mut self) -> Option<&mut Engine> {
        self.engine.as_mut()
    }

    /// Get reference to the operator (if set up).
    pub fn operator(&self) -> Option<&Operator> {
        self.operator.as_ref()
    }

    /// Get the current state.
    pub fn state(&self) -> SimulationState {
        self.state
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simulation_basic() {
        let grid = Grid::uniform(20, 20, 20, 1e-3);
        let mut sim = Simulation::new(grid);

        sim.set_end_condition(EndCondition::Timesteps(100))
            .set_verbose(0)
            .set_show_progress(false);

        // Add a simple excitation
        let exc = Excitation::gaussian(1e9, 0.5, 2, (10, 10, 10));
        sim.add_excitation(exc);

        let stats = sim.run().unwrap();
        assert_eq!(stats.timesteps, 100);
        assert!(stats.speed_mcells_per_sec > 0.0);
    }
}
