//! Python bindings for openEMS using PyO3.
//!
//! This module provides Python bindings that expose the core openEMS
//! functionality to Python users.

#[cfg(feature = "python")]
use pyo3::exceptions::PyRuntimeError;
#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
use crate::fdtd::{
    BoundaryConditions, EndCondition, EngineType, Excitation, GpuEngine,
    Simulation as RustSimulation,
};
#[cfg(feature = "python")]
use crate::geometry::Grid as RustGrid;

/// Python wrapper for Grid
#[cfg(feature = "python")]
#[pyclass(name = "Grid")]
pub struct PyGrid {
    inner: RustGrid,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyGrid {
    /// Create a uniform grid
    #[staticmethod]
    fn uniform(nx: usize, ny: usize, nz: usize, delta: f64) -> Self {
        Self {
            inner: RustGrid::uniform(nx, ny, nz, delta),
        }
    }

    /// Create a grid from mesh line arrays
    #[staticmethod]
    fn from_lines(x: Vec<f64>, y: Vec<f64>, z: Vec<f64>) -> Self {
        Self {
            inner: RustGrid::cartesian(x, y, z),
        }
    }

    /// Get the number of cells
    fn num_cells(&self) -> usize {
        self.inner.dimensions().total()
    }

    /// Get the minimum cell size
    fn cell_size(&self) -> (f64, f64, f64) {
        self.inner.cell_size()
    }
}

/// Simulation statistics returned after running
#[cfg(feature = "python")]
#[pyclass(name = "SimulationStats")]
pub struct PySimulationStats {
    #[pyo3(get)]
    timesteps: u64,
    #[pyo3(get)]
    wall_time: f64,
    #[pyo3(get)]
    speed_mcells_per_sec: f64,
    #[pyo3(get)]
    peak_energy: f64,
    #[pyo3(get)]
    final_energy: f64,
}

#[cfg(feature = "python")]
#[pymethods]
impl PySimulationStats {
    fn __repr__(&self) -> String {
        format!(
            "SimulationStats(timesteps={}, wall_time={:.3}s, speed={:.2} MC/s)",
            self.timesteps, self.wall_time, self.speed_mcells_per_sec
        )
    }
}

/// Excitation configuration stored before simulation
#[cfg(feature = "python")]
#[derive(Clone)]
struct ExcitationConfig {
    freq: f64,
    bandwidth: f64,
    direction: usize,
    position: (usize, usize, usize),
}

/// Python wrapper for openEMS simulation
#[cfg(feature = "python")]
#[pyclass(name = "OpenEMS")]
pub struct PyOpenEMS {
    grid: Option<RustGrid>,
    num_timesteps: u64,
    end_criteria_db: Option<f64>,
    engine_type: EngineType,
    excitations: Vec<ExcitationConfig>,
    boundary_cond: BoundaryConditions,
    verbose: u8,
    show_progress: bool,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyOpenEMS {
    /// Create a new openEMS simulation
    #[new]
    #[pyo3(signature = (num_timesteps=10000))]
    fn new(num_timesteps: u64) -> Self {
        Self {
            grid: None,
            num_timesteps,
            end_criteria_db: None,
            engine_type: EngineType::Parallel,
            excitations: Vec::new(),
            boundary_cond: BoundaryConditions::all_pec(),
            verbose: 1,
            show_progress: true,
        }
    }

    /// Check if GPU engine is available
    #[staticmethod]
    fn gpu_available() -> bool {
        GpuEngine::is_available()
    }

    /// Set the computational grid
    fn set_grid(&mut self, grid: &PyGrid) {
        self.grid = Some(grid.inner.clone());
    }

    /// Set the engine type
    /// Options: "basic", "simd", "parallel", "gpu"
    fn set_engine_type(&mut self, engine: &str) -> PyResult<()> {
        self.engine_type = match engine.to_lowercase().as_str() {
            "basic" => EngineType::Basic,
            "simd" => EngineType::Simd,
            "parallel" => EngineType::Parallel,
            "gpu" | "gpu_v2" => EngineType::Gpu,
            _ => {
                return Err(PyRuntimeError::new_err(format!(
                    "Unknown engine type: {}. Use 'basic', 'simd', 'parallel', or 'gpu'",
                    engine
                )))
            }
        };
        Ok(())
    }

    /// Set verbose level (0 = silent, 1 = normal, 2 = verbose, 3 = debug)
    fn set_verbose(&mut self, level: u8) {
        self.verbose = level;
    }

    /// Enable/disable progress bar
    fn set_show_progress(&mut self, show: bool) {
        self.show_progress = show;
    }

    /// Set Gaussian excitation (legacy API - use add_gauss_excite instead)
    #[pyo3(signature = (f0, fc))]
    fn set_gauss_excite(&mut self, f0: f64, fc: f64) {
        // Default excitation at center, z-direction
        self.excitations.clear();
        self.excitations.push(ExcitationConfig {
            freq: f0,
            bandwidth: fc,
            direction: 2,
            position: (0, 0, 0), // Will be set to center when grid is known
        });
    }

    /// Add Gaussian pulse excitation
    #[pyo3(signature = (f0, fc, direction=2, position=(0,0,0)))]
    fn add_gauss_excite(
        &mut self,
        f0: f64,
        fc: f64,
        direction: usize,
        position: (usize, usize, usize),
    ) {
        self.excitations.push(ExcitationConfig {
            freq: f0,
            bandwidth: fc,
            direction,
            position,
        });
    }

    /// Set the boundary conditions
    /// List of 6 strings: [x_min, x_max, y_min, y_max, z_min, z_max]
    /// Options: "pec", "pmc", "mur", "pml_8"
    fn set_boundary_cond(&mut self, bc: Vec<String>) -> PyResult<()> {
        if bc.len() != 6 {
            return Err(PyRuntimeError::new_err(
                "Boundary conditions must be a list of 6 elements",
            ));
        }

        // For now, we only support PEC boundaries
        // TODO: Implement other boundary types
        for b in &bc {
            match b.to_lowercase().as_str() {
                "pec" | "0" => {}
                "pmc" | "1" => {
                    return Err(PyRuntimeError::new_err("PMC boundaries not yet supported"))
                }
                "mur" | "2" => {
                    return Err(PyRuntimeError::new_err("Mur ABC not yet supported via Python"))
                }
                "pml_8" | "pml" | "3" => {
                    return Err(PyRuntimeError::new_err("PML not yet supported via Python"))
                }
                _ => {
                    return Err(PyRuntimeError::new_err(format!(
                        "Unknown boundary condition: {}",
                        b
                    )))
                }
            }
        }

        self.boundary_cond = BoundaryConditions::all_pec();
        Ok(())
    }

    /// Set end criteria based on energy decay in dB
    fn set_end_criteria(&mut self, db: f64) {
        self.end_criteria_db = Some(db);
    }

    /// Run the simulation and return statistics
    fn run(&self) -> PyResult<PySimulationStats> {
        let grid = self
            .grid
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Grid not set. Call set_grid() first."))?;

        let mut sim = RustSimulation::new(grid.clone());
        sim.set_engine_type(self.engine_type);
        sim.set_verbose(self.verbose);

        if let Some(db) = self.end_criteria_db {
            sim.set_end_condition(EndCondition::EnergyDecay(db));
        } else {
            sim.set_end_condition(EndCondition::Timesteps(self.num_timesteps));
        }

        // Add excitations
        for exc in &self.excitations {
            let position = if exc.position == (0, 0, 0) {
                // Default to center of grid
                let dims = grid.dimensions();
                (dims.nx / 2, dims.ny / 2, dims.nz / 2)
            } else {
                exc.position
            };

            sim.add_excitation(Excitation::gaussian(
                exc.freq,
                exc.bandwidth / exc.freq, // Convert bandwidth to relative
                exc.direction,
                position,
            ));
        }

        // Setup and run
        sim.setup()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let result = sim
            .run()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        Ok(PySimulationStats {
            timesteps: result.timesteps,
            wall_time: result.wall_time,
            speed_mcells_per_sec: result.speed_mcells_per_sec,
            peak_energy: result.peak_energy,
            final_energy: result.final_energy,
        })
    }
}

/// Python module definition
#[cfg(feature = "python")]
#[pymodule]
fn openems_rust(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGrid>()?;
    m.add_class::<PyOpenEMS>()?;
    m.add_class::<PySimulationStats>()?;

    // Add constants
    m.add("C0", crate::constants::C0)?;
    m.add("EPS0", crate::constants::EPS0)?;
    m.add("MU0", crate::constants::MU0)?;
    m.add("Z0", crate::constants::Z0)?;
    m.add("VERSION", crate::VERSION)?;

    // Convenience re-exports
    m.add("Grid", m.getattr("Grid")?)?;
    m.add("OpenEMS", m.getattr("OpenEMS")?)?;
    m.add("SimulationStats", m.getattr("SimulationStats")?)?;

    Ok(())
}
