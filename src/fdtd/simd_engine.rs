//! SIMD-accelerated FDTD engine implementation.
//!
//! This engine uses SIMD operations along z-slices for improved performance
//! over the basic scalar engine. It operates single-threaded but with
//! vectorization, making it faster than BasicEngine while simpler than
//! ParallelEngine.

use crate::arrays::{Dimensions, VectorField3D};
use crate::extensions::Extension;
use crate::fdtd::batch::{
    BatchResult, EnergySample, EngineBatch, ExcitationWaveform, TerminationReason,
};
use crate::fdtd::engine_impl::EngineImpl;
use crate::fdtd::{Excitation, Operator};
use crate::Result;
use instant::Instant;

/// SIMD-accelerated FDTD engine.
///
/// Performs FDTD time-stepping using SIMD operations on z-slices for
/// improved performance. The field update kernels operate on contiguous
/// z-lines, enabling compiler auto-vectorization.
pub struct SimdEngine {
    /// Electric field (Ex, Ey, Ez)
    e_field: VectorField3D,
    /// Magnetic field (Hx, Hy, Hz)
    h_field: VectorField3D,
    /// Current timestep
    timestep: u64,
    /// Grid dimensions
    dimensions: Dimensions,
    /// E-field update coefficients
    e_coeff: crate::fdtd::operator::EFieldCoefficients,
    /// H-field update coefficients
    h_coeff: crate::fdtd::operator::HFieldCoefficients,
    /// Timestep size (dt)
    dt: f64,
}

impl EngineImpl for SimdEngine {
    fn new(operator: &Operator) -> Result<Self> {
        let dims = operator.dimensions();

        Ok(Self {
            e_field: VectorField3D::new(dims),
            h_field: VectorField3D::new(dims),
            timestep: 0,
            dimensions: dims,
            e_coeff: operator.e_coefficients().clone(),
            h_coeff: operator.h_coefficients().clone(),
            dt: operator.timestep(),
        })
    }

    fn run_batch<E>(&mut self, batch: EngineBatch<E>) -> Result<BatchResult>
    where
        E: Extension,
    {
        let start = Instant::now();
        let mut timesteps_executed = 0u64;
        let mut energy_samples = Vec::new();
        let mut peak_energy = 0.0f64;

        // Pre-batch extension initialization
        let mut extensions = batch.extensions;
        for ext in extensions.iter_mut() {
            ext.pre_batch(self)?;
        }

        // Determine number of steps to execute
        let num_steps = match batch.num_steps {
            Some(n) => n,
            None => batch
                .termination
                .max_timesteps
                .unwrap_or(u64::MAX)
                .min(u64::MAX),
        };

        // Main timestep loop
        for step_idx in 0..num_steps {
            // Update H-field (uses E-field)
            self.update_h_simd();

            // Apply extensions: post_update_h
            for ext in extensions.iter_mut() {
                ext.apply_step(self, self.timestep)?;
            }

            // Update E-field (uses H-field)
            self.update_e_simd();

            // Apply excitations
            for exc in &batch.excitations {
                let value = match &exc.waveform {
                    ExcitationWaveform::Sampled(samples) => {
                        if step_idx < samples.len() as u64 {
                            samples[step_idx as usize]
                        } else {
                            0.0
                        }
                    }
                    ExcitationWaveform::Analytical {
                        excitation_type,
                        amplitude,
                        start_timestep,
                    } => {
                        let dt = self.dt;
                        let t = (start_timestep + step_idx) as f64 * dt;
                        // Create temporary excitation to evaluate the waveform
                        let exc = Excitation {
                            excitation_type: excitation_type.clone(),
                            direction: exc.direction,
                            position: exc.position,
                            amplitude: *amplitude,
                            soft_source: exc.soft_source,
                        };
                        exc.evaluate(t) as f32
                    }
                };

                let (i, j, k) = exc.position;
                let component = match exc.direction {
                    0 => &mut self.e_field.x,
                    1 => &mut self.e_field.y,
                    2 => &mut self.e_field.z,
                    _ => {
                        return Err(crate::Error::Config(
                            "Invalid excitation direction".to_string(),
                        ))
                    }
                };

                if exc.soft_source {
                    let old = component.get(i, j, k);
                    component.set(i, j, k, old + value);
                } else {
                    component.set(i, j, k, value);
                }
            }

            // Apply extensions: post_update_e
            for ext in extensions.iter_mut() {
                ext.apply_step(self, self.timestep)?;
            }

            self.timestep += 1;
            timesteps_executed += 1;

            // Energy monitoring
            if batch.energy_monitoring.sample_interval > 0
                && step_idx % batch.energy_monitoring.sample_interval == 0
            {
                let sample = EnergySample::from_fields(self.timestep, &self.e_field, &self.h_field);
                if batch.energy_monitoring.track_peak {
                    peak_energy = peak_energy.max(sample.total_energy);
                }
                energy_samples.push(sample);
            }

            // Check termination conditions periodically
            if step_idx > 0 && step_idx % batch.termination.check_interval == 0 {
                // Check for extension-requested termination
                for ext in extensions.iter() {
                    if let Some(reason) = ext.check_termination() {
                        // Post-batch extension cleanup
                        for ext in extensions.iter_mut() {
                            ext.post_batch(self)?;
                        }

                        return Ok(BatchResult {
                            timesteps_executed,
                            termination_reason: TerminationReason::ExtensionStop { reason },
                            energy_samples,
                            elapsed_time: start.elapsed(),
                        });
                    }
                }

                // Check energy decay
                if let Some(threshold_db) = batch.termination.energy_decay_db {
                    if peak_energy > 0.0 && !energy_samples.is_empty() {
                        let current_energy = energy_samples.last().unwrap().total_energy;
                        let decay_db = 10.0 * (peak_energy / current_energy.max(1e-30)).log10();

                        if decay_db >= threshold_db {
                            // Post-batch extension cleanup
                            for ext in extensions.iter_mut() {
                                ext.post_batch(self)?;
                            }

                            return Ok(BatchResult {
                                timesteps_executed,
                                termination_reason: TerminationReason::EnergyDecay {
                                    final_decay_db: decay_db,
                                },
                                energy_samples,
                                elapsed_time: start.elapsed(),
                            });
                        }
                    }
                }

                // Check max timesteps
                if let Some(max_ts) = batch.termination.max_timesteps {
                    if timesteps_executed >= max_ts {
                        break;
                    }
                }
            }
        }

        // Post-batch extension cleanup
        for ext in extensions.iter_mut() {
            ext.post_batch(self)?;
        }

        Ok(BatchResult {
            timesteps_executed,
            termination_reason: TerminationReason::StepsCompleted,
            energy_samples,
            elapsed_time: start.elapsed(),
        })
    }

    fn current_timestep(&self) -> u64 {
        self.timestep
    }

    fn read_fields(&mut self) -> (&VectorField3D, &VectorField3D) {
        (&self.e_field, &self.h_field)
    }

    fn write_fields(&mut self) -> (&mut VectorField3D, &mut VectorField3D) {
        (&mut self.e_field, &mut self.h_field)
    }

    fn reset(&mut self) {
        let dims = self.dimensions;
        self.e_field = VectorField3D::new(dims);
        self.h_field = VectorField3D::new(dims);
        self.timestep = 0;
    }
}

impl SimdEngine {
    /// SIMD-optimized H-field update operating on z-lines.
    ///
    /// This method processes z-slices for improved cache locality and
    /// automatic vectorization. The compiler can auto-vectorize the inner
    /// k-loop since it operates on contiguous memory.
    fn update_h_simd(&mut self) {
        let dims = self.dimensions;
        let h_coeff = &self.h_coeff;
        let nz = dims.nz;

        for i in 0..dims.nx {
            for j in 0..dims.ny {
                // Get z-slices for vectorized operations
                let hx_line = self.h_field.x.z_slice_mut(i, j);
                let hy_line = self.h_field.y.z_slice_mut(i, j);
                let hz_line = self.h_field.z.z_slice_mut(i, j);

                let ez_line = self.e_field.z.z_slice(i, j);
                let ez_jp1 = if j + 1 < dims.ny {
                    self.e_field.z.z_slice(i, j + 1)
                } else {
                    ez_line // boundary
                };

                let ey_line = self.e_field.y.z_slice(i, j);
                let ex_line = self.e_field.x.z_slice(i, j);
                let ex_jp1 = if j + 1 < dims.ny {
                    self.e_field.x.z_slice(i, j + 1)
                } else {
                    ex_line
                };

                let ey_ip1 = if i + 1 < dims.nx {
                    self.e_field.y.z_slice(i + 1, j)
                } else {
                    ey_line
                };

                let ez_ip1 = if i + 1 < dims.nx {
                    self.e_field.z.z_slice(i + 1, j)
                } else {
                    ez_line
                };

                let da_x = h_coeff.da[0].z_slice(i, j);
                let db_x = h_coeff.db[0].z_slice(i, j);
                let da_y = h_coeff.da[1].z_slice(i, j);
                let db_y = h_coeff.db[1].z_slice(i, j);
                let da_z = h_coeff.da[2].z_slice(i, j);
                let db_z = h_coeff.db[2].z_slice(i, j);

                // Update Hx: curl_x = dEz/dy - dEy/dz
                for k in 0..nz {
                    let dez_dy = ez_jp1[k] - ez_line[k];
                    let dey_dz = if k + 1 < nz {
                        ey_line[k + 1] - ey_line[k]
                    } else {
                        0.0
                    };
                    let curl_x = dez_dy - dey_dz;
                    hx_line[k] = da_x[k] * hx_line[k] + db_x[k] * curl_x;
                }

                // Update Hy: curl_y = dEx/dz - dEz/dx
                for k in 0..nz {
                    let dex_dz = if k + 1 < nz {
                        ex_line[k + 1] - ex_line[k]
                    } else {
                        0.0
                    };
                    let dez_dx = ez_ip1[k] - ez_line[k];
                    let curl_y = dex_dz - dez_dx;
                    hy_line[k] = da_y[k] * hy_line[k] + db_y[k] * curl_y;
                }

                // Update Hz: curl_z = dEy/dx - dEx/dy
                for k in 0..nz {
                    let dey_dx = ey_ip1[k] - ey_line[k];
                    let dex_dy = ex_jp1[k] - ex_line[k];
                    let curl_z = dey_dx - dex_dy;
                    hz_line[k] = da_z[k] * hz_line[k] + db_z[k] * curl_z;
                }
            }
        }
    }

    /// SIMD-optimized E-field update operating on z-lines.
    ///
    /// Similar to update_h_simd, this processes z-slices for improved
    /// vectorization. Note that E-field updates start at index 1 to handle
    /// boundary conditions properly.
    fn update_e_simd(&mut self) {
        let dims = self.dimensions;
        let e_coeff = &self.e_coeff;
        let nz = dims.nz;

        for i in 1..dims.nx {
            for j in 1..dims.ny {
                let ex_line = self.e_field.x.z_slice_mut(i, j);
                let ey_line = self.e_field.y.z_slice_mut(i, j);
                let ez_line = self.e_field.z.z_slice_mut(i, j);

                let hz_line = self.h_field.z.z_slice(i, j);
                let hz_jm1 = self.h_field.z.z_slice(i, j - 1);
                let hz_im1 = self.h_field.z.z_slice(i - 1, j);

                let hy_line = self.h_field.y.z_slice(i, j);
                let hy_im1 = self.h_field.y.z_slice(i - 1, j);

                let hx_line = self.h_field.x.z_slice(i, j);
                let hx_jm1 = self.h_field.x.z_slice(i, j - 1);

                let ca_x = e_coeff.ca[0].z_slice(i, j);
                let cb_x = e_coeff.cb[0].z_slice(i, j);
                let ca_y = e_coeff.ca[1].z_slice(i, j);
                let cb_y = e_coeff.cb[1].z_slice(i, j);
                let ca_z = e_coeff.ca[2].z_slice(i, j);
                let cb_z = e_coeff.cb[2].z_slice(i, j);

                // Update Ex: curl = dHz/dy - dHy/dz
                for k in 1..nz {
                    let dhz_dy = hz_line[k] - hz_jm1[k];
                    let dhy_dz = hy_line[k] - hy_line[k - 1];
                    let curl_x = dhz_dy - dhy_dz;
                    ex_line[k] = ca_x[k] * ex_line[k] + cb_x[k] * curl_x;
                }

                // Update Ey: curl = dHx/dz - dHz/dx
                for k in 1..nz {
                    let dhx_dz = hx_line[k] - hx_line[k - 1];
                    let dhz_dx = hz_line[k] - hz_im1[k];
                    let curl_y = dhx_dz - dhz_dx;
                    ey_line[k] = ca_y[k] * ey_line[k] + cb_y[k] * curl_y;
                }

                // Update Ez: curl = dHy/dx - dHx/dy
                for k in 1..nz {
                    let dhy_dx = hy_line[k] - hy_im1[k];
                    let dhx_dy = hx_line[k] - hx_jm1[k];
                    let curl_z = dhy_dx - dhx_dy;
                    ez_line[k] = ca_z[k] * ez_line[k] + cb_z[k] * curl_z;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Grid;

    #[test]
    fn test_simd_engine_creation() {
        let grid = Grid::uniform(10, 10, 10, 1e-3);
        let boundaries = crate::fdtd::BoundaryConditions::all_pec();
        let operator = Operator::new(grid, boundaries).unwrap();
        let engine = SimdEngine::new(&operator);
        assert!(engine.is_ok());
    }

    #[test]
    fn test_simd_engine_timestep() {
        let grid = Grid::uniform(10, 10, 10, 1e-3);
        let boundaries = crate::fdtd::BoundaryConditions::all_pec();
        let operator = Operator::new(grid, boundaries).unwrap();
        let mut engine = SimdEngine::new(&operator).unwrap();

        assert_eq!(engine.current_timestep(), 0);

        // Define a minimal no-op extension for testing
        struct NoOpExtension;
        impl crate::extensions::Extension for NoOpExtension {
            fn name(&self) -> &str {
                "NoOp"
            }
            fn apply_step<E>(&mut self, _engine: &mut E, _step: u64) -> crate::Result<()>
            where
                E: crate::fdtd::EngineImpl,
            {
                Ok(())
            }
        }

        let batch: EngineBatch<NoOpExtension> = EngineBatch {
            num_steps: Some(10),
            excitations: vec![],
            extensions: vec![],
            termination: Default::default(),
            energy_monitoring: Default::default(),
        };

        let result = engine.run_batch(batch);
        assert!(result.is_ok());
        assert_eq!(engine.current_timestep(), 10);
    }

    #[test]
    fn test_simd_engine_reset() {
        let grid = Grid::uniform(10, 10, 10, 1e-3);
        let boundaries = crate::fdtd::BoundaryConditions::all_pec();
        let operator = Operator::new(grid, boundaries).unwrap();
        let mut engine = SimdEngine::new(&operator).unwrap();

        // Run some timesteps
        // Define a minimal no-op extension for testing
        struct NoOpExtension;
        impl crate::extensions::Extension for NoOpExtension {
            fn name(&self) -> &str {
                "NoOp"
            }
            fn apply_step<E>(&mut self, _engine: &mut E, _step: u64) -> crate::Result<()>
            where
                E: crate::fdtd::EngineImpl,
            {
                Ok(())
            }
        }

        let batch: EngineBatch<NoOpExtension> = EngineBatch {
            num_steps: Some(5),
            excitations: vec![],
            extensions: vec![],
            termination: Default::default(),
            energy_monitoring: Default::default(),
        };
        engine.run_batch(batch).unwrap();

        assert_eq!(engine.current_timestep(), 5);

        // Reset
        engine.reset();
        assert_eq!(engine.current_timestep(), 0);

        // Verify fields are zeroed
        let (e, h) = engine.read_fields();
        assert_eq!(e.x.get(5, 5, 5), 0.0);
        assert_eq!(h.y.get(5, 5, 5), 0.0);
    }

    #[test]
    fn test_simd_engine_field_access() {
        let grid = Grid::uniform(10, 10, 10, 1e-3);
        let boundaries = crate::fdtd::BoundaryConditions::all_pec();
        let operator = Operator::new(grid, boundaries).unwrap();
        let mut engine = SimdEngine::new(&operator).unwrap();

        // Write to fields
        {
            let (e, h) = engine.write_fields();
            e.x.set(5, 5, 5, 1.0);
            h.y.set(3, 3, 3, 2.0);
        }

        // Read back
        let (e, h) = engine.read_fields();
        assert_eq!(e.x.get(5, 5, 5), 1.0);
        assert_eq!(h.y.get(3, 3, 3), 2.0);
    }
}
