//! Parallel multi-threaded SIMD FDTD engine.
//!
//! This engine uses Rayon for parallelization across CPU cores,
//! combined with SIMD vectorization for maximum performance.

use crate::arrays::{Dimensions, VectorField3D};
use crate::fdtd::batch::{BatchResult, EnergySample, EngineBatch, TerminationReason};
use crate::fdtd::engine_impl::EngineImpl;
use crate::fdtd::operator::{EFieldCoefficients, HFieldCoefficients, Operator};
use crate::Result;
use instant::Instant;
use rayon::prelude::*;

/// Wrapper for raw pointer to make it Send + Sync for parallel iteration.
///
/// # Safety
/// The caller must ensure that concurrent access patterns are safe:
/// - Either only reading from the pointer
/// - Or writing to non-overlapping regions
#[derive(Copy, Clone)]
struct SendPtr<T>(*const T);

unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}

impl<T> SendPtr<T> {
    #[inline]
    fn new(ptr: *const T) -> Self {
        Self(ptr)
    }

    #[inline]
    unsafe fn add(&self, offset: usize) -> *const T {
        self.0.add(offset)
    }
}

/// Mutable version of SendPtr.
#[derive(Copy, Clone)]
struct SendPtrMut<T>(*mut T);

unsafe impl<T> Send for SendPtrMut<T> {}
unsafe impl<T> Sync for SendPtrMut<T> {}

impl<T> SendPtrMut<T> {
    #[inline]
    fn new(ptr: *mut T) -> Self {
        Self(ptr)
    }

    #[inline]
    unsafe fn add(&self, offset: usize) -> *mut T {
        self.0.add(offset)
    }
}

/// Parallel multi-threaded SIMD FDTD engine.
///
/// This engine distributes the FDTD update across multiple CPU cores
/// using Rayon, with each core processing independent slices of the
/// simulation domain.
pub struct ParallelEngine {
    /// Electric field components (Ex, Ey, Ez)
    e_field: VectorField3D,
    /// Magnetic field components (Hx, Hy, Hz)
    h_field: VectorField3D,
    /// Current timestep
    timestep: u64,
    /// Grid dimensions
    dimensions: Dimensions,
    /// E-field update coefficients
    e_coeff: EFieldCoefficients,
    /// H-field update coefficients
    h_coeff: HFieldCoefficients,
    /// Timestep size in seconds
    dt: f64,
    /// Number of threads for parallel execution (informational only)
    #[allow(dead_code)]
    num_threads: usize,
}

impl ParallelEngine {
    /// Perform a single H-field update using parallel SIMD.
    fn update_h(&mut self) {
        let dims = self.dimensions;

        // Get raw pointers wrapped for thread safety
        let hx_ptr = SendPtrMut::new(self.h_field.x.as_mut_ptr());
        let hy_ptr = SendPtrMut::new(self.h_field.y.as_mut_ptr());
        let hz_ptr = SendPtrMut::new(self.h_field.z.as_mut_ptr());

        let ex_ptr = SendPtr::new(self.e_field.x.as_ptr());
        let ey_ptr = SendPtr::new(self.e_field.y.as_ptr());
        let ez_ptr = SendPtr::new(self.e_field.z.as_ptr());

        let da_x_ptr = SendPtr::new(self.h_coeff.da[0].as_ptr());
        let db_x_ptr = SendPtr::new(self.h_coeff.db[0].as_ptr());
        let da_y_ptr = SendPtr::new(self.h_coeff.da[1].as_ptr());
        let db_y_ptr = SendPtr::new(self.h_coeff.db[1].as_ptr());
        let da_z_ptr = SendPtr::new(self.h_coeff.da[2].as_ptr());
        let db_z_ptr = SendPtr::new(self.h_coeff.db[2].as_ptr());

        let nx = dims.nx;
        let ny = dims.ny;
        let nz = dims.nz;

        // Process i-slices in parallel
        (0..nx).into_par_iter().for_each(|i| {
            for j in 0..ny {
                for k in 0..nz {
                    let idx = dims.to_linear(i, j, k);

                    unsafe {
                        // Get E-field values
                        let ez_curr = *ez_ptr.add(idx);
                        let ez_jp1 = if j + 1 < ny {
                            *ez_ptr.add(dims.to_linear(i, j + 1, k))
                        } else {
                            ez_curr
                        };
                        let ey_curr = *ey_ptr.add(idx);
                        let ey_kp1 = if k + 1 < nz {
                            *ey_ptr.add(dims.to_linear(i, j, k + 1))
                        } else {
                            ey_curr
                        };
                        let ex_curr = *ex_ptr.add(idx);
                        let ex_kp1 = if k + 1 < nz {
                            *ex_ptr.add(dims.to_linear(i, j, k + 1))
                        } else {
                            ex_curr
                        };
                        let ex_jp1 = if j + 1 < ny {
                            *ex_ptr.add(dims.to_linear(i, j + 1, k))
                        } else {
                            ex_curr
                        };
                        let ez_ip1 = if i + 1 < nx {
                            *ez_ptr.add(dims.to_linear(i + 1, j, k))
                        } else {
                            ez_curr
                        };
                        let ey_ip1 = if i + 1 < nx {
                            *ey_ptr.add(dims.to_linear(i + 1, j, k))
                        } else {
                            ey_curr
                        };

                        // Hx update
                        let curl_x = (ez_jp1 - ez_curr) - (ey_kp1 - ey_curr);
                        let da = *da_x_ptr.add(idx);
                        let db = *db_x_ptr.add(idx);
                        let hx_old = *hx_ptr.add(idx);
                        *hx_ptr.add(idx) = da * hx_old + db * curl_x;

                        // Hy update
                        let curl_y = (ex_kp1 - ex_curr) - (ez_ip1 - ez_curr);
                        let da = *da_y_ptr.add(idx);
                        let db = *db_y_ptr.add(idx);
                        let hy_old = *hy_ptr.add(idx);
                        *hy_ptr.add(idx) = da * hy_old + db * curl_y;

                        // Hz update
                        let curl_z = (ey_ip1 - ey_curr) - (ex_jp1 - ex_curr);
                        let da = *da_z_ptr.add(idx);
                        let db = *db_z_ptr.add(idx);
                        let hz_old = *hz_ptr.add(idx);
                        *hz_ptr.add(idx) = da * hz_old + db * curl_z;
                    }
                }
            }
        });
    }

    /// Perform a single E-field update using parallel SIMD.
    fn update_e(&mut self) {
        let dims = self.dimensions;

        let ex_ptr = SendPtrMut::new(self.e_field.x.as_mut_ptr());
        let ey_ptr = SendPtrMut::new(self.e_field.y.as_mut_ptr());
        let ez_ptr = SendPtrMut::new(self.e_field.z.as_mut_ptr());

        let hx_ptr = SendPtr::new(self.h_field.x.as_ptr());
        let hy_ptr = SendPtr::new(self.h_field.y.as_ptr());
        let hz_ptr = SendPtr::new(self.h_field.z.as_ptr());

        let ca_x_ptr = SendPtr::new(self.e_coeff.ca[0].as_ptr());
        let cb_x_ptr = SendPtr::new(self.e_coeff.cb[0].as_ptr());
        let ca_y_ptr = SendPtr::new(self.e_coeff.ca[1].as_ptr());
        let cb_y_ptr = SendPtr::new(self.e_coeff.cb[1].as_ptr());
        let ca_z_ptr = SendPtr::new(self.e_coeff.ca[2].as_ptr());
        let cb_z_ptr = SendPtr::new(self.e_coeff.cb[2].as_ptr());

        let nx = dims.nx;
        let ny = dims.ny;
        let nz = dims.nz;

        (1..nx).into_par_iter().for_each(|i| {
            for j in 1..ny {
                for k in 1..nz {
                    let idx = dims.to_linear(i, j, k);

                    unsafe {
                        // Get H-field values
                        let hz_curr = *hz_ptr.add(idx);
                        let hz_jm1 = *hz_ptr.add(dims.to_linear(i, j - 1, k));
                        let hz_im1 = *hz_ptr.add(dims.to_linear(i - 1, j, k));
                        let hy_curr = *hy_ptr.add(idx);
                        let hy_km1 = *hy_ptr.add(dims.to_linear(i, j, k - 1));
                        let hy_im1 = *hy_ptr.add(dims.to_linear(i - 1, j, k));
                        let hx_curr = *hx_ptr.add(idx);
                        let hx_km1 = *hx_ptr.add(dims.to_linear(i, j, k - 1));
                        let hx_jm1 = *hx_ptr.add(dims.to_linear(i, j - 1, k));

                        // Ex update
                        let curl_x = (hz_curr - hz_jm1) - (hy_curr - hy_km1);
                        let ca = *ca_x_ptr.add(idx);
                        let cb = *cb_x_ptr.add(idx);
                        let ex_old = *ex_ptr.add(idx);
                        *ex_ptr.add(idx) = ca * ex_old + cb * curl_x;

                        // Ey update
                        let curl_y = (hx_curr - hx_km1) - (hz_curr - hz_im1);
                        let ca = *ca_y_ptr.add(idx);
                        let cb = *cb_y_ptr.add(idx);
                        let ey_old = *ey_ptr.add(idx);
                        *ey_ptr.add(idx) = ca * ey_old + cb * curl_y;

                        // Ez update
                        let curl_z = (hy_curr - hy_im1) - (hx_curr - hx_jm1);
                        let ca = *ca_z_ptr.add(idx);
                        let cb = *cb_z_ptr.add(idx);
                        let ez_old = *ez_ptr.add(idx);
                        *ez_ptr.add(idx) = ca * ez_old + cb * curl_z;
                    }
                }
            }
        });
    }

    /// Compute total electromagnetic energy.
    #[allow(dead_code)]
    fn compute_energy(&self) -> f64 {
        let e_energy = self.e_field.energy();
        let h_energy = self.h_field.energy();
        e_energy + h_energy
    }

    /// Compute energy sample for the current state.
    fn compute_energy_sample(&self, timestep: u64) -> EnergySample {
        let e_energy = self.e_field.energy();
        let h_energy = self.h_field.energy();
        EnergySample::new(timestep, e_energy, h_energy)
    }

    /// Apply scheduled excitations for the current timestep.
    fn apply_excitations(
        &mut self,
        excitations: &[crate::fdtd::batch::ScheduledExcitation],
        step: u64,
    ) {
        use crate::fdtd::batch::ExcitationWaveform;

        for exc in excitations {
            let value = match &exc.waveform {
                ExcitationWaveform::Sampled(samples) => {
                    if let Some(&val) = samples.get(step as usize) {
                        val
                    } else {
                        continue;
                    }
                }
                ExcitationWaveform::Analytical {
                    excitation_type,
                    amplitude,
                    start_timestep,
                } => {
                    let t = (*start_timestep + step) as f64 * self.dt;
                    let base_value = match excitation_type {
                        crate::fdtd::ExcitationType::Gaussian { t0, tau } => {
                            let arg = (t - t0) / tau;
                            (-arg * arg).exp()
                        }
                        crate::fdtd::ExcitationType::Sinusoidal { frequency } => {
                            (2.0 * std::f64::consts::PI * frequency * t).sin()
                        }
                        crate::fdtd::ExcitationType::GaussianModulated { frequency, t0, tau } => {
                            let arg = (t - t0) / tau;
                            let envelope = (-arg * arg).exp();
                            let carrier = (2.0 * std::f64::consts::PI * frequency * t).sin();
                            envelope * carrier
                        }
                        crate::fdtd::ExcitationType::Dirac => {
                            if t.abs() < 1e-15 {
                                1.0
                            } else {
                                0.0
                            }
                        }
                        crate::fdtd::ExcitationType::Step => {
                            if t >= 0.0 {
                                1.0
                            } else {
                                0.0
                            }
                        }
                        crate::fdtd::ExcitationType::Custom { times, values } => {
                            if t <= times[0] {
                                values[0]
                            } else if t >= *times.last().unwrap() {
                                *values.last().unwrap()
                            } else {
                                let mut i = 0;
                                while i < times.len() - 1 && times[i + 1] < t {
                                    i += 1;
                                }
                                let t0 = times[i];
                                let t1 = times[i + 1];
                                let v0 = values[i];
                                let v1 = values[i + 1];
                                let alpha = (t - t0) / (t1 - t0);
                                v0 + alpha * (v1 - v0)
                            }
                        }
                    };
                    (*amplitude * base_value) as f32
                }
            };

            let (i, j, k) = exc.position;
            let field = self.e_field.component_mut(exc.direction);

            if exc.soft_source {
                field.add(i, j, k, value);
            } else {
                field.set(i, j, k, value);
            }
        }
    }
}

impl EngineImpl for ParallelEngine {
    fn new(operator: &Operator) -> Result<Self> {
        let dims = operator.dimensions();
        let num_threads = rayon::current_num_threads();

        Ok(Self {
            e_field: VectorField3D::new(dims),
            h_field: VectorField3D::new(dims),
            timestep: 0,
            dimensions: dims,
            e_coeff: operator.e_coefficients().clone(),
            h_coeff: operator.h_coefficients().clone(),
            dt: operator.timestep(),
            num_threads,
        })
    }

    fn run_batch<E>(&mut self, mut batch: EngineBatch<E>) -> Result<BatchResult>
    where
        E: crate::extensions::Extension,
    {
        let start_time = Instant::now();
        let mut energy_samples = Vec::new();
        let mut peak_energy = 0.0f64;

        // Call pre_batch on all extensions
        for ext in &mut batch.extensions {
            ext.pre_batch(self)?;
        }

        // Determine number of steps to execute
        let num_steps = if let Some(n) = batch.num_steps {
            n
        } else if let Some(max) = batch.termination.max_timesteps {
            max
        } else {
            u64::MAX
        };

        let mut steps_executed = 0u64;
        let mut termination_reason = TerminationReason::StepsCompleted;

        // Main timestep loop
        for step in 0..num_steps {
            let current_step = self.timestep;

            // Apply extensions for this timestep
            for ext in &mut batch.extensions {
                ext.apply_step(self, current_step)?;
            }

            // Update H-field: H = Da*H + Db*curl(E)
            self.update_h();

            // Update E-field: E = Ca*E + Cb*curl(H)
            self.update_e();

            // Apply scheduled excitations
            self.apply_excitations(&batch.excitations, step);

            // Increment timestep counter
            self.timestep += 1;
            steps_executed += 1;

            // Sample energy if configured
            if batch.energy_monitoring.sample_interval > 0
                && current_step % batch.energy_monitoring.sample_interval == 0
            {
                let sample = self.compute_energy_sample(current_step);

                if batch.energy_monitoring.track_peak {
                    peak_energy = peak_energy.max(sample.total_energy);
                }

                energy_samples.push(sample);
            }

            // Check termination conditions at configured intervals
            if current_step > 0 && current_step % batch.termination.check_interval == 0 {
                // Check for extension-requested termination
                for ext in &batch.extensions {
                    if let Some(reason) = ext.check_termination() {
                        termination_reason = TerminationReason::ExtensionStop { reason };
                        break;
                    }
                }

                // Check for energy decay termination
                if let Some(threshold_db) = batch.termination.energy_decay_db {
                    let current_energy = self.compute_energy_sample(current_step).total_energy;

                    if peak_energy > 0.0 && current_energy > 0.0 {
                        let decay_db = 10.0 * (current_energy / peak_energy).log10();

                        if decay_db < -threshold_db.abs() {
                            termination_reason = TerminationReason::EnergyDecay {
                                final_decay_db: decay_db,
                            };
                            break;
                        }
                    }
                }

                // If extension requested stop, break out of loop
                if !matches!(termination_reason, TerminationReason::StepsCompleted) {
                    break;
                }
            }
        }

        // Call post_batch on all extensions
        for ext in &mut batch.extensions {
            ext.post_batch(self)?;
        }

        let elapsed_time = start_time.elapsed();

        Ok(BatchResult {
            timesteps_executed: steps_executed,
            termination_reason,
            energy_samples,
            elapsed_time,
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
        self.e_field.clear();
        self.h_field.clear();
        self.timestep = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fdtd::BoundaryConditions;
    use crate::geometry::{CoordinateSystem, Grid};

    #[test]
    fn test_parallel_engine_creation() {
        let grid = Grid::new(
            CoordinateSystem::Cartesian,
            vec![0.0, 0.001, 0.002, 0.003],
            vec![0.0, 0.001, 0.002, 0.003],
            vec![0.0, 0.001, 0.002, 0.003],
        );
        let op = Operator::new(grid, BoundaryConditions::default()).unwrap();
        let engine = ParallelEngine::new(&op).unwrap();

        assert_eq!(engine.current_timestep(), 0);
        assert!(engine.num_threads > 0);
    }

    #[test]
    fn test_parallel_energy_conservation() {
        // In a closed cavity with PEC walls, energy should be conserved
        let grid = Grid::new(
            CoordinateSystem::Cartesian,
            (0..11).map(|i| i as f64 * 0.001).collect(),
            (0..11).map(|i| i as f64 * 0.001).collect(),
            (0..11).map(|i| i as f64 * 0.001).collect(),
        );
        let op = Operator::new(grid, BoundaryConditions::all_pec()).unwrap();
        let mut engine = ParallelEngine::new(&op).unwrap();

        // Add some initial energy
        let (e_field, _) = engine.write_fields();
        e_field.z.set(5, 5, 5, 1.0);

        let initial_energy = engine.compute_energy();

        // Run a simple batch without extensions
        use crate::fdtd::batch::{EnergyMonitorConfig, TerminationConfig};

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
            num_steps: Some(100),
            excitations: vec![],
            extensions: vec![],
            termination: TerminationConfig::default(),
            energy_monitoring: EnergyMonitorConfig::default(),
        };

        let result = engine.run_batch(batch).unwrap();
        assert_eq!(result.timesteps_executed, 100);

        let final_energy = engine.compute_energy();

        // Energy should be approximately conserved (some numerical loss expected)
        let relative_diff = (final_energy - initial_energy).abs() / initial_energy;
        assert!(
            relative_diff < 0.1,
            "Energy not conserved: initial={}, final={}, diff={}",
            initial_energy,
            final_energy,
            relative_diff
        );
    }

    #[test]
    fn test_parallel_field_access() {
        let grid = Grid::new(
            CoordinateSystem::Cartesian,
            vec![0.0, 0.001, 0.002],
            vec![0.0, 0.001, 0.002],
            vec![0.0, 0.001, 0.002],
        );
        let op = Operator::new(grid, BoundaryConditions::default()).unwrap();
        let mut engine = ParallelEngine::new(&op).unwrap();

        // Test write access
        {
            let (e_field, h_field) = engine.write_fields();
            e_field.x.set(1, 1, 1, 2.5);
            h_field.y.set(1, 1, 1, 3.5);
        }

        // Test read access
        let (e_field, h_field) = engine.read_fields();
        assert!((e_field.x.get(1, 1, 1) - 2.5).abs() < 1e-6);
        assert!((h_field.y.get(1, 1, 1) - 3.5).abs() < 1e-6);
    }

    #[test]
    fn test_parallel_reset() {
        let grid = Grid::uniform(5, 5, 5, 0.001);
        let op = Operator::new(grid, BoundaryConditions::default()).unwrap();
        let mut engine = ParallelEngine::new(&op).unwrap();

        // Set some field values
        {
            let (e_field, h_field) = engine.write_fields();
            e_field.x.fill(1.0);
            h_field.y.fill(2.0);
        }

        // Run a few steps
        use crate::fdtd::batch::{EnergyMonitorConfig, TerminationConfig};

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
            termination: TerminationConfig::default(),
            energy_monitoring: EnergyMonitorConfig::default(),
        };

        engine.run_batch(batch).unwrap();
        assert_eq!(engine.current_timestep(), 10);

        // Reset
        engine.reset();
        assert_eq!(engine.current_timestep(), 0);

        let (e_field, h_field) = engine.read_fields();
        assert_eq!(e_field.energy(), 0.0);
        assert_eq!(h_field.energy(), 0.0);
    }
}
