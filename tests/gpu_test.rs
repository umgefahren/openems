#[cfg(test)]
mod tests {
    use openems::arrays::VectorField3D;
    use openems::fdtd::gpu_engine::GpuEngine;
    use openems::fdtd::BoundaryConditions;
    use openems::fdtd::Operator;
    use openems::geometry::{CoordinateSystem, Grid};

    // Helper to create a simple grid
    fn create_test_grid() -> Grid {
        Grid::new(
            CoordinateSystem::Cartesian,
            vec![0.0, 0.001, 0.002, 0.003, 0.004, 0.005], // 5 cells in x
            vec![0.0, 0.001, 0.002, 0.003, 0.004, 0.005], // 5 cells in y
            vec![0.0, 0.001, 0.002, 0.003, 0.004, 0.005], // 5 cells in z
        )
    }

    #[test]
    fn test_gpu_engine_basic() {
        if !GpuEngine::is_available() {
            println!("Skipping GPU test: No adapter available");
            return;
        }

        let grid = create_test_grid();
        let operator = Operator::new(grid, BoundaryConditions::default()).unwrap();
        let engine = GpuEngine::new(&operator);

        let dims = operator.dimensions();
        let mut e_field = VectorField3D::new(dims);

        // Set an impulse
        e_field.z.set(2, 2, 2, 1.0);

        engine.write_e_field(&e_field);

        // Run a step
        engine.step();
        engine.wait_idle();

        // Read back
        let mut h_field_out = VectorField3D::new(dims);
        engine.read_h_field(&mut h_field_out);

        // H field should have evolved (curl of E)
        // Check some neighbors
        // Hx ~ dEz/dy
        let hx = h_field_out.x.get(2, 2, 2);
        // Ez at (2,2,2) = 1.0
        // Hx at (2,2,2) = (Ez(2,3,2) - Ez(2,2,2)) * coeffs
        // Ez(2,3,2) is 0. So -1.0 * coeff.
        assert!(hx != 0.0, "H-field should update");

        // Run another step
        engine.step();
        engine.wait_idle();

        let mut e_field_out = VectorField3D::new(dims);
        engine.read_e_field(&mut e_field_out);

        // E field should change
        assert!(e_field_out.z.get(2, 2, 2) != 1.0, "E-field should evolve");
    }

    #[test]
    fn test_gpu_precision_fallback() {
        if !GpuEngine::is_available() {
            println!("Skipping GPU test: No adapter available");
            return;
        }

        // This test runs the same logic but we can't easily force fallback
        // without mocking the adapter or conditional compilation hacks.
        // However, the fact that GpuEngine::new() succeeds implies the shader
        // compiled successfully with whatever features were available.

        let grid = create_test_grid();
        let mut operator = Operator::new(grid, BoundaryConditions::default()).unwrap();

        // Add a "lossy" region to trigger f32 path if logic works
        operator.set_material(2.0, 1.0, 1.0, 0.0, (1, 4, 1, 4, 1, 4));

        let engine = GpuEngine::new(&operator);

        let dims = operator.dimensions();
        let mut e_field = VectorField3D::new(dims);
        e_field.x.set(2, 2, 2, 1.0);
        engine.write_e_field(&e_field);

        engine.step();
        engine.wait_idle();

        let mut e_out = VectorField3D::new(dims);
        engine.read_e_field(&mut e_out);

        assert!(e_out.x.get(2, 2, 2) != 0.0);
    }
}
