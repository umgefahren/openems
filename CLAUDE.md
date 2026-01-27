# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a complete Rust rewrite of the openEMS FDTD (Finite-Difference Time-Domain) electromagnetic field solver. The goal is to match or exceed C++ performance while providing better safety, cross-platform support, and static linking.

## Build Commands

```bash
cargo build                    # Debug build
cargo build --release          # Optimized release build
cargo build --all-features     # Build with all features (python, hdf5)
maturin develop                # Python bindings (development mode)
```

## Testing

```bash
cargo test                              # Run all tests
cargo test --all-features               # Run all tests with all features
cargo test fdtd::engine::tests          # Run tests in a specific module
cargo test test_gaussian_pulse          # Run a single test by name
cargo test -- --nocapture               # Show println! output
```

## Linting and Formatting

```bash
cargo fmt --all -- --check     # Check formatting
cargo fmt                      # Auto-format code
cargo clippy --all-targets -- -D warnings   # Lint (CI enforces no warnings)
```

## Benchmarks

```bash
cargo run --example benchmark --release  # Quick performance benchmark
cargo bench                              # Full Criterion benchmarks
```

## Architecture

### Core Modules

- `src/fdtd/` - FDTD engine, operator, simulation control
  - `operator.rs` - Pre-computes FDTD update coefficients
  - `engine.rs` - Field update implementations (basic, SIMD, parallel, GPU)
  - `simulation.rs` - High-level simulation controller
- `src/arrays/` - SIMD-optimized field arrays (`VectorField3D`)
- `src/geometry/` - Grid definitions and coordinate systems
- `src/extensions/` - Boundary conditions, materials, excitations
- `src/processing/` - Probes, SAR calculation, mode matching
- `src/nf2ff/` - Near-field to far-field transformation
- `src/io/` - VTK, HDF5, XML file I/O

### Engine Variants

The FDTD engine has four implementations selectable via `--engine`:
- `basic` - Single-threaded reference implementation
- `simd` - AVX2/AVX-512 accelerated (auto-dispatch)
- `parallel` - Multi-threaded via Rayon (default)
- `gpu` - WebGPU-accelerated (experimental)

### Extensions System

Extensions hook into the engine update cycle:
1. `pre_update_h` - Before H-field update
2. `post_update_h` - After H-field update
3. `pre_update_e` - Before E-field update
4. `post_update_e` - After E-field update

Implemented extensions: PML, Mur ABC, dispersive materials (Lorentz/Drude/Debye), conducting sheets, lumped RLC, TF/SF boundaries, steady-state detection, cylindrical coordinates.

### Performance Targets

- Engine: 700+ MC/s (mega-cells per second)
- SIMD speedup: ~2x over scalar
- Parallel speedup: ~10x on 8+ core machines

## Key Documentation

- **PORTING_STATUS.md** - Tracks C++ module porting status
- **FEATURE_PARITY.md** - Feature comparison with original C++
- **LANGUAGE_DECISION.md** - Documents why Rust was chosen

## CLI Usage

```bash
openems <INPUT.xml> [OPTIONS]
  --engine [basic|simd|parallel|gpu]   # Engine type
  --num-threads <N>                    # Thread count (0 = auto)
  --disable-dumps                      # Skip field dumps
  --debug-material                     # Dump material distribution
  --debug-operator                     # Dump operator coefficients
  -v, --verbose [0-3]                  # Verbosity level
```
