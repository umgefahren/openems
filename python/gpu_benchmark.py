#!/usr/bin/env python3
"""
GPU Engine Performance Benchmark: Simulation Time vs Grid Size

This script benchmarks the Rust GPU FDTD engine by running a realistic
electromagnetic simulation (3D rectangular cavity resonator with a
Gaussian pulse excitation) at various grid sizes.

The cavity resonator is a physically meaningful scenario where EM waves
bounce between PEC (perfect electric conductor) walls, creating standing
wave patterns at resonant frequencies.

Usage:
    # First, build the Rust library with Python bindings:
    maturin develop --release --features python

    # Then run this benchmark:
    python python/gpu_benchmark.py
"""

import time
import sys
import numpy as np

# Try to import the Rust bindings
try:
    import openems_rust as ems
    from openems_rust import Grid, OpenEMS, SimulationStats, C0, VERSION
    RUST_AVAILABLE = True
except ImportError as e:
    RUST_AVAILABLE = False
    print(f"Rust openems_rust module not found: {e}")
    print("Build with: maturin develop --release --features python")
    print("")


def run_cavity_simulation(grid_size: int, timesteps: int, engine: str) -> dict:
    """
    Run a 3D rectangular cavity resonator simulation.

    This is a physically realistic FDTD simulation where:
    - A cubic PEC cavity of size L x L x L is created
    - A Gaussian pulse excitation is placed at the center
    - EM waves propagate and reflect off the cavity walls
    - The simulation runs for a fixed number of timesteps

    Args:
        grid_size: Number of cells in each dimension (N for N³ grid)
        timesteps: Number of FDTD timesteps to run
        engine: Engine type ("gpu", "parallel", "simd", "basic")

    Returns:
        Dictionary with simulation results and timing
    """
    # Physical parameters for the cavity
    # Cavity size: 100mm x 100mm x 100mm
    cavity_size = 0.1  # meters
    cell_size = cavity_size / grid_size  # uniform mesh

    # Create uniform grid
    grid = Grid.uniform(grid_size, grid_size, grid_size, cell_size)

    # Calculate appropriate excitation frequency
    # For a cubic cavity, the dominant TE101 mode resonant frequency is:
    # f_101 = c/(2L) * sqrt(2) ≈ 2.12 GHz for L=100mm
    # We use a broadband Gaussian to excite multiple modes
    f0 = 2e9  # 2 GHz center frequency
    fc = 1e9  # 1 GHz bandwidth (-20dB)

    # Create simulation
    sim = OpenEMS(num_timesteps=timesteps)
    sim.set_grid(grid)
    sim.set_engine_type(engine)

    # Set PEC boundaries (cavity walls)
    sim.set_boundary_cond(["pec", "pec", "pec", "pec", "pec", "pec"])

    # Add Gaussian pulse excitation at the center of the cavity
    # Polarized in z-direction
    center = grid_size // 2
    sim.add_gauss_excite(f0, fc, direction=2, position=(center, center, center))

    # Disable progress bar and verbose output for benchmarking
    sim.set_verbose(0)
    sim.set_show_progress(False)

    # Run the simulation
    stats = sim.run()

    return {
        "grid_size": grid_size,
        "cells": grid_size ** 3,
        "timesteps": stats.timesteps,
        "wall_time": stats.wall_time,
        "mcells_per_sec": stats.speed_mcells_per_sec,
        "peak_energy": stats.peak_energy,
        "final_energy": stats.final_energy,
        "engine": engine,
    }


def run_benchmark():
    """Run the full benchmark suite."""
    print("=" * 60)
    print("GPU FDTD Engine Performance Benchmark")
    print("Simulation: 3D Rectangular Cavity Resonator")
    print("=" * 60)
    print()

    # Check GPU availability
    if OpenEMS.gpu_available():
        print("GPU engine: Available")
    else:
        print("GPU engine: NOT AVAILABLE")
        print("The benchmark will run with CPU engines only.")
        print()

    # Grid sizes to test - including larger sizes where GPU should excel
    grid_sizes = [64, 96, 128, 160, 192, 256, 320, 384, 448, 512]

    # Number of timesteps
    timesteps = 100

    # Results storage
    gpu_results = []
    gpu_v2_results = []
    cpu_results = []

    # Warm up
    print("Warming up...")
    try:
        _ = run_cavity_simulation(32, 50, "gpu_v2")
        print("GPU V2 warmup complete.")
    except Exception as e:
        print(f"GPU V2 warmup failed: {e}")

    try:
        _ = run_cavity_simulation(32, 50, "gpu")
        print("GPU (original) warmup complete.")
    except Exception as e:
        print(f"GPU (original) warmup failed: {e}")

    _ = run_cavity_simulation(32, 50, "parallel")
    print("CPU warmup complete.")
    print()

    # Run GPU V2 benchmarks (optimized engine)
    if OpenEMS.gpu_available():
        print("Running GPU V2 (optimized) benchmarks...")
        print("-" * 60)
        print(f"{'Size':>8} {'Cells':>12} {'Steps':>8} {'Time (s)':>12} {'MC/s':>12}")
        print("-" * 60)

        for size in grid_sizes:
            try:
                result = run_cavity_simulation(size, timesteps, "gpu_v2")
                gpu_v2_results.append(result)
                print(f"{size:>5}³  {result['cells']:>12,} {result['timesteps']:>8} "
                      f"{result['wall_time']:>12.4f} {result['mcells_per_sec']:>12.2f}")
            except Exception as e:
                print(f"{size:>5}³  FAILED: {e}")
        print()

    # Run original GPU benchmarks for comparison (skip sizes > 448 due to buffer limits)
    if OpenEMS.gpu_available():
        print("Running GPU (original) benchmarks...")
        print("-" * 60)
        print(f"{'Size':>8} {'Cells':>12} {'Steps':>8} {'Time (s)':>12} {'MC/s':>12}")
        print("-" * 60)

        for size in grid_sizes:
            if size > 448:  # Original GPU has buffer size limits
                print(f"{size:>5}³  SKIPPED (buffer size limit)")
                continue
            try:
                result = run_cavity_simulation(size, timesteps, "gpu")
                gpu_results.append(result)
                print(f"{size:>5}³  {result['cells']:>12,} {result['timesteps']:>8} "
                      f"{result['wall_time']:>12.4f} {result['mcells_per_sec']:>12.2f}")
            except Exception as e:
                print(f"{size:>5}³  FAILED: {str(e)[:50]}")
        print()

    # Run CPU (Parallel SIMD) benchmarks
    print("Running CPU (Parallel SIMD) benchmarks...")
    print("-" * 60)
    print(f"{'Size':>8} {'Cells':>12} {'Steps':>8} {'Time (s)':>12} {'MC/s':>12}")
    print("-" * 60)

    for size in grid_sizes:
        # Skip very large grids for CPU (too slow)
        if size > 160:
            continue
        try:
            result = run_cavity_simulation(size, timesteps, "parallel")
            cpu_results.append(result)
            print(f"{size:>5}³  {result['cells']:>12,} {result['timesteps']:>8} "
                  f"{result['wall_time']:>12.4f} {result['mcells_per_sec']:>12.2f}")
        except Exception as e:
            print(f"{size:>5}³  FAILED: {e}")
    print()

    return gpu_v2_results, gpu_results, cpu_results


def plot_results(gpu_v2_results: list, gpu_results: list, cpu_results: list, save_path: str = "gpu_performance.png"):
    """Create performance plots."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available. Install with: pip install matplotlib")
        return

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('GPU FDTD Engine Performance: 3D Cavity Resonator Simulation',
                 fontsize=14, fontweight='bold')

    # Extract data
    if gpu_v2_results:
        gpu_v2_sizes = [r['grid_size'] for r in gpu_v2_results]
        gpu_v2_cells = [r['cells'] for r in gpu_v2_results]
        gpu_v2_times = [r['wall_time'] for r in gpu_v2_results]
        gpu_v2_speeds = [r['mcells_per_sec'] for r in gpu_v2_results]
    else:
        gpu_v2_sizes = gpu_v2_cells = gpu_v2_times = gpu_v2_speeds = []

    if gpu_results:
        gpu_sizes = [r['grid_size'] for r in gpu_results]
        gpu_cells = [r['cells'] for r in gpu_results]
        gpu_times = [r['wall_time'] for r in gpu_results]
        gpu_speeds = [r['mcells_per_sec'] for r in gpu_results]
    else:
        gpu_sizes = gpu_cells = gpu_times = gpu_speeds = []

    if cpu_results:
        cpu_sizes = [r['grid_size'] for r in cpu_results]
        cpu_cells = [r['cells'] for r in cpu_results]
        cpu_times = [r['wall_time'] for r in cpu_results]
        cpu_speeds = [r['mcells_per_sec'] for r in cpu_results]
    else:
        cpu_sizes = cpu_cells = cpu_times = cpu_speeds = []

    # Plot 1: Wall time vs Grid size (log scale)
    ax1 = axes[0, 0]
    if gpu_v2_times:
        ax1.semilogy(gpu_v2_sizes, gpu_v2_times, 'g-^', linewidth=2, markersize=8, label='GPU V2 (optimized)')
    if gpu_times:
        ax1.semilogy(gpu_sizes, gpu_times, 'b-o', linewidth=2, markersize=8, label='GPU (original)')
    if cpu_times:
        ax1.semilogy(cpu_sizes, cpu_times, 'r-s', linewidth=2, markersize=8, label='CPU (Parallel)')
    ax1.set_xlabel('Grid Size (N for N³ grid)', fontsize=11)
    ax1.set_ylabel('Wall Time (seconds)', fontsize=11)
    ax1.set_title('Simulation Time vs Grid Size', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3, which='both')

    # Plot 2: Wall time vs Total cells (log-log)
    ax2 = axes[0, 1]
    if gpu_v2_cells and gpu_v2_times:
        ax2.loglog(gpu_v2_cells, gpu_v2_times, 'g-^', linewidth=2, markersize=8, label='GPU V2 (optimized)')
    if gpu_cells and gpu_times:
        ax2.loglog(gpu_cells, gpu_times, 'b-o', linewidth=2, markersize=8, label='GPU (original)')
    if cpu_cells and cpu_times:
        ax2.loglog(cpu_cells, cpu_times, 'r-s', linewidth=2, markersize=8, label='CPU (Parallel)')

    # Add O(n) reference line
    ref_data = gpu_v2_cells if gpu_v2_cells else (gpu_cells if gpu_cells else [])
    ref_times = gpu_v2_times if gpu_v2_times else (gpu_times if gpu_times else [])
    if ref_data and ref_times and len(ref_data) >= 2:
        ref_cells = np.array(ref_data)
        ref_time = ref_times[0] * (ref_cells / ref_data[0])
        ax2.loglog(ref_cells, ref_time, 'k--', alpha=0.5, label='O(N) reference')

    ax2.set_xlabel('Total Cells', fontsize=11)
    ax2.set_ylabel('Wall Time (seconds)', fontsize=11)
    ax2.set_title('Simulation Time vs Total Cells (log-log)', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3, which='both')

    # Plot 3: Performance (MC/s) vs Grid size
    ax3 = axes[1, 0]
    if gpu_v2_speeds:
        ax3.plot(gpu_v2_sizes, gpu_v2_speeds, 'g-^', linewidth=2, markersize=8, label='GPU V2 (optimized)')
        avg_gpu_v2 = np.mean(gpu_v2_speeds)
        ax3.axhline(y=avg_gpu_v2, color='g', linestyle='--', alpha=0.5)
    if gpu_speeds:
        ax3.plot(gpu_sizes, gpu_speeds, 'b-o', linewidth=2, markersize=8, label='GPU (original)')
        avg_gpu = np.mean(gpu_speeds)
        ax3.axhline(y=avg_gpu, color='b', linestyle='--', alpha=0.5)
    if cpu_speeds:
        ax3.plot(cpu_sizes, cpu_speeds, 'r-s', linewidth=2, markersize=8, label='CPU (Parallel)')
        avg_cpu = np.mean(cpu_speeds)
        ax3.axhline(y=avg_cpu, color='r', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Grid Size (N for N³ grid)', fontsize=11)
    ax3.set_ylabel('Performance (MC/s)', fontsize=11)
    ax3.set_title('Throughput vs Grid Size', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: GPU V2 Speedup over CPU and original GPU
    ax4 = axes[1, 1]
    if gpu_v2_speeds and cpu_speeds:
        # Find common sizes
        gpu_v2_dict = {r['grid_size']: r['mcells_per_sec'] for r in gpu_v2_results}
        cpu_dict = {r['grid_size']: r['mcells_per_sec'] for r in cpu_results}
        common_sizes = sorted(set(gpu_v2_dict.keys()) & set(cpu_dict.keys()))

        if common_sizes:
            speedups = [gpu_v2_dict[s] / cpu_dict[s] for s in common_sizes]
            x = np.arange(len(common_sizes))
            width = 0.35

            bars = ax4.bar(x, speedups, width, color='green', alpha=0.7, label='GPU V2 vs CPU')
            ax4.set_xticks(x)
            ax4.set_xticklabels([f'{s}³' for s in common_sizes])
            ax4.axhline(y=1.0, color='k', linestyle='-', alpha=0.3)

            avg_speedup = np.mean(speedups)
            ax4.axhline(y=avg_speedup, color='green', linestyle='--', alpha=0.7,
                       label=f'V2 vs CPU avg: {avg_speedup:.1f}x')
            ax4.legend()

            # Add value labels on bars
            for bar, speedup in zip(bars, speedups):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                        f'{speedup:.2f}x', ha='center', va='bottom', fontsize=8)
    else:
        ax4.text(0.5, 0.5, 'GPU V2 or CPU data not available\nfor speedup comparison',
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)

    ax4.set_xlabel('Grid Size', fontsize=11)
    ax4.set_ylabel('Speedup (GPU V2 / CPU)', fontsize=11)
    ax4.set_title('GPU V2 Speedup over CPU', fontsize=12)
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {save_path}")
    plt.show()


def print_summary(gpu_v2_results: list, gpu_results: list, cpu_results: list):
    """Print benchmark summary statistics."""
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if gpu_v2_results:
        avg_gpu_v2_speed = np.mean([r['mcells_per_sec'] for r in gpu_v2_results])
        max_gpu_v2_speed = max([r['mcells_per_sec'] for r in gpu_v2_results])
        print(f"GPU V2 (optimized) Average Speed: {avg_gpu_v2_speed:.2f} MC/s")
        print(f"GPU V2 (optimized) Peak Speed:    {max_gpu_v2_speed:.2f} MC/s")

    if gpu_results:
        avg_gpu_speed = np.mean([r['mcells_per_sec'] for r in gpu_results])
        max_gpu_speed = max([r['mcells_per_sec'] for r in gpu_results])
        print(f"GPU (original) Average Speed: {avg_gpu_speed:.2f} MC/s")
        print(f"GPU (original) Peak Speed:    {max_gpu_speed:.2f} MC/s")

    if cpu_results:
        avg_cpu_speed = np.mean([r['mcells_per_sec'] for r in cpu_results])
        max_cpu_speed = max([r['mcells_per_sec'] for r in cpu_results])
        print(f"CPU Average Speed: {avg_cpu_speed:.2f} MC/s")
        print(f"CPU Peak Speed:    {max_cpu_speed:.2f} MC/s")

    if gpu_v2_results and cpu_results:
        # Calculate speedup for common sizes
        gpu_v2_dict = {r['grid_size']: r['mcells_per_sec'] for r in gpu_v2_results}
        cpu_dict = {r['grid_size']: r['mcells_per_sec'] for r in cpu_results}
        common_sizes = set(gpu_v2_dict.keys()) & set(cpu_dict.keys())

        if common_sizes:
            speedups = [gpu_v2_dict[s] / cpu_dict[s] for s in common_sizes]
            print(f"\nGPU V2 Speedup over CPU:")
            print(f"  Average: {np.mean(speedups):.2f}x")
            print(f"  Maximum: {max(speedups):.2f}x")
            print(f"  Minimum: {min(speedups):.2f}x")

    if gpu_v2_results and gpu_results:
        # Calculate V2 vs original GPU improvement
        gpu_v2_dict = {r['grid_size']: r['mcells_per_sec'] for r in gpu_v2_results}
        gpu_dict = {r['grid_size']: r['mcells_per_sec'] for r in gpu_results}
        common_sizes = set(gpu_v2_dict.keys()) & set(gpu_dict.keys())

        if common_sizes:
            improvements = [gpu_v2_dict[s] / gpu_dict[s] for s in common_sizes]
            print(f"\nGPU V2 Improvement over Original GPU:")
            print(f"  Average: {np.mean(improvements):.2f}x")
            print(f"  Maximum: {max(improvements):.2f}x")
            print(f"  Minimum: {min(improvements):.2f}x")


def main():
    if not RUST_AVAILABLE:
        sys.exit(1)

    print(f"openEMS Rust version: {VERSION}")
    print(f"Speed of light: {C0:.0f} m/s")
    print()

    # Run benchmarks
    gpu_v2_results, gpu_results, cpu_results = run_benchmark()

    # Print summary
    print_summary(gpu_v2_results, gpu_results, cpu_results)

    # Create plots
    if gpu_v2_results or gpu_results or cpu_results:
        plot_results(gpu_v2_results, gpu_results, cpu_results)
    else:
        print("No results to plot!")


if __name__ == "__main__":
    main()
