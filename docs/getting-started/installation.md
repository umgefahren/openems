# Installation

## Requirements

- Python 3.9 or later
- pip (Python package installer)

## Installation via pip

The easiest way to install openEMS is via pip:

```bash
pip install openems-rust
```

## Installation from Source

For the latest development version, you can install directly from the repository:

### Prerequisites

1. **Rust toolchain**: Install via [rustup](https://rustup.rs/):
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

2. **Maturin**: Build tool for Rust Python extensions:
   ```bash
   pip install maturin
   ```

### Building from Source

```bash
# Clone the repository
git clone https://github.com/umgefahren/openems.git
cd openems

# Build and install in development mode
maturin develop --features python

# Or build a wheel for distribution
maturin build --release --features python
pip install target/wheels/openems_rust-*.whl
```

## Verifying Installation

After installation, verify that openEMS is working correctly:

```python
import openems

# Check version
print(f"openEMS version: {openems.VERSION}")

# Verify constants are available
print(f"Speed of light: {openems.C0} m/s")

# Create a simple grid
grid = openems.Grid.uniform(10, 10, 10, delta=1e-3)
print(f"Grid cells: {grid.num_cells()}")
```

Expected output:
```
openEMS version: 1.0.0
Speed of light: 299792458.0 m/s
Grid cells: 1000
```

## Optional Dependencies

For visualization and post-processing, you may want to install:

```bash
pip install numpy matplotlib scipy
```

## Troubleshooting

### ImportError: No module named 'openems'

Make sure you installed the package correctly:
```bash
pip show openems-rust
```

If not found, reinstall:
```bash
pip install --force-reinstall openems-rust
```

### Build errors with Maturin

Ensure you have:
- Rust 1.70+ installed
- Python development headers (on Linux: `python3-dev` or `python3-devel`)

```bash
# Ubuntu/Debian
sudo apt install python3-dev

# Fedora/RHEL
sudo dnf install python3-devel

# macOS (with Homebrew Python)
# Headers are included with the Homebrew Python installation
```

## Platform-Specific Notes

### Linux

For best performance, ensure you have a recent version of glibc (2.17+).

### macOS

Supports both Intel and Apple Silicon (ARM64) Macs.

### Windows

Pre-built wheels are available for Windows. If building from source, ensure you have Visual Studio Build Tools installed.
