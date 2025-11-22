# ace-active-carver

Uncertainty-driven on-the-fly active learning system using Pacemaker (ACE potentials) and LAMMPS.

## Prerequisites

- **OS**: Linux (Ubuntu/Debian recommended)
- **Python**: 3.10
- **Package Manager**: [uv](https://github.com/astral-sh/uv)
- **Compilers**: C++ compiler (g++ / clang++) supporting C++11 or later, CMake.

## Setup

1. **Install uv** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Initialize Environment**:
   Run the setup script to install Python dependencies:
   ```bash
   ./setup_env.sh
   ```

3. **Install LAMMPS (Manual Step)**
   The project requires a custom build of LAMMPS with the `ML-PACE` package enabled. The `lammps` Python package is installed by `uv`, but it requires the shared library `liblammps.so` to function correctly with the PACE potential.

   **Build Instructions:**
   ```bash
   # 1. Clone LAMMPS
   git clone -b stable https://github.com/lammps/lammps.git mylammps
   cd mylammps

   # 2. Build
   mkdir build && cd build
   cmake ../cmake \
       -D PKG_PYTHON=ON \
       -D PKG_ML-PACE=ON \
       -D BUILD_SHARED_LIBS=ON \
       -D PYTHON_EXECUTABLE=$(uv python find)

   make -j$(nproc)

   # 3. Set environment variables
   # Add the build folder to LD_LIBRARY_PATH so Python can find liblammps.so
   export LD_LIBRARY_PATH=$PWD:$LD_LIBRARY_PATH
   ```

## Directory Structure

- `src/`: Source code
- `tests/`: Test files
- `data/`: Temporary storage for learning data and dumps

## Development

- **Format code**: `uv run black .`
- **Sort imports**: `uv run isort .`
