# ace-active-carver

Uncertainty-driven on-the-fly active learning system using [Pacemaker](https://github.com/ICAMS/pacemaker) (ACE potentials) and [LAMMPS](https://www.lammps.org/).

This project implements a **Delta-Learning** workflow where an Atomic Cluster Expansion (ACE) potential is trained to correct the difference between an empirical potential (Lennard-Jones) and First-Principles calculations (DFT/Espresso). The system autonomously explores the phase space using Molecular Dynamics (MD) and triggers retraining whenever the uncertainty of the ACE potential exceeds a predefined threshold.

## Architecture

The active learning loop consists of the following steps:

1.  **MD Simulation**: Runs a Molecular Dynamics simulation using LAMMPS with a hybrid potential (`lj/cut` + `pace/extrapolation`).
2.  **Uncertainty Detection**: Monitors the extrapolation grade ($\gamma$) of the ACE potential on-the-fly via `fix pace/extrapolation`. If $\gamma$ exceeds a threshold, the simulation halts.
3.  **Cluster Carving**: Extracts atomic clusters centered on high-uncertainty regions from the MD snapshot.
4.  **Labeling**: Computes the target values (Energy/Forces) for the clusters using Quantum Espresso (DFT) and subtracts the baseline LJ contribution.
    $$ E_{target} = E_{DFT} - E_{LJ} $$
    $$ F_{target} = F_{DFT} - F_{LJ} $$
5.  **Training**: Retrains/Fine-tunes the ACE potential using Pacemaker on the newly labeled dataset.
6.  **Resume**: Resumes the MD simulation from the last checkpoint using the updated potential.

## Prerequisites

- **OS**: Linux (Ubuntu/Debian recommended)
- **Python**: 3.10
- **Package Manager**: [uv](https://github.com/astral-sh/uv)
- **DFT Software**: Quantum Espresso (`pw.x` executable reachable in PATH).
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
   The project requires a custom build of LAMMPS with the `ML-PACE` package enabled.

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

## Configuration

The system is configured via `config.yaml`. The file should contain the following sections:

```yaml
md_params:
  elements: ["Ag", "Pd"]       # List of element symbols
  timestep: 0.001              # MD timestep in ps
  temperature: 300.0           # Temperature in Kelvin
  pressure: 1.0                # Pressure in bar
  restart_freq: 1000           # Frequency of writing restart files
  n_steps: 100000              # Total steps per iteration
  initial_structure: "start.data" # Path to initial LAMMPS data file

al_params:
  r_core: 4.0                  # Core radius for cluster carving
  r_buffer: 2.0                # Buffer radius for cluster carving
  gamma_threshold: 2.0         # Uncertainty threshold to trigger AL
  n_clusters: 5                # Number of clusters to extract per halt
  initial_potential: "pot.yace" # Path to initial ACE potential

dft_params:
  pseudo_dir: "./pseudos"      # Directory containing pseudopotentials
  ecutwfc: 40.0                # Wavefunction cutoff (Ry)
  kpts: [2, 2, 2]              # K-points grid

lj_params:
  epsilon: 1.0
  sigma: 1.0
  cutoff: 2.5
```

## Usage

Ensure your environment is activated and dependencies are installed.

```bash
# Activate virtual environment
source .venv/bin/activate

# Run the main controller
python src/main.py
```

The system will create `data/iteration_X` directories for each AL cycle, storing logs, dumps, and trained potentials.

## Testing

The project includes a suite of unit and integration tests using `pytest`.

```bash
# Run all tests
pytest tests/

# Run with output logging
pytest -s tests/
```

### Test Scope
- **Unit Tests**: Verify Delta calculation logic (checking $E_{DFT} - E_{LJ}$) and LAMMPS input file generation.
- **Integration Tests**: Mock the entire AL loop (LAMMPS, Espresso, Pacemaker) to verify the workflow logic (Halt -> Train -> Resume).

## Directory Structure

- `src/`: Source code
    - `main.py`: Main controller loop.
    - `md_engine.py`: Interface with LAMMPS.
    - `labeler.py`: Delta-learning label generation.
    - `trainer.py`: Interface with Pacemaker.
    - `active_learning.py`: Cluster carving logic.
    - `config.py`: Configuration data classes.
- `tests/`: Test files (`test_integration.py`, `test_labeler.py`, `test_md_engine.py`).
- `data/`: Runtime data storage (ignored by git).
