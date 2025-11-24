# ace-active-carver

Uncertainty-driven on-the-fly active learning system using [Pacemaker](https://github.com/ICAMS/pacemaker) (ACE potentials) and [LAMMPS](https://www.lammps.org/).

This project implements a **Hybrid MD-kMC Active Learning** framework where an Atomic Cluster Expansion (ACE) potential is trained to correct the difference between an empirical potential (Lennard-Jones) and First-Principles calculations (DFT/Espresso). The system autonomously explores phase space using both Molecular Dynamics (MD) for thermal sampling and Kinetic Monte Carlo (kMC) for rare-event evolution, triggering retraining whenever the model's uncertainty exceeds a threshold.

## Table of Contents
1. [Prerequisites & Build Up](#prerequisites--build-up)
2. [Structure Generation](#structure-generation)
3. [Labeling (Delta Learning)](#labeling-delta-learning)
4. [Training](#training)
5. [Simulation Modes](#simulation-modes)
    - [Conventional MD](#conventional-md)
    - [Conventional kMC](#conventional-kmc)
    - [Hybrid MD/kMC Active Learning](#hybrid-mdkmc-active-learning)
6. [Usage](#usage)
7. [Configuration](#configuration)

---

## Prerequisites & Build Up

This project relies on a specific ecosystem of tools.

### 1. System Requirements
- **OS**: Linux (Ubuntu/Debian recommended).
- **Python**: 3.10.
- **Package Manager**: [uv](https://github.com/astral-sh/uv).
- **Compilers**: C++11 compatible compiler (g++/clang++), CMake.
- **DFT Engine**: Quantum Espresso (`pw.x` in PATH).

### 2. Environment Setup
Install `uv` and initialize the Python environment:
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install python dependencies (creates .venv)
./setup_env.sh
```

### 3. Compiling LAMMPS with ML-PACE
The system requires a custom LAMMPS build with the `ML-PACE` package enabled to run ACE potentials.

```bash
# Clone LAMMPS stable branch
git clone -b stable https://github.com/lammps/lammps.git mylammps
cd mylammps

# Create build directory
mkdir build && cd build

# Configure with CMake
# Ensure you point PYTHON_EXECUTABLE to the uv environment python
cmake ../cmake \
    -D PKG_PYTHON=ON \
    -D PKG_ML-PACE=ON \
    -D BUILD_SHARED_LIBS=ON \
    -D PYTHON_EXECUTABLE=$(uv python find)

# Build
make -j$(nproc)
make install

# Install or Set Library Path
export LD_LIBRARY_PATH=$PWD:$LD_LIBRARY_PATH
```

---

## Structure Generation

The **Seed Generation** phase (Phase 1) creates the initial training set if no potential exists. It is handled by `src/workflows/seed_generation.py` and includes:

1.  **Random Generation**: Uses `pyxtal` to generate diverse random crystal structures.
2.  **Scenario-Driven Generation**: Generates specific configurations like Surfaces, Interfaces, and Defects using `src/scenario_generation`.
3.  **Pre-optimization**: Relaxes structures using a foundational model (MACE) to remove unphysical high-energy states.
4.  **Sampling**: Selects the most diverse structures using `DirectSampler` (ACE descriptors + BIRCH clustering).

**Usage**: Automatically triggered by `src/main.py` if `initial_potential` is missing.

---

## Labeling (Delta Learning)

The project uses a **Delta-Learning** strategy implemented in `src/labeling/strategies/delta_labeler.py`.
The target for machine learning is the difference between DFT and a baseline empirical potential.

$$ E_{target} = E_{DFT} - E_{LJ} $$
$$ F_{target} = F_{DFT} - F_{LJ} $$

-   **Interface**: `DeltaLabeler` uses `ase.calculators.espresso.Espresso` for DFT and `ShiftedLennardJones` for the baseline.
-   **Parallelism**: Labeling tasks are executed in parallel using `ProcessPoolExecutor`, with each task running in an isolated temporary directory to prevent file conflicts.

---

## Training

Training is performed using **Pacemaker** via the `PacemakerTrainer` class in `src/training/strategies/pacemaker.py`.

-   **Pass-Through Configuration**: The system acts as a thin wrapper around Pacemaker. You can configure the potential architecture, loss weights, and fitting parameters directly in `config.yaml` under the `ace_model` section. The trainer handles file generation (`input.yaml`) and data management.
-   **Active Set Management**: Before retraining, the Basis Set (`.asi` file) is updated using `pace_activeset` to include new chemical environments.
-   **Multi-Stage Learning**: Supports transfer learning or mixing multiple base potentials by specifying a list of `initial_potentials` in the config.

---

## Simulation Modes

### Conventional MD
Handled by `LAMMPSRunner` in `src/engines/lammps/runner.py`.
-   Runs `lmp_serial` (or configured command) via `subprocess`.
-   Dynamically generates `in.lammps` input files.
-   **Uncertainty**: Monitors `fix pace/extrapolation`. If the extrapolation grade $\gamma$ exceeds `gamma_threshold`, the simulation halts, returning `SimulationState.UNCERTAIN`.

### Conventional kMC
Handled by `OffLatticeKMCEngine` in `src/engines/kmc.py`.
-   **Algorithm**: Off-Lattice kMC using the Dimer method for saddle point search.
-   **Graph-Based Cluster Move**: Identifies moving molecules/clusters using Numba-optimized BFS on CSR matrices (`scipy.sparse`).
-   **Parallel Search**: Dispatches multiple dimer searches in parallel processes.
-   **Uncertainty**: Checks $\gamma$ during the dimer search (FineTuna-style). If high uncertainty is found, the search aborts and triggers Active Learning.

### Hybrid MD/kMC Active Learning
The core workflow is orchestrated by `ActiveLearningOrchestrator` in `src/workflows/orchestrator.py`.

1.  **MD Phase**: Thermal sampling. If uncertainty -> Halt -> Train -> Resume.
2.  **MD Complete**: Pass final structure to kMC.
3.  **kMC Phase**: Rare event evolution. If uncertainty -> Halt -> Train -> Retry.
4.  **Loop**: Cycle continues indefinitely or until max steps.

---

## Usage

### 1. Activate Environment
```bash
source .venv/bin/activate
```

### 2. Configure (Drag & Drop)
This system separates **Physical/Experiment** settings from **Infrastructure/Environment** settings to maximize portability.

*   **Experiment Config (`config.yaml`)**: Defines physics (Temperature, Elements), Model Architecture (Pacemaker settings), and Experiment Metadata.
*   **Meta Config (`meta_config.yaml`)**: Defines environment paths (DFT executables, Pseudo directories) that change from machine to machine.

Edit these files or simply drop in your preferred versions.

### 3. Run
You can specify configuration paths via CLI arguments.

```bash
# Run with default configs (config.yaml, meta_config.yaml)
python src/main.py

# Run with specific configs (Drag & Drop equivalent)
python src/main.py --config experiments/exp01.yaml --meta env/workstation.yaml
```

### 4. Experiment Output
The system automatically creates an experiment directory defined in `config.yaml` (`experiment.output_dir`).

-   **Reproducibility**: `config.yaml` and `meta_config.yaml` are **automatically backed up** to this directory at the start of the run.
-   **Artifacts**:
    -   `data/`: Training data and potentials.
    -   `logs/`: detailed logs.
    -   `training_log.csv`: Metrics (RMSE, Gamma, Active Set Size).

---

## Configuration

### Example `config.yaml` (Physics & Model)
```yaml
experiment:
  name: "Al-Cu-Sintering"
  output_dir: "output/run_001"

ace_model:
  pacemaker_config:
    cutoff: 7.0
    potential:
      elements: ["Al", "Cu"]
      bonds:
        N: 3
        max_deg: 6
    fitting:
      weighting:
        energy: 100.0
        force: 1.0

md_params:
  timestep: 1.0
  temperature: 600.0
```

### Example `meta_config.yaml` (Environment)
```yaml
dft:
  command: "mpirun -np 32 pw.x"
  pseudo_dir: "/opt/pseudos"
  sssp_json_path: "/opt/pseudos/SSSP_precision.json"

lammps:
  command: "lmp_mpi"
```

---

## SSSP Pseudopotential Setup

This project uses the **SSSP (Standard Solid State Pseudopotentials)** database for automatic pseudopotential selection and DFT parameter configuration.

### 1. Download SSSP Pseudopotentials

Download the SSSP precision library:

```bash
# Create directory for pseudopotentials
mkdir -p ~/qe_calc/pp_precision
cd ~/qe_calc/pp_precision

# Download SSSP 1.3.0 PBE precision library
wget https://archive.materialscloud.org/record/file?filename=SSSP_1.3.0_PBE_precision.tar.gz&record_id=1677
tar -xzf SSSP_1.3.0_PBE_precision.tar.gz
```

The JSON metadata file (`SSSP_1.3.0_PBE_precision.json`) contains:
- Pseudopotential filenames for each element
- MD5 checksums for validation
- Recommended cutoff energies (`ecutwfc`, `ecutrho`)
- Pseudopotential type (NC/US/PAW)

### 2. Configure SSSP in config.yaml

```yaml
# In meta_config.yaml
dft:
  sssp_json_path: "/home/tomo/qe_calc/pp_precision/SSSP_1.3.0_PBE_precision.json"
  pseudo_dir: "/home/tomo/qe_calc/pp_precision"
  command: "mpirun -np 4 pw.x"

# In config.yaml
dft_params:
  kpoint_density: 60  # Å, for high precision (SSSP standard)
```
