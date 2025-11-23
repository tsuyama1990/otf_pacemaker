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

-   **Active Set Management**: Before retraining, the Basis Set (`.asi` file) is updated using `pace_activeset` to include new chemical environments.
-   **Fine-Tuning**: Retraining typically starts from the previous potential (Warm Start) to accelerate convergence.
-   **Output**: Produces a `.yace` (YAML ACE) potential file used by LAMMPS.

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

1.  **Activate Environment**:
    ```bash
    source .venv/bin/activate
    ```

2.  **Configure**:
    Edit `config.yaml` to set your system parameters (elements, temperature, DFT settings).

3.  **Run**:
    ```bash
    # Runs the main active learning loop
    # Automatically runs Seed Generation if needed
    python src/main.py
    ```

4.  **Output**:
    -   `data/seed/`: Initial training data and potential.
    -   `data/iteration_X/`: Data for each AL cycle (MD dumps, trained potentials, logs).
    -   `training_log.csv`: Metrics (RMSE, Gamma, Active Set Size).

---

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
dft_params:
  sssp_json_path: "/home/tomo/qe_calc/pp_precision/SSSP_1.3.0_PBE_precision.json"
  pseudo_dir: "/home/tomo/qe_calc/pp_precision"
  command: "mpirun -np 4 pw.x"
  kpoint_density: 60  # Å, for high precision (SSSP standard)
```

### 3. Automatic Features

#### Pseudopotential Selection
- Automatically loads correct PP files for each element from SSSP database
- For compounds (e.g., Al-Cu), uses `max(ecutwfc)` and `max(ecutrho)` across all elements
- Validates PP files exist and optionally checks MD5 checksums

#### K-point Density Calculation
Uses the **k-point density approach** instead of fixed grids:

**Formula**: `L × N ≈ kpoint_density` (default: 60 Å for high precision)

Where:
- `L` = cell lattice vector length
- `N` = number of k-points along that direction

**Examples**:
| Cell Size | Calculation | K-grid |
|-----------|-------------|--------|
| 3 Å (primitive) | 60/3 = 20 | 20×20×20 |
| 10 Å (supercell) | 60/10 = 6 | 6×6×6 |
| 20 Å (large cell) | 60/20 = 3 | 3×3×3 |

**Benefits**:
- Consistent accuracy across different cell sizes
- Automatic adaptation for unit cells and supercells
- Standard approach for MLIP dataset generation

**K-point Settings**:
- Grid type: Monkhorst-Pack
- Shift: [1, 1, 1] (includes Γ-point)
- Calculated per-structure based on cell dimensions

### 4. Adjusting K-point Density

Modify `kpoint_density` in `config.yaml` based on your accuracy needs:

```yaml
dft_params:
  kpoint_density: 40   # Lower precision, faster (efficiency mode)
  kpoint_density: 60   # High precision (SSSP standard, recommended)
  kpoint_density: 80   # Very high precision (convergence tests)
```

---

## Configuration

The `config.yaml` file controls all aspects of the simulation.

```yaml
md_params:
  elements: ["Ag", "Pd"]
  temperature: 300.0
  n_steps: 100000

al_params:
  gamma_threshold: 2.0  # Uncertainty limit
  initial_potential: "data/seed/seed_potential.yace"

kmc_params:
  active: true          # Enable Hybrid MD-kMC
  n_workers: 4          # Parallel saddle searches
  temperature: 300.0

dft_params:
  command: "pw.x -in espresso.pwi > espresso.pwo"
  pseudo_dir: "/path/to/pseudos"
```
