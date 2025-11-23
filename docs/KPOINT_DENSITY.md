# SSSP K-point Density Approach

## Overview

This project uses the **k-point density approach** for automatic k-point grid calculation in DFT calculations. This is the modern standard for machine learning potential (MLIP) dataset generation, as it ensures consistent accuracy across structures with varying cell sizes.

## Theory

### The Rule
**Cell length (L) × Number of k-points (N) ≈ constant**

This ensures constant density in reciprocal space (k-space), which directly relates to the accuracy of the electronic structure calculation.

### Recommended Densities

| Precision Level | Density (Å) | Use Case |
|----------------|-------------|----------|
| Efficiency | 40 | Quick testing, low-accuracy screening |
| **Standard (SSSP)** | **60** | **High precision, recommended for MLIP** |
| Convergence | 80 | Very high precision, convergence tests |

## Implementation

### Automatic Calculation

The `calculate_kpoints()` function in `src/utils/sssp_loader.py` automatically computes k-grids:

```python
from src.utils.sssp_loader import calculate_kpoints

# For a structure with cell lengths [10, 10, 15] Å
kpts, shift = calculate_kpoints(atoms.cell, kpoint_density=60)
# Returns: kpts=(6, 6, 4), shift=(1, 1, 1)
```

### Formula

For each lattice vector direction:

```
N_k = max(1, round(kpoint_density / L))
```

Where:
- `N_k` = number of k-points along direction
- `kpoint_density` = target density in Å (default: 60)
- `L` = length of lattice vector in Å

### Examples

#### Small Primitive Cell (3 Å)
```
L = 3 Å
N = round(60 / 3) = 20
K-grid: 20 × 20 × 20
```

#### Medium Supercell (10 Å)
```
L = 10 Å
N = round(60 / 10) = 6
K-grid: 6 × 6 × 6
```

#### Large MD Cell (20 Å)
```
L = 20 Å
N = round(60 / 20) = 3
K-grid: 3 × 3 × 3
```

## Monkhorst-Pack Settings

All k-point grids use:
- **Grid type**: Monkhorst-Pack
- **Shift**: [1, 1, 1]
- **Effect**: Includes the Γ-point (k=0) in the grid

This is the standard choice for most materials and ensures good sampling of the Brillouin zone.

## Configuration

Set in `config.yaml`:

```yaml
dft_params:
  kpoint_density: 60  # Å, adjust based on precision needs
```

## Advantages Over Fixed Grids

### Traditional Approach (Fixed Grid)
```yaml
# Problem: Same grid for all cell sizes
kpts: [4, 4, 4]  # Too coarse for small cells, overkill for large cells
```

### K-point Density Approach
```python
# Automatic adaptation:
# 3 Å cell  → 20×20×20 (fine grid)
# 20 Å cell → 3×3×3 (coarse grid)
```

**Benefits**:
1. ✅ Consistent accuracy across different cell sizes
2. ✅ Automatic for unit cells, supercells, and MD snapshots
3. ✅ Standard in MLIP community (MACE, NequIP, etc.)
4. ✅ Prevents over/under-sampling

## References

- SSSP (Standard Solid State Pseudopotentials): [Materials Cloud](https://www.materialscloud.org/discover/sssp)
- Monkhorst-Pack grids: [Phys. Rev. B 13, 5188 (1976)](https://doi.org/10.1103/PhysRevB.13.5188)
- K-point convergence: [Quantum ESPRESSO tutorial](https://www.quantum-espresso.org/resources/tutorials)
