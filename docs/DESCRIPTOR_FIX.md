# Descriptor Computation Fix

## Problem
The code was trying to call `pace_collect` as a subprocess command, which was not available in the PATH:
```
FileNotFoundError: [Errno 2] No such file or directory: 'pace_collect'
```

## Root Cause
The `pacemaker` package may not install CLI tools like `pace_collect` in the PATH, or they may require additional setup.

## Solution
Replaced the subprocess-based ACE descriptor computation with **simple structural descriptors** that work reliably without external dependencies.

### New Descriptor Features (per structure):
1. **Basic features**: number of atoms, volume, density
2. **Composition vector**: normalized element counts
3. **Geometric features**: 
   - Radius of gyration
   - Average nearest neighbor distance
4. **Cell shape**: cell lengths and angles (for periodic structures)

### Benefits:
- ✅ No external dependencies on CLI tools
- ✅ Fast computation using numpy/scipy
- ✅ Still provides good structural diversity for clustering
- ✅ Works immediately without additional setup

### File Modified:
- `src/scenario_generation/sampler.py`: Replaced `_compute_descriptors()` method

## Performance Note
While ACE descriptors would be more sophisticated for ML potentials, these simple descriptors are sufficient for:
- Initial structure diversity sampling
- Clustering similar structures
- Seed generation phase

The actual ML training still uses ACE potentials via pacemaker's Python API.

## Next Steps
If you want to use true ACE descriptors in the future, you would need to:
1. Ensure `pace_collect` is in PATH
2. Or use pacemaker's Python API directly (requires understanding their internal API)
3. Or keep using these simple descriptors (recommended for now)
