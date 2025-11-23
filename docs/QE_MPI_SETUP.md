# Quantum Espresso MPI Configuration

## Current Status
✅ **MPI is now enabled** for Quantum Espresso in `config.yaml`

## Configuration

### Command Format
```yaml
dft_params:
  command: "mpirun -np 4 pw.x"
```

This will run `pw.x` with 4 MPI processes in parallel.

### Adjusting Number of Processes

The `-np 4` means 4 MPI processes. Adjust based on:

1. **Your CPU cores**: Check with `nproc` or `lscpu`
2. **System size**: Small systems (< 50 atoms) may not benefit from many cores
3. **Memory**: Each process needs memory

**Recommended values**:
- Small systems (< 20 atoms): `-np 2` to `-np 4`
- Medium systems (20-50 atoms): `-np 4` to `-np 8`
- Large systems (> 50 atoms): `-np 8` to `-np 16`

### Alternative MPI Launchers

Depending on your system, you might need different launchers:

```yaml
# OpenMPI (most common)
command: "mpirun -np 4 pw.x"

# Intel MPI
command: "mpiexec -n 4 pw.x"

# SLURM (on HPC clusters)
command: "srun -n 4 pw.x"

# With specific binding (for performance)
command: "mpirun -np 4 --bind-to core pw.x"
```

## Parallel Labeling

Note that `config.yaml` also has:
```yaml
al_params:
  num_parallel_labeling: 4
```

This controls **how many DFT calculations run simultaneously**. Each will use the MPI processes specified in `command`.

**Example**:
- `num_parallel_labeling: 4`
- `command: "mpirun -np 4 pw.x"`
- **Total cores used**: 4 × 4 = 16 cores

Make sure this doesn't exceed your available cores!

## Verification

To check if MPI is working, look for these in the QE output:
```
Parallel version (MPI), running on     4 processors
```

## Troubleshooting

### "mpirun not found"
Install MPI:
```bash
sudo apt-get install openmpi-bin libopenmpi-dev
```

### "pw.x not found"
Ensure Quantum Espresso is installed and in PATH:
```bash
which pw.x
# or specify full path:
command: "mpirun -np 4 /usr/bin/pw.x"
```

### Performance Issues
- Reduce `-np` if you see poor scaling
- Check CPU affinity with `--bind-to core`
- Monitor with `htop` to see if all cores are utilized

## Before Running

Make sure you also:
1. ✅ Install Quantum Espresso
2. ✅ Download pseudopotentials
3. ✅ Update `pseudo_dir` in config.yaml to actual path
4. ✅ Ensure pseudopotential files exist (e.g., `Al.UPF`, `Cu.UPF`)
