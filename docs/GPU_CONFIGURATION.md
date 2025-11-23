# GPU Configuration Fix for MACE

## Problem
The MACE calculator was using CPU despite GPU being available because:
1. The `MACEFilter` class defaulted to `device="cpu"`
2. No device parameter was passed when instantiating `MACEFilter` in `seed_generation.py`

## Solution Implemented

### 1. Updated `src/core/config.py`
- Added `device: str = "cuda"` to `GenerationParams` dataclass
- This allows configuration of the device used for MACE filtering

### 2. Updated `src/scenario_generation/filter.py`
- Changed default device parameter from `"cpu"` to `None`
- Added auto-detection logic: if `device=None`, automatically detects GPU availability using `torch.cuda.is_available()`
- Falls back to CPU if PyTorch is not available or no GPU is detected

### 3. Updated `src/workflows/seed_generation.py`
- Modified `MACEFilter` instantiation to pass `device=gen_params.device`
- Now respects the device setting from `config.yaml`

### 4. Updated `config.yaml`
- Added `generation` section with:
  ```yaml
  generation:
    device: "cuda"  # Device for MACE filtering
    pre_optimization:
      enabled: false
      model: "medium"
      fmax: 0.1
      steps: 50
      device: "cuda"
    scenarios: []
  ```

## How to Use

### Option 1: Explicit Configuration (Recommended)
Set the device in `config.yaml`:
```yaml
generation:
  device: "cuda"  # or "cpu"
```

### Option 2: Auto-Detection
Remove the `device` parameter from the config, and the code will automatically detect GPU availability.

## Verification

Run your script again and you should see:
```
Using CUDA
```
instead of:
```
Using CPU
```

You can verify GPU is being used by monitoring GPU usage:
```bash
watch -n 1 nvidia-smi
```

## Additional Notes

- The same pattern applies to `pre_optimization.device` for MACE pre-optimization
- Both default to `"cuda"` in the config
- The auto-detection ensures graceful fallback to CPU if GPU is unavailable
