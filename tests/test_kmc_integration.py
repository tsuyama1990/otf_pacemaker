"""Test imports and basic structure of new components."""
import sys
import pytest
from unittest.mock import MagicMock, patch

# Mock pyace
sys.modules["pyace"] = MagicMock()
sys.modules["pyace.PyACECalculator"] = MagicMock()

from src.core.enums import KMCStatus
from src.core.interfaces import KMCEngine, KMCResult
from src.core.config import KMCParams, ALParams
from src.engines.kmc import OffLatticeKMCEngine
from ase import Atoms

def test_kmc_engine_initialization():
    kmc_params = KMCParams(active=True, check_interval=5)
    al_params = ALParams(
        gamma_threshold=0.1,
        n_clusters=1,
        r_core=1.0,
        box_size=10.0,
        initial_potential="pot.yace",
        potential_yaml_path="pot.yaml"
    )
    engine = OffLatticeKMCEngine(kmc_params, al_params)
    assert engine.kmc_params.check_interval == 5
    assert engine.al_params.gamma_threshold == 0.1

@patch("src.engines.kmc.MinModeAtoms")
@patch("src.engines.kmc.DimerControl")
@patch("src.engines.kmc.FIRE")
def test_kmc_run_step_structure(mock_fire, mock_dimer_control, mock_min_mode):
    # Setup mocks
    kmc_params = KMCParams(active=True, n_searches=1, check_interval=5)
    al_params = ALParams(
        gamma_threshold=10.0, # High threshold to avoid early exit
        n_clusters=1,
        r_core=1.0,
        box_size=10.0,
        initial_potential="pot.yace",
        potential_yaml_path="pot.yaml"
    )
    engine = OffLatticeKMCEngine(kmc_params, al_params)

    atoms = Atoms("H2", positions=[[0,0,0], [1,0,0]])

    # Mock FIRE
    mock_opt_instance = MagicMock()
    mock_opt_instance.converged.return_value = True
    mock_fire.return_value = mock_opt_instance

    mock_min_mode_instance = MagicMock()
    mock_min_mode_instance.get_potential_energy.return_value = 1.0 # Saddle E
    mock_min_mode.return_value = mock_min_mode_instance

    # DummyExecutor to avoid multiprocessing
    class DummyExecutor:
        def __init__(self, max_workers=1): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def submit(self, fn, *args, **kwargs):
            res = fn(*args, **kwargs)
            f = MagicMock()
            f.result.return_value = res
            return f

    def dummy_as_completed(futures):
        return futures

    # Run
    with patch("src.engines.kmc.ProcessPoolExecutor", side_effect=DummyExecutor), \
         patch("src.engines.kmc.as_completed", side_effect=dummy_as_completed), \
         patch("src.engines.kmc.PyACECalculator") as mock_calc_cls:

        # Mock calculator
        mock_calc = MagicMock()
        mock_calc.results.get.return_value = None
        mock_calc.get_potential_energy.return_value = 0.0 # Initial E
        mock_calc_cls.return_value = mock_calc

        result = engine.run_step(atoms, "pot.yace")

    # Barrier = 1.0 - 0.0 = 1.0 > 0.01. So event is found.
    assert isinstance(result, KMCResult)
    assert result.status in [KMCStatus.SUCCESS, KMCStatus.NO_EVENT]

@patch("src.engines.kmc.MinModeAtoms")
@patch("src.engines.kmc.DimerControl")
@patch("src.engines.kmc.FIRE")
def test_kmc_parallel_execution(mock_fire, mock_dimer_control, mock_min_mode):
    """Test that parallel execution path runs without error."""
    # We use n_workers=2
    kmc_params = KMCParams(active=True, n_searches=2, n_workers=2)
    al_params = ALParams(
        gamma_threshold=10.0,
        n_clusters=1,
        r_core=1.0,
        box_size=10.0,
        initial_potential="pot.yace",
        potential_yaml_path="pot.yaml"
    )
    engine = OffLatticeKMCEngine(kmc_params, al_params)

    atoms = Atoms("H2", positions=[[0,0,0], [1,0,0]])

    # Mock FIRE
    mock_opt = MagicMock()
    mock_opt.converged.return_value = True
    mock_fire.return_value = mock_opt

    mock_min_mode.return_value.get_potential_energy.return_value = 1.0

    class DummyExecutor:
        def __init__(self, max_workers=1): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def submit(self, fn, *args, **kwargs):
            # The real submit calls fn(*args, **kwargs).
            res = fn(*args, **kwargs)
            f = MagicMock()
            f.result.return_value = res
            return f

    def dummy_as_completed(futures):
        return futures

    with patch("src.engines.kmc.ProcessPoolExecutor", side_effect=DummyExecutor), \
         patch("src.engines.kmc.as_completed", side_effect=dummy_as_completed), \
         patch("src.engines.kmc.PyACECalculator") as mock_calc_cls:

        mock_calc = MagicMock()
        mock_calc.results.get.return_value = None
        mock_calc.get_potential_energy.return_value = 0.0
        mock_calc_cls.return_value = mock_calc

        result = engine.run_step(atoms, "pot.yace")

        # Should run successfully
        assert isinstance(result, KMCResult)

@patch("src.engines.kmc.MinModeAtoms")
@patch("src.engines.kmc.DimerControl")
@patch("src.engines.kmc.FIRE")
def test_active_region_selection(mock_fire, mock_dimer_control, mock_min_mode):
    from src.engines.kmc import _select_active_atoms

    # Setup atoms: 2 atoms, one at z=0, one at z=15
    atoms = Atoms("CoTi", positions=[[0,0,0], [0,0,15]])

    # 1. Surface mode
    params = KMCParams(active_region_mode="surface", active_z_cutoff=10.0)
    indices = _select_active_atoms(atoms, params)
    assert len(indices) == 1
    assert indices[0] == 1 # The one at z=15

    # 2. Species mode
    params = KMCParams(active_region_mode="species", active_species=["Co"])
    indices = _select_active_atoms(atoms, params)
    assert len(indices) == 1
    assert indices[0] == 0 # Co is at index 0

    # 3. Both
    params = KMCParams(active_region_mode="surface_and_species", active_z_cutoff=10.0, active_species=["Ti"])
    indices = _select_active_atoms(atoms, params)
    assert len(indices) == 1
    assert indices[0] == 1 # Ti at z=15

@patch("src.engines.kmc.MinModeAtoms")
@patch("src.engines.kmc.DimerControl")
@patch("src.engines.kmc.FIRE")
def test_kmc_high_gamma_interruption(mock_fire, mock_dimer_control, mock_min_mode):
    # Set active_region_mode="all" (or just ensure defaults pass)
    # Default is "surface_and_species" with Co,Ti,O and z>10.
    # Our atoms are H2 at z=0. So we must override.
    kmc_params = KMCParams(
        active=True, n_searches=1, check_interval=2,
        active_region_mode="species", active_species=["H"]
    )
    al_params = ALParams(
        gamma_threshold=0.5,
        n_clusters=1,
        r_core=1.0,
        box_size=10.0,
        initial_potential="pot.yace",
        potential_yaml_path="pot.yaml"
    )
    engine = OffLatticeKMCEngine(kmc_params, al_params)

    atoms = Atoms("H2", positions=[[0,0,0], [1,0,0]])

    # Mock FIRE to NOT converge immediately, so loop runs
    mock_opt_instance = MagicMock()
    mock_opt_instance.converged.return_value = False
    mock_fire.return_value = mock_opt_instance

    class DummyExecutor:
        def __init__(self, max_workers=1): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def submit(self, fn, *args, **kwargs):
            try:
                res = fn(*args, **kwargs)
                f = MagicMock()
                f.result.return_value = res
                return f
            except Exception as e:
                # If execution fails, return a future that raises exception
                f = MagicMock()
                f.result.side_effect = e
                return f

    def dummy_as_completed(futures):
        return futures

    with patch("src.engines.kmc.ProcessPoolExecutor", side_effect=DummyExecutor), \
         patch("src.engines.kmc.as_completed", side_effect=dummy_as_completed), \
         patch("src.engines.kmc.PyACECalculator") as mock_calc_cls:

        # Mock calculator to return high gamma
        mock_calc = MagicMock()
        mock_calc.results.get.return_value = [1.0, 0.2] # Max 1.0 > 0.5
        mock_calc_cls.return_value = mock_calc

        result = engine.run_step(atoms, "pot.yace")

        # Verify status
        assert result.status == KMCStatus.UNCERTAIN
