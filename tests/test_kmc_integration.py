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

    # Mock MinModeAtoms to return energy for barrier check
    mock_min_mode_instance = MagicMock()
    mock_min_mode_instance.get_potential_energy.return_value = 1.0 # Saddle E
    # We also need atoms to have E
    # engine.run_step calls atoms.get_potential_energy() via ase calc (mocked) or directly.
    # Since we mocked the calc, we should mock get_potential_energy on atoms or calc.
    # But MinModeAtoms is mocked, so dimer_atoms.get_potential_energy() returns MagicMock.

    mock_min_mode.return_value = mock_min_mode_instance

    # Run
    with patch("src.engines.kmc.PyACECalculator") as mock_calc_cls:
        # Mock calculator
        mock_calc = MagicMock()
        mock_calc.results.get.return_value = None
        mock_calc.get_potential_energy.return_value = 0.0 # Initial E
        mock_calc_cls.return_value = mock_calc

        # We need to make sure the atoms copy in run_step gets this mock calc.
        # And that get_potential_energy returns float.

        # When engine calls atoms.get_potential_energy(), it calls calc.get_potential_energy().

        result = engine.run_step(atoms, "pot.yace")

    # Barrier = 1.0 - 0.0 = 1.0 > 0.01. So event is found.

    assert isinstance(result, KMCResult)
    # Since we found an event, status should be SUCCESS (or NO_EVENT if rate calc fails/randomness)
    # Rate calc: barrier=1.0. T=300. Rate ~ exp(-40) ~ very small.
    # But random choice might select it if it's the only one.
    assert result.status in [KMCStatus.SUCCESS, KMCStatus.NO_EVENT]

@patch("src.engines.kmc.MinModeAtoms")
@patch("src.engines.kmc.DimerControl")
@patch("src.engines.kmc.FIRE")
def test_kmc_high_gamma_interruption(mock_fire, mock_dimer_control, mock_min_mode):
    kmc_params = KMCParams(active=True, n_searches=1, check_interval=2)
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

    with patch("src.engines.kmc.PyACECalculator") as mock_calc_cls:
        # Mock calculator to return high gamma
        mock_calc = MagicMock()
        mock_calc.results.get.return_value = [1.0, 0.2] # Max 1.0 > 0.5
        mock_calc_cls.return_value = mock_calc

        result = engine.run_step(atoms, "pot.yace")

    assert result.status == KMCStatus.UNCERTAIN
