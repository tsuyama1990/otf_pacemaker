import pytest
from unittest.mock import MagicMock, patch, ANY
from pathlib import Path
from ase import Atoms
import numpy as np

from src.sampling.strategies.max_vol import MaxVolSampler
from src.workflows.orchestrator import ActiveLearningOrchestrator
from src.core.config import Config, MDParams, ALParams, DFTParams, LJParams
from src.core.enums import SimulationState
from src.core.interfaces import MDEngine, StructureGenerator, Labeler, Trainer

@pytest.fixture
def mock_atoms():
    atoms = Atoms('H2', positions=[[0, 0, 0], [0, 0, 1]])
    atoms.arrays['f_f_gamma'] = np.array([0.1, 0.9])
    atoms.arrays['type'] = np.array([1, 1])
    return atoms

@pytest.fixture
def max_vol_sampler():
    return MaxVolSampler(pace_select_cmd="mock_pace_select")

def test_max_vol_sampler_missing_kwargs(max_vol_sampler):
    with pytest.raises(ValueError):
        max_vol_sampler.sample(dump_file="dump.lammpstrj")

@patch('src.sampling.strategies.max_vol.read')
@patch('src.sampling.strategies.max_vol.write')
@patch('src.sampling.strategies.max_vol.subprocess.run')
def test_max_vol_sampler_success(mock_run, mock_write, mock_read, max_vol_sampler, mock_atoms, tmp_path):
    # Setup mocks
    mock_read.side_effect = [
        [mock_atoms], # First read (dump file frames)
        [mock_atoms]  # Second read (selected frames)
    ]

    # Mock file existence checks
    with patch('src.sampling.strategies.max_vol.Path.exists', return_value=True):
         samples = max_vol_sampler.sample(
             dump_file="dump.lammpstrj",
             potential_yaml_path="potential.yaml",
             asi_path="potential.asi",
             n_clusters=5,
             elements=["H"]
         )

    # Verification
    assert len(samples) == 1
    # Check if correct atom selected (index 1 has gamma 0.9)
    assert samples[0][1] == 1

    # Check subprocess call
    mock_run.assert_called_once()
    args, _ = mock_run.call_args
    cmd_list = args[0]
    assert cmd_list[0] == "mock_pace_select"
    assert "-p" in cmd_list
    assert "-a" in cmd_list
    assert "-m" in cmd_list

@patch('src.workflows.orchestrator.os.chdir')
@patch('src.workflows.orchestrator.Path.mkdir')
@patch('src.workflows.orchestrator.Path.exists')
@patch('src.workflows.orchestrator.read')
def test_orchestrator_initial_asi_generation(mock_read, mock_exists, mock_mkdir, mock_chdir):
    # Setup Config
    config = Config(
        md_params=MDParams(
            timestep=1.0, temperature=300, pressure=1.0, n_steps=100,
            elements=["Al"], initial_structure="start.xyz", masses={"Al": 26.98}
        ),
        al_params=ALParams(
            gamma_threshold=0.1, n_clusters=2, r_core=3.0, box_size=10.0,
            initial_potential="pot.yace", potential_yaml_path="pot.yaml",
            initial_dataset_path="data.pckl", initial_active_set_path=None
        ),
        dft_params=DFTParams(ecutwfc=30, kpts=(1,1,1), pseudo_dir=".", command="pw.x"),
        lj_params=LJParams(epsilon=1.0, sigma=1.0, cutoff=2.5)
    )

    # Mocks
    md_engine = MagicMock(spec=MDEngine)
    sampler = MagicMock()
    generator = MagicMock(spec=StructureGenerator)
    labeler = MagicMock(spec=Labeler)
    trainer = MagicMock(spec=Trainer)

    # Simulate loop breaking to avoid infinite loop
    md_engine.run.side_effect = [SimulationState.COMPLETED]

    # Initial file check behavior
    def exists_side_effect(self):
        # We need potential and structure to exist
        if str(self).endswith("pot.yace"): return True
        if str(self).endswith("start.xyz"): return True
        return False

    # We patch exists on the class, but side_effect needs to handle instance
    # This is tricky with patch('pathlib.Path.exists').
    # Let's simplify and just rely on the logic flow.

    orch = ActiveLearningOrchestrator(config, md_engine, sampler, generator, labeler, trainer)

    # We want to verify that trainer.update_active_set is called
    # when initial_active_set_path is None and initial_dataset_path is set.

    # To run just the init part, we can't easily stop __init__ or run,
    # but run() does the check at start.

    # Mocking resolve_path to return dummy paths
    orch._resolve_path = MagicMock(side_effect=lambda x, y: Path(x))

    # Mock trainer.update_active_set
    trainer.update_active_set.return_value = "generated.asi"

    # Run
    try:
        orch.run()
    except Exception:
        pass # Expected since we mocked things loosely

    # Verify update_active_set called
    trainer.update_active_set.assert_called_once_with("data.pckl", "pot.yaml")
