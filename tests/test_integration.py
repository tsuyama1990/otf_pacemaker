import pytest
from unittest.mock import MagicMock, patch, call
from pathlib import Path
import sys

# We need to verify that the loop runs: Uncertainty -> Extract -> Label -> Train -> Resume
# We'll mock the components in src.main

def test_active_learning_loop(tmp_path):
    # 1. Setup Mocks

    # Mock Config
    mock_config = MagicMock()
    mock_config.lj_params.epsilon = 1.0
    mock_config.lj_params.sigma = 1.0
    mock_config.lj_params.cutoff = 2.5

    mock_config.md_params.elements = ['Ag', 'Pd']
    mock_config.md_params.timestep = 0.001
    mock_config.md_params.temperature = 300
    mock_config.md_params.pressure = 1.0
    mock_config.md_params.restart_freq = 1000
    mock_config.md_params.n_steps = 100
    # Use absolute paths
    mock_config.md_params.initial_structure = str(tmp_path / "start.data")

    mock_config.al_params.r_core = 4.0
    mock_config.al_params.r_buffer = 2.0
    mock_config.al_params.gamma_threshold = 0.1
    mock_config.al_params.n_clusters = 1
    mock_config.al_params.initial_potential = str(tmp_path / "pot.yace")

    mock_config.dft_params.pseudo_dir = "."
    mock_config.dft_params.ecutwfc = 30
    mock_config.dft_params.kpts = (1,1,1)

    # Mock classes
    with patch('src.main.Config') as MockConfigCls, \
         patch('src.main.LAMMPSRunner') as MockRunnerCls, \
         patch('src.main.ClusterCarver') as MockCarverCls, \
         patch('src.main.DeltaLabeler') as MockLabelerCls, \
         patch('src.main.PacemakerTrainer') as MockTrainerCls, \
         patch('src.main.read') as mock_ase_read, \
         patch('ase.calculators.espresso.Espresso') as MockEspresso:

        # Configure Config.from_yaml
        MockConfigCls.from_yaml.return_value = mock_config

        # Configure Espresso
        MockEspresso.return_value = MagicMock()

        # Configure LAMMPSRunner instance
        mock_runner = MockRunnerCls.return_value
        # Sequence of states: UNCERTAIN (loop 1), then COMPLETED (loop 2 to exit)
        from src.md_engine import SimulationState
        mock_runner.run_md.side_effect = [SimulationState.UNCERTAIN, SimulationState.COMPLETED]

        # Configure ClusterCarver
        mock_carver = MockCarverCls.return_value
        mock_carver.extract_cluster.return_value = MagicMock() # Returns a dummy atom object

        # Configure DeltaLabeler
        mock_labeler = MockLabelerCls.return_value
        mock_labeler.compute_delta.return_value = MagicMock() # Returns labeled atoms

        # Configure PacemakerTrainer
        mock_trainer = MockTrainerCls.return_value
        mock_trainer.prepare_dataset.return_value = "dataset.pckl"

        def train_side_effect(*args, **kwargs):
            Path("new_pot.yace").touch()
            return "new_pot.yace"

        mock_trainer.train.side_effect = train_side_effect

        # Configure ASE read (Snapshot loading)
        # Must return an Atoms object with 'type' array for symbol mapping logic
        mock_atoms = MagicMock()
        mock_atoms.__len__.return_value = 10
        # Mock get_array('type')
        # Our code: types = atoms.get_array('type')
        import numpy as np
        mock_atoms.get_array.return_value = np.array([1, 2] * 5)
        # Mock set_chemical_symbols (called in main)

        mock_ase_read.return_value = mock_atoms

        # Prepare file system mocks
        # The code checks for 'config.yaml' existence
        # And creates 'data/iteration_X'
        # And checks 'dump.lammpstrj'
        # And 'potential' file existence

        # We can use pyfakefs or just mock Path.exists.
        # But 'src.main' uses 'Path("config.yaml").exists()'.
        # Since we are patching classes, we might need to patch Path used in src.main.
        # But Path is imported directly.

        # Easier approach: Create the files in tmp_path and run main from there?
        # But main code does os.chdir.

        # Let's rely on side_effect of 'exists' if we patch Path? No, too risky.
        # Let's set up the actual file system in tmp_path and change CWD before importing/running main.

        # We already imported src.main.
        # Let's ensure we are in tmp_path.

        import os
        os.chdir(tmp_path)
        Path("config.yaml").touch()
        Path("start.data").touch()
        Path("pot.yace").touch()
        # We need dump.lammpstrj inside data/iteration_1?
        # No, the code runs MD in data/iteration_1, so dump is there.
        # We need to ensure when main checks dump_file.exists(), it returns True.
        # The code does: dump_file = Path("dump.lammpstrj"); if not dump_file.exists()...
        # Since we are mocking runner, the file won't be created by LAMMPS.
        # So we must pre-create it?
        # But main creates the directory `data/iteration_1` then chdirs into it.
        # So we can't pre-create the file inside the folder that doesn't exist yet (or is created by main).

        # Wait, if we mock `os.chdir`, we stay in root.
        # But main logic relies on chdir.

        # Strategy: Use `side_effect` on `runner.run_md` to create the dump file!
        # Note: side_effect list elements are returned as is, not called (unless side_effect ITSELF is a callable).
        # So we use a single callable side_effect that handles the logic.

        def run_md_side_effect(*args, **kwargs):
            # If this is the first call (we can check call_count or just use a counter)
            # Since we mock run_md, we can rely on checking if dump file exists,
            # or just use a closure attribute.

            if not hasattr(run_md_side_effect, "called"):
                run_md_side_effect.called = True
                # First call: Create dump and return UNCERTAIN
                Path("dump.lammpstrj").touch()
                # Also create restart.chk because the next iteration expects it for resuming
                Path("restart.chk").touch()
                return SimulationState.UNCERTAIN
            else:
                return SimulationState.COMPLETED

        mock_runner.run_md.side_effect = run_md_side_effect

        # Import and run main
        from src.main import main

        # We need to make sure the potential file logic works.
        # code: abs_potential_path = Path(current_potential).resolve()
        # if not abs_potential_path.exists(): break

        main()

        # 2. Verify Assertions

        # Check if MD was run twice
        assert mock_runner.run_md.call_count == 2

        # First run: initial potential, not restart
        args1, kwargs1 = mock_runner.run_md.call_args_list[0]
        assert kwargs1['is_restart'] is False
        assert "pot.yace" in kwargs1['potential_path'] # Resolves to absolute path

        # Second run: resumed (is_restart=True), new potential
        args2, kwargs2 = mock_runner.run_md.call_args_list[1]
        assert kwargs2['is_restart'] is True
        assert "new_pot.yace" in kwargs2['potential_path']

        # Check if extraction happened
        mock_carver.extract_cluster.assert_called()

        # Check if labeling happened
        mock_labeler.compute_delta.assert_called()

        # Check if training happened
        mock_trainer.prepare_dataset.assert_called()
        mock_trainer.train.assert_called_once()

        # Check if train called with new dataset and old potential (for fine tuning)
        # trainer.train(dataset_path, str(abs_potential_path))
        call_args = mock_trainer.train.call_args
        assert "dataset.pckl" in str(call_args[0][0])
        assert "pot.yace" in str(call_args[0][1])
