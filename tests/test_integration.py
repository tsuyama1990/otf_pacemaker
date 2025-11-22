import pytest
from unittest.mock import MagicMock, patch, call
from pathlib import Path
import sys
from src.workflow import ActiveLearningWorkflow
from src.enums import SimulationState

def test_workflow(tmp_path):
    # Setup Mocks for Components
    mock_config = MagicMock()
    mock_config.md_params.initial_structure = "start.data"
    mock_config.md_params.n_steps = 100
    mock_config.md_params.elements = ['Ag']
    mock_config.al_params.initial_potential = "pot.yace"
    mock_config.al_params.gamma_threshold = 0.1
    mock_config.al_params.n_clusters = 1

    mock_engine = MagicMock()
    # Sequence: Uncertain -> Completed
    mock_engine.run.side_effect = [SimulationState.UNCERTAIN, SimulationState.COMPLETED]

    mock_sampler = MagicMock()
    mock_sampler.sample.return_value = [0]

    mock_generator = MagicMock()
    mock_generator.generate.return_value = MagicMock()

    mock_labeler = MagicMock()
    mock_labeler.label.return_value = MagicMock()

    mock_trainer = MagicMock()
    mock_trainer.prepare_dataset.return_value = "data.pckl"

    def train_side_effect(*args, **kwargs):
        Path("new_pot.yace").touch()
        return "new_pot.yace"

    mock_trainer.train.side_effect = train_side_effect

    # Setup Filesystem
    import os
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        Path("start.data").touch()
        Path("pot.yace").touch()
        Path("new_pot.yace").touch() # Ensure existence for verification

        workflow = ActiveLearningWorkflow(
            config=mock_config,
            md_engine=mock_engine,
            sampler=mock_sampler,
            generator=mock_generator,
            labeler=mock_labeler,
            trainer=mock_trainer
        )

        # We need to handle the dump file and restart file existence for the loop logic
        # since we mock the engine, the engine won't create them.
        # Workflow logic checks for dump file in _handle_uncertainty.

        # We can mock Path.exists or pre-create files in the iteration folders?
        # Workflow creates 'data/iteration_X'.
        # Engine runs there.
        # We need 'dump.lammpstrj' to exist when UNCERTAIN happens.

        # We can add a side_effect to engine.run that creates the files

        def run_side_effect(*args, **kwargs):
            # Create dump file in current directory (which is data/iter_X)
            Path("dump.lammpstrj").touch()
            # Create restart file for next iteration
            Path("restart.chk").touch()

            # Return state based on call count logic we set up via .side_effect list in the mock
            # But here we are overriding it.
            # Let's use an iterator
            if not hasattr(run_side_effect, 'iter'):
                run_side_effect.iter = iter([SimulationState.UNCERTAIN, SimulationState.COMPLETED])
            return next(run_side_effect.iter)

        mock_engine.run.side_effect = run_side_effect

        # Mock ase.io.read because we read the dump file
        with patch('src.workflow.read') as mock_read:
            mock_atoms = MagicMock()
            mock_atoms.get_array.return_value = [1] # types
            mock_read.return_value = mock_atoms

            workflow.run()

        # Verification
        assert mock_engine.run.call_count == 2

        # Check first run
        args1, kwargs1 = mock_engine.run.call_args_list[0]
        assert kwargs1['is_restart'] is False
        assert "pot.yace" in kwargs1['potential_path']

        # Check second run
        args2, kwargs2 = mock_engine.run.call_args_list[1]
        assert kwargs2['is_restart'] is True
        assert "new_pot.yace" in kwargs2['potential_path']

        mock_sampler.sample.assert_called_once()
        mock_generator.generate.assert_called_once()
        mock_labeler.label.assert_called_once()
        mock_trainer.train.assert_called_once()

    finally:
        os.chdir(original_cwd)
