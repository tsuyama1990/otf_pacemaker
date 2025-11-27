
import pytest
from unittest.mock import MagicMock, patch, ANY
from src.workflows.orchestrator import ActiveLearningOrchestrator, _run_md_task
from src.core.config import Config, ExplorationStage, MDParams
from src.core.enums import SimulationState
from pathlib import Path

@pytest.fixture
def mock_config():
    config = MagicMock(spec=Config)
    config.exploration_schedule = [
        ExplorationStage(iter_start=1, iter_end=5, temp=[300.0, 500.0], press=[1.0, 10.0])
    ]
    config.md_params = MagicMock(spec=MDParams)
    config.md_params.temperature = 300.0
    config.md_params.pressure = 1.0
    config.md_params.n_md_walkers = 4
    config.md_params.n_steps = 100
    config.al_params.gamma_threshold = 2.0
    config.al_params.num_parallel_labeling = 2
    return config

def test_get_md_conditions(mock_config):
    orchestrator = ActiveLearningOrchestrator(
        mock_config, MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock()
    )

    # Test inside schedule
    conditions = orchestrator._get_md_conditions(3)
    assert 300.0 <= conditions["temperature"] <= 500.0
    assert 1.0 <= conditions["pressure"] <= 10.0

    # Test outside schedule
    conditions = orchestrator._get_md_conditions(10)
    assert conditions["temperature"] == 300.0
    assert conditions["pressure"] == 1.0

def test_parallel_md_execution(mock_config):
    orchestrator = ActiveLearningOrchestrator(
        mock_config, MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock()
    )

    # Mock Executor
    with patch("src.workflows.orchestrator.ProcessPoolExecutor") as MockExecutor:
        mock_executor_instance = MockExecutor.return_value
        mock_executor_instance.__enter__.return_value = mock_executor_instance

        # Mock futures
        mock_future = MagicMock()
        mock_future.result.return_value = (SimulationState.COMPLETED, Path("dump.lammpstrj.1"))
        mock_executor_instance.submit.return_value = mock_future

        # Mock other dependencies to allow run() to proceed to MD block
        orchestrator._load_state = MagicMock(return_value={"iteration": 0})
        orchestrator._save_state = MagicMock()
        orchestrator._resolve_path = MagicMock(return_value=Path("potential.yace"))
        orchestrator._prepare_structure_path = MagicMock(return_value="structure.data")

        # Mock os.chdir to avoid actual directory changes
        with patch("os.chdir"), patch("pathlib.Path.mkdir"), patch("pathlib.Path.exists", return_value=True):
            # Run one iteration and break loop
            with patch("src.workflows.orchestrator.ActiveLearningOrchestrator._trigger_al") as mock_al:
                 # Force break after one loop
                 orchestrator.md_engine.run.side_effect = Exception("Stop Loop")
                 # Wait, we can't easily break the while True loop without exception or mocking logic.
                 # Let's rely on breaking when _load_state returns high iteration? No.
                 # Let's mock _save_state to raise StopIteration?
                 orchestrator._save_state.side_effect = [None, StopIteration]

                 try:
                    orchestrator.run()
                 except StopIteration:
                    pass

        # Verify submit was called n_md_walkers times
        assert mock_executor_instance.submit.call_count == 4

        # Verify args passed to submit
        # args: fn, md_engine, pot_path, steps, gamma, input, restart, temp, press, seed
        calls = mock_executor_instance.submit.call_args_list
        seen_seeds = set()
        for call in calls:
            args = call[0]
            # args[0] is function _run_md_task
            # args[1] is md_engine
            # ...
            # args[7] is temp
            # args[8] is press
            # args[9] is seed
            temp = args[7]
            press = args[8]
            seed = args[9]

            assert 300.0 <= temp <= 500.0
            assert 1.0 <= press <= 10.0
            seen_seeds.add(seed)

        assert len(seen_seeds) == 4 # Seeds should be unique (mostly)
