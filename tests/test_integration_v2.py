import unittest
from unittest.mock import MagicMock, patch
import logging
import sys

# Mock huge parts of the system before importing Orchestrator
sys.modules["pyace"] = MagicMock()
sys.modules["pacemaker"] = MagicMock()

# Set logging to critical to silence logs during tests
logging.basicConfig(level=logging.CRITICAL)

from ase import Atoms
from src.core.config import Config
from src.core.interfaces import MDEngine, KMCEngine, Sampler, StructureGenerator, Labeler, Trainer
from src.core.enums import SimulationState, KMCStatus
from src.workflows.orchestrator import ActiveLearningOrchestrator

class TestIntegrationV2(unittest.TestCase):
    def setUp(self):
        # Create Dummy Config
        self.config = Config.from_dict({
            "md_params": {
                "timestep": 1.0, "temperature": 300, "pressure": 0, "n_steps": 100,
                "elements": ["Ag"], "initial_structure": "dummy.xyz", "masses": {"Ag": 107.87}
            },
            "al_params": {
                "gamma_threshold": 0.1, "n_clusters": 2, "r_core": 3.0, "box_size": 10.0,
                "initial_potential": "pot.yace", "potential_yaml_path": "pot.yaml",
                "query_strategy": "uncertainty"
            },
            "kmc_params": {"active": False},
            "dft_params": {"sssp_json_path": "sssp.json", "pseudo_dir": ".", "command": "pw.x"},
            "training_params": {"ace_cutoff": 5.0, "ladder_strategy": True}
        })

        # Mock Components
        self.md_engine = MagicMock(spec=MDEngine)
        self.kmc_engine = MagicMock(spec=KMCEngine)
        self.sampler = MagicMock(spec=Sampler)
        self.generator = MagicMock(spec=StructureGenerator)
        self.labeler = MagicMock(spec=Labeler)
        self.trainer = MagicMock(spec=Trainer)

        # Instantiate Orchestrator
        self.orchestrator = ActiveLearningOrchestrator(
            self.config, self.md_engine, self.kmc_engine, self.sampler,
            self.generator, self.labeler, self.trainer
        )

        # Patch Validator inside orchestrator
        self.orchestrator.validator = MagicMock()
        self.orchestrator.validator.validate.return_value = {"status": "SUCCESS"}

    @patch("src.workflows.orchestrator.Path")
    @patch("src.workflows.orchestrator.os.chdir")
    @patch("src.workflows.orchestrator.write")
    @patch("src.workflows.orchestrator.read")
    @patch("src.workflows.orchestrator.json")
    @patch("builtins.open")
    def test_full_loop_iteration(self, mock_open, mock_json, mock_read, mock_write, mock_chdir, mock_path):
        # Setup Mocks for Paths
        mock_path_obj = MagicMock()
        mock_path.return_value = mock_path_obj
        mock_path.cwd.return_value = mock_path_obj
        mock_path_obj.exists.return_value = True
        mock_path_obj.resolve.return_value = mock_path_obj

        # State: Iteration 0
        mock_json.load.return_value = {"iteration": 0, "current_potential": "pot.yace", "current_asi": None}

        # MD returns UNCERTAIN
        self.md_engine.run.return_value = SimulationState.UNCERTAIN

        # Mock Sampler
        self.sampler.sample.return_value = [(Atoms("Ag"), 0)]

        # Mock Generator
        self.generator.generate_cell.return_value = Atoms("Ag")

        # Mock Parallel Labeling
        self.orchestrator._label_clusters_parallel = MagicMock(return_value=[Atoms("Ag")])

        # Mock Trainer
        self.trainer.prepare_dataset.return_value = "data.pckl"
        self.trainer.train.return_value = "new_pot.yace"

        def side_effect_stop(*args, **kwargs):
            raise StopIteration("Stop Loop")

        self.orchestrator._save_state = MagicMock(side_effect=[None, side_effect_stop])

        try:
            self.orchestrator.run()
        except StopIteration:
            pass
        except Exception as e:
            pass

        # Verify Flow
        self.md_engine.run.assert_called()
        self.sampler.sample.assert_called()
        self.trainer.train.assert_called()

        # Check that it was called with iteration=1
        # Use ANY call check since logic might vary slightly on how args are captured
        self.trainer.train.assert_any_call(
            dataset_path="data.pckl",
            initial_potential=unittest.mock.ANY,
            potential_yaml_path=unittest.mock.ANY,
            asi_path=unittest.mock.ANY,
            iteration=1
        )

if __name__ == "__main__":
    unittest.main()
