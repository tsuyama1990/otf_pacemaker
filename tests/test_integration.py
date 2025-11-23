"""Integration tests for the system optimization and integration (Phase 3)."""

import unittest
from unittest.mock import MagicMock, patch
import shutil
import tempfile
from pathlib import Path
from ase import Atoms
from src.core.config import Config
from src.main import main
from src.workflows.orchestrator import ActiveLearningOrchestrator
from src.utils.logger import CSVLogger

class TestIntegrationPhase3(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.original_cwd = Path.cwd()
        # Copy config.yaml if it exists, or create a dummy one
        self.config_path = Path(self.test_dir) / "config.yaml"

        # Create dummy config
        self.config_data = {
            "md_params": {
                "timestep": 0.5,
                "temperature": 300,
                "pressure": 0,
                "n_steps": 100,
                "elements": ["Al"],
                "initial_structure": "structure.xyz",
                "masses": {"Al": 26.98},
                "restart_freq": 50,
                "dump_freq": 50
            },
            "al_params": {
                "gamma_threshold": 0.1,
                "n_clusters": 2,
                "r_core": 3.0,
                "box_size": 10.0,
                "initial_potential": "potential.yace",
                "potential_yaml_path": "potential.yaml",
                "initial_dataset_path": "dataset.pckl",
                "num_parallel_labeling": 2
            },
            "dft_params": {
                "ecutwfc": 30,
                "kpts": [1, 1, 1],
                "pseudo_dir": "pseudos",
                "command": "echo"
            },
            "lj_params": {
                "epsilon": 1.0,
                "sigma": 1.0,
                "cutoff": 3.0
            }
        }

        import yaml
        with open(self.config_path, "w") as f:
            yaml.dump(self.config_data, f)

        # Create dummy structure
        atoms = Atoms("Al2", positions=[[0,0,0], [2,0,0]], cell=[10,10,10], pbc=True)
        from ase.io import write
        write(Path(self.test_dir) / "structure.xyz", atoms)

        # Create dummy pseudo dir
        (Path(self.test_dir) / "pseudos").mkdir()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @patch("src.main.Espresso")
    @patch("src.main.SeedGenerator")
    @patch("src.main.ActiveLearningOrchestrator")
    def test_main_runs_phase1_if_potential_missing(self, mock_orchestrator, mock_seed_gen, mock_espresso):
        """Test that Phase 1 is triggered if initial potential is missing."""
        # potential.yace does not exist in self.test_dir

        # Mock Espresso to avoid runtime errors during instantiation in main
        mock_espresso.return_value = MagicMock()

        # Create mock instances
        mock_seed_instance = mock_seed_gen.return_value
        mock_orch_instance = mock_orchestrator.return_value

        # We need to simulate the seed generator actually creating the file
        # so that Phase 2 continues (or main copies it).
        def side_effect_run():
            # Create the file that main expects
            (Path("data/seed/seed_potential.yace")).parent.mkdir(parents=True, exist_ok=True)
            (Path("data/seed/seed_potential.yace")).touch()

        mock_seed_instance.run.side_effect = side_effect_run

        # Need to ensure that main runs in self.test_dir context
        import os
        current_dir = os.getcwd()
        try:
            os.chdir(self.test_dir)
            main()
        finally:
            os.chdir(current_dir)

        # Check Phase 1 was called
        mock_seed_gen.assert_called_once()
        mock_seed_instance.run.assert_called_once()

        # Check Phase 2 was called
        mock_orchestrator.assert_called_once()
        mock_orch_instance.run.assert_called_once()

    @patch("src.main.Espresso")
    @patch("src.main.SeedGenerator")
    @patch("src.main.ActiveLearningOrchestrator")
    def test_main_skips_phase1_if_potential_exists(self, mock_orchestrator, mock_seed_gen, mock_espresso):
        """Test that Phase 1 is skipped if initial potential exists."""
        # Mock Espresso
        mock_espresso.return_value = MagicMock()

        import os
        current_dir = os.getcwd()
        try:
            os.chdir(self.test_dir)

            # Create the potential file
            (Path(self.test_dir) / "potential.yace").touch()

            main()
        finally:
            os.chdir(current_dir)

        # Check Phase 1 was NOT called
        mock_seed_gen.assert_not_called()

        # Check Phase 2 was called
        mock_orchestrator.assert_called_once()

    def test_parallel_labeling(self):
        """Test the parallel labeling logic in Orchestrator."""
        # Setup mocks for dependencies
        config = Config.from_yaml(self.config_path)
        mock_md = MagicMock()
        mock_sampler = MagicMock()
        mock_generator = MagicMock()
        mock_labeler = MagicMock()
        mock_trainer = MagicMock()

        # Setup mock behavior
        atoms = Atoms("Al", positions=[[0,0,0]])
        mock_labeler.label.return_value = atoms

        orchestrator = ActiveLearningOrchestrator(
            config, mock_md, mock_sampler, mock_generator, mock_labeler, mock_trainer
        )

        # Create a list of clusters
        clusters = [atoms.copy() for _ in range(4)]

        # Test with ProcessPoolExecutor mocked to avoid pickling issues and ensure we are testing logic
        with patch("src.workflows.orchestrator.ProcessPoolExecutor") as MockExecutor:
            instance = MockExecutor.return_value
            instance.__enter__.return_value = instance

            # Mock submit to return a Future-like object that returns the input atom (labeled)
            from concurrent.futures import Future
            def side_effect_submit(fn, labeler, atom):
                f = Future()
                f.set_result(atom) # In real life fn returns atom, here we assume it worked
                return f

            instance.submit.side_effect = side_effect_submit

            results = orchestrator._label_clusters_parallel(clusters)

            self.assertEqual(len(results), 4)
            MockExecutor.assert_called_with(max_workers=2)

    def test_csv_logger(self):
        """Test that CSVLogger writes correctly."""
        log_path = Path(self.test_dir) / "test_log.csv"
        logger = CSVLogger(str(log_path))

        logger.log_metrics(1, 0.5, 10, 100, 0.01, 0.02)

        self.assertTrue(log_path.exists())
        import csv
        with open(log_path, "r") as f:
            reader = list(csv.reader(f))
            self.assertEqual(len(reader), 2) # Header + 1 row
            self.assertEqual(reader[1][0], "1")
            self.assertEqual(reader[1][1], "0.5")

if __name__ == "__main__":
    unittest.main()
