import unittest
from unittest.mock import MagicMock, patch, mock_open
import yaml
import os
from src.training.strategies.pacemaker import PacemakerTrainer
from src.core.config import TrainingParams

class TestPacemakerTrainerDynamic(unittest.TestCase):
    def setUp(self):
        self.params = TrainingParams(
            ladder_strategy=True,
            initial_max_deg=4,
            final_max_deg=8,
            ladder_interval=2,
            max_training_time=100
        )
        self.trainer = PacemakerTrainer(self.params)

    @patch("src.training.strategies.pacemaker.get_available_vram")
    @patch("src.training.strategies.pacemaker.suggest_batch_size")
    def test_get_dynamic_config(self, mock_suggest, mock_vram):
        mock_vram.return_value = 8000
        mock_suggest.return_value = 64

        # Iteration 0
        conf0 = self.trainer._get_dynamic_config(0)
        self.assertEqual(conf0["max_deg"], 4)
        self.assertEqual(conf0["batch_size"], 64)

        # Iteration 1 (interval=2, still 0)
        conf1 = self.trainer._get_dynamic_config(1)
        self.assertEqual(conf1["max_deg"], 4)

        # Iteration 2 (interval=2, step=1 -> max_deg = 4 + 1 = 5)
        conf2 = self.trainer._get_dynamic_config(2)
        self.assertEqual(conf2["max_deg"], 5)

        # Iteration 10 (step=5 -> max_deg = 4 + 5 = 9 -> cap at 8)
        conf10 = self.trainer._get_dynamic_config(10)
        self.assertEqual(conf10["max_deg"], 8)

    @patch("subprocess.run")
    @patch("src.training.strategies.pacemaker.PacemakerTrainer._update_and_sample_dataset")
    @patch("pandas.read_pickle")
    @patch("pathlib.Path.exists")
    def test_prune_active_set(self, mock_exists, mock_pd, mock_update, mock_run):
        mock_exists.return_value = True
        self.trainer.prune_active_set("potential.asi", threshold=0.95)

        mock_run.assert_called_with(
            ["pace_activeset", "-a", "potential.asi", "--prune", "0.95", "--overwrite"],
            check=True,
            capture_output=True,
            text=True
        )

if __name__ == "__main__":
    unittest.main()
