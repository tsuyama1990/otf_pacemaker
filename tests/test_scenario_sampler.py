import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from ase import Atoms
import sys

# Mock dependencies
sys.modules["pyace"] = MagicMock()
mock_sklearn = MagicMock()
mock_birch = MagicMock()
mock_sklearn.cluster.Birch = mock_birch
sys.modules["sklearn"] = mock_sklearn
sys.modules["sklearn.cluster"] = MagicMock(Birch=mock_birch)

from src.scenario_generation.sampler import DirectSampler

class TestDirectSampler(unittest.TestCase):
    def setUp(self):
        self.sampler = DirectSampler(n_clusters=2)
        self.structures = [
            Atoms("H2", positions=[[0, 0, 0], [0, 0, 1]]),
            Atoms("H2", positions=[[0, 0, 0], [0, 0, 1.1]]),
            Atoms("H2", positions=[[0, 0, 0], [0, 0, 1.2]]),
        ]

    @patch("src.scenario_generation.sampler.ACESampler")
    def test_sample(self, MockACESampler):
        mock_ace_instance = MockACESampler.return_value
        # Return different descriptors for each structure
        mock_ace_instance.compute_descriptors.side_effect = [
            np.array([[1.0, 0.0], [1.0, 0.0]]), # Struct 1
            np.array([[0.0, 1.0], [0.0, 1.0]]), # Struct 2
            np.array([[0.5, 0.5], [0.5, 0.5]]), # Struct 3
        ]

        # Setup Mock Birch to split into 2 clusters
        mock_birch_instance = mock_birch.return_value
        mock_birch_instance.fit_predict.return_value = np.array([0, 1, 0])

        selected = self.sampler.sample(self.structures)

        self.assertEqual(len(selected), 2)
        MockACESampler.assert_called()
        mock_birch.assert_called_with(n_clusters=2)

if __name__ == "__main__":
    unittest.main()
