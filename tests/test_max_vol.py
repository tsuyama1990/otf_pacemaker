import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from ase import Atoms
import sys

# Mock pyace
mock_pyace = MagicMock()
sys.modules["pyace"] = mock_pyace

from src.sampling.strategies.max_vol import MaxVolSampler

class TestMaxVolSampler(unittest.TestCase):
    def setUp(self):
        self.sampler = MaxVolSampler()
        self.structures = [
            Atoms("H2", positions=[[0, 0, 0], [0, 0, 1]]),
            Atoms("H2", positions=[[0, 0, 0], [0, 0, 1.1]]),
            Atoms("H2", positions=[[0, 0, 0], [0, 0, 1.2]]),
        ]
        self.potential_path = "dummy.yace"

    @patch("src.sampling.strategies.max_vol.ACESampler")
    def test_sample_fallback_qr(self, MockACESampler):
        # Temporarily remove SelectMaxVol
        original_select = getattr(mock_pyace, "SelectMaxVol", None)
        del mock_pyace.SelectMaxVol

        try:
            mock_instance = MockACESampler.return_value
            mock_instance.compute_descriptors.side_effect = [
                np.array([[1.0, 0.0]]),
                np.array([[0.0, 1.0]]),
                np.array([[0.9, 0.0]])
            ]
            mock_instance.calculator.get_property.return_value = np.array([0.1])

            results = self.sampler.sample(
                structures=self.structures,
                potential_path=self.potential_path,
                n_clusters=2
            )

            self.assertEqual(len(results), 2)
            selected_atoms = [r[0] for r in results]
            self.assertIn(self.structures[0], selected_atoms)
            self.assertIn(self.structures[1], selected_atoms)

        finally:
            if original_select:
                mock_pyace.SelectMaxVol = original_select

if __name__ == "__main__":
    unittest.main()
