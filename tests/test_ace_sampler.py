import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from ase import Atoms
import sys

# Mock pyace module before importing ACESampler
sys.modules["pyace"] = MagicMock()

from src.sampling.strategies.ace_sampler import ACESampler

class TestACESampler(unittest.TestCase):
    def setUp(self):
        self.potential_path = "dummy_potential.yace"
        self.sampler = ACESampler(self.potential_path)
        # Mock the calculator instance
        self.mock_calc = MagicMock()
        self.sampler._calculator = self.mock_calc

    def test_compute_descriptors(self):
        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1]])
        expected_descriptors = np.array([[1.0, 2.0], [3.0, 4.0]])

        self.mock_calc.get_property.return_value = expected_descriptors

        descriptors = self.sampler.compute_descriptors(atoms)

        self.mock_calc.get_property.assert_called_with("B", atoms)
        np.testing.assert_array_equal(descriptors, expected_descriptors)

    def test_compute_gamma(self):
        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1]])
        expected_gammas = np.array([0.5, 1.2])

        self.mock_calc.get_property.return_value = expected_gammas

        gamma = self.sampler.compute_gamma(atoms)

        self.mock_calc.get_property.assert_called_with("gamma", atoms)
        self.assertEqual(gamma, 1.2)

if __name__ == "__main__":
    unittest.main()
