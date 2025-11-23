import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from ase import Atoms
from src.scenario_generation.strategies.small_cell import SmallCellGenerator
from src.sampling.strategies.max_gamma import MaxGammaSampler

class TestSmallCellGenerator(unittest.TestCase):
    def setUp(self):
        self.box_size = 10.0
        self.r_core = 3.0
        # Fixed: Removed stoichiometry_tolerance
        self.generator = SmallCellGenerator(
            box_size=self.box_size,
            r_core=self.r_core,
            stoichiometric_ratio={'Fe': 1.0}, # Updated to correct arg
            lammps_cmd="lmp_serial"
        )

        # Create a dummy atoms object
        # 10x10x10 cubic cell, 2 atoms
        self.atoms = Atoms('Fe2',
                           positions=[[0, 0, 0], [2, 0, 0]],
                           cell=[10, 10, 10],
                           pbc=True)

    @patch('src.scenario_generation.strategies.small_cell.LAMMPS')
    @patch('src.scenario_generation.strategies.small_cell.ExpCellFilter')
    @patch('src.scenario_generation.strategies.small_cell.FIRE')
    def test_generate_structure(self, mock_fire, mock_filter, mock_lammps):
        # Setup mocks
        mock_calc = MagicMock()
        mock_lammps.return_value = mock_calc

        # Mock FIRE run to do nothing
        mock_opt = MagicMock()
        mock_fire.return_value = mock_opt

        # Run generate with atom 0 as center
        center_id = 0
        small_cell = self.generator.generate_cell(self.atoms, center_id, "dummy.yace") # Updated method name

        # Assertions on the generated cell
        self.assertTrue(small_cell.pbc.all())
        np.testing.assert_array_almost_equal(small_cell.cell.lengths(), [self.box_size]*3)

        positions = small_cell.get_positions()
        self.assertEqual(len(small_cell), 2)

        # Find atom corresponding to original center (closest to 5,5,5)
        center_pos = np.array([5.0, 5.0, 5.0])
        dists = np.linalg.norm(positions - center_pos, axis=1)
        self.assertLess(np.min(dists), 0.01) # One atom should be at center

    @patch('src.scenario_generation.strategies.small_cell.LAMMPS')
    @patch('src.scenario_generation.strategies.small_cell.ExpCellFilter')
    @patch('src.scenario_generation.strategies.small_cell.FIRE')
    def test_relaxation_called(self, mock_fire, mock_filter, mock_lammps):
        mock_calc = MagicMock()
        mock_lammps.return_value = mock_calc
        mock_opt = MagicMock()
        mock_fire.return_value = mock_opt

        self.generator.generate_cell(self.atoms, 0, "dummy.yace") # Updated method name

        mock_lammps.assert_called()
        args, kwargs = mock_lammps.call_args
        self.assertIn('parameters', kwargs)
        self.assertIn('pair_style', kwargs['parameters'])
        self.assertEqual(kwargs['parameters']['pair_style'], 'pace')

        mock_filter.assert_called()
        mock_fire.assert_called()
        mock_opt.run.assert_called()


class TestMaxGammaSampler(unittest.TestCase):
    def test_sample(self):
        sampler = MaxGammaSampler()
        atoms = Atoms('H5')
        # Setup arrays
        # 5 atoms. Gamma values: [0.1, 0.5, 0.2, 0.9, 0.0]
        # Max is index 3 (0.9), then index 1 (0.5)
        atoms.set_array('f_f_gamma', np.array([0.1, 0.5, 0.2, 0.9, 0.0]))

        # Test n_clusters = 1
        indices = sampler.sample(atoms=atoms, n_clusters=1)
        # Result is list of tuples (atoms, index)
        self.assertEqual(indices[0][1], 3)

        # Test n_clusters = 2
        indices = sampler.sample(atoms=atoms, n_clusters=2)
        # Should be sorted max first
        self.assertEqual(indices[0][1], 3)
        self.assertEqual(indices[1][1], 1)

    def test_sample_raises_error_on_missing(self):
        sampler = MaxGammaSampler()
        atoms = Atoms('H10')
        # No gamma array

        with self.assertRaises(ValueError):
             sampler.sample(atoms=atoms, n_clusters=2)

if __name__ == '__main__':
    unittest.main()
