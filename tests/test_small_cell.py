import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from ase import Atoms
from src.active_learning import SmallCellGenerator

class TestSmallCellGenerator(unittest.TestCase):
    def setUp(self):
        self.box_size = 10.0
        self.r_core = 3.0
        self.generator = SmallCellGenerator(
            box_size=self.box_size,
            r_core=self.r_core,
            stoichiometry_tolerance=0.1,
            elements=['Fe']
        )

        # Create a dummy atoms object
        # 10x10x10 cubic cell, 2 atoms
        self.atoms = Atoms('Fe2',
                           positions=[[0, 0, 0], [2, 0, 0]],
                           cell=[10, 10, 10],
                           pbc=True)

    @patch('src.active_learning.LAMMPS')
    @patch('src.active_learning.ExpCellFilter')
    @patch('src.active_learning.FIRE')
    def test_generate_cell_structure(self, mock_fire, mock_filter, mock_lammps):
        # Setup mocks
        mock_calc = MagicMock()
        mock_lammps.return_value = mock_calc

        # Mock FIRE run to do nothing
        mock_opt = MagicMock()
        mock_fire.return_value = mock_opt

        # Run generate_cell with atom 0 as center
        center_id = 0
        small_cell = self.generator.generate_cell(self.atoms, center_id, "dummy.yace")

        # Assertions on the generated cell
        self.assertTrue(small_cell.pbc.all())
        np.testing.assert_array_almost_equal(small_cell.cell.lengths(), [self.box_size]*3)

        # Check if atoms are shifted correctly
        # Original atom 0 is at 0,0,0. New box center is 5,5,5.
        # Atom 0 should be at 5,5,5.
        # Atom 1 was at 2,0,0 (dist 2). Should be at 7,5,5.

        positions = small_cell.get_positions()
        # We don't know order for sure if implementation changed, but we expect 2 atoms
        self.assertEqual(len(small_cell), 2)

        # Find atom corresponding to original center (closest to 5,5,5)
        center_pos = np.array([5.0, 5.0, 5.0])
        dists = np.linalg.norm(positions - center_pos, axis=1)
        self.assertLess(np.min(dists), 0.01) # One atom should be at center

    @patch('src.active_learning.LAMMPS')
    @patch('src.active_learning.ExpCellFilter')
    @patch('src.active_learning.FIRE')
    def test_relaxation_called(self, mock_fire, mock_filter, mock_lammps):
        mock_calc = MagicMock()
        mock_lammps.return_value = mock_calc
        mock_opt = MagicMock()
        mock_fire.return_value = mock_opt

        self.generator.generate_cell(self.atoms, 0, "dummy.yace")

        # Verify LAMMPS calculator setup
        mock_lammps.assert_called()
        # Check arguments passed to LAMMPS (partial check)
        args, kwargs = mock_lammps.call_args
        self.assertIn('pair_style', kwargs['parameters'])
        self.assertEqual(kwargs['parameters']['pair_style'], 'pace')

        # Verify Optimization
        mock_filter.assert_called()
        mock_fire.assert_called()
        mock_opt.run.assert_called()

if __name__ == '__main__':
    unittest.main()
