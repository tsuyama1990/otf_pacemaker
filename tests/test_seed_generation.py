"""Tests for the seed generation pipeline."""

import unittest
from unittest.mock import MagicMock, patch, mock_open
import numpy as np
from ase import Atoms
import sys
from pathlib import Path

# Ensure src is in path
sys.path.append(str(Path(__file__).parent.parent))

from src.generation.candidate import RandomStructureGenerator
from src.generation.filter import MACEFilter
from src.generation.sampler import DirectSampler
from src.seed_generation import main as seed_main
from src.config import Config, MDParams, DFTParams, LJParams, ALParams

class TestRandomStructureGenerator(unittest.TestCase):
    def test_generate(self):
        # Mock pyxtal
        with patch('src.generation.candidate.pyxtal') as mock_pyxtal_lib:
            mock_struc = MagicMock()
            mock_struc.valid = True
            mock_struc.to_ase.return_value = Atoms('Al2', positions=[[0,0,0], [1,1,1]], cell=[5,5,5])

            mock_pyxtal_instance = MagicMock()
            mock_pyxtal_instance.from_random.return_value = None
            mock_pyxtal_lib.return_value = mock_pyxtal_instance

            # We need to ensure subsequent calls return new instances or reset
            # The code calls pyxtal() each loop.

            # We also need to handle the case where pyxtal is None (ImportError)
            # but here we mock it as existing.

            # Logic:
            # loop calls pyxtal() -> gets mock_pyxtal_instance
            # mock_pyxtal_instance.from_random(...)
            # checks mock_pyxtal_instance.valid
            # calls mock_pyxtal_instance.to_ase()

            mock_pyxtal_instance.valid = True
            mock_pyxtal_instance.to_ase.return_value = Atoms('Al2')

            generator = RandomStructureGenerator(elements=['Al'], max_atoms=4)
            structures = generator.generate(n_structures=5)

            self.assertEqual(len(structures), 5)
            self.assertIsInstance(structures[0], Atoms)

class TestMACEFilter(unittest.TestCase):
    def test_filter(self):
        # Mock mace_mp
        with patch('src.generation.filter.mace_mp') as mock_mace_class:
            mock_calc = MagicMock()
            mock_mace_class.return_value = mock_calc

            # Mock atoms behavior when calc is set
            # We can't easily mock the side effect of atoms.get_forces() using the calc
            # because ASE calls calc.get_forces(atoms).
            # So we should mock the atoms object or the calc's methods if we attached it manually?
            # In the code:
            # atoms.calc = self.calc
            # forces = atoms.get_forces()

            # ASE's atoms.get_forces() calls self.calc.get_forces(self).
            mock_calc.get_forces.return_value = np.array([[0.0, 0.0, 0.0]])
            mock_calc.get_potential_energy.return_value = -10.0

            m_filter = MACEFilter(force_cutoff=1.0)

            atoms_good = Atoms('H', positions=[[0, 0, 0]])
            atoms_bad = Atoms('H', positions=[[1, 1, 1]])

            # We need to control return values for different calls
            # Side effect: 1st call good, 2nd call bad

            def get_forces_side_effect(atoms):
                # ASE atoms comparison is by identity usually unless overridden
                if atoms is atoms_good:
                    return np.array([[0.1, 0.0, 0.0]])
                elif atoms is atoms_bad:
                    return np.array([[5.0, 0.0, 0.0]]) # > 1.0
                return np.array([[0.0, 0.0, 0.0]])

            mock_calc.get_forces.side_effect = get_forces_side_effect

            filtered = m_filter.filter([atoms_good, atoms_bad])

            self.assertEqual(len(filtered), 1)
            self.assertEqual(filtered[0], atoms_good)

class TestDirectSampler(unittest.TestCase):
    def test_sample(self):
        # Mock Birch
        with patch('src.generation.sampler.Birch') as mock_birch_class:
            mock_birch = MagicMock()
            mock_birch_class.return_value = mock_birch

            # Mock fit_predict
            # 3 structures. 2 clusters.
            # 0 -> cluster 0
            # 1 -> cluster 0
            # 2 -> cluster 1
            mock_birch.fit_predict.return_value = np.array([0, 0, 1])

            # Mock compute_descriptors
            # We need to mock the subprocess or the private method
            with patch('src.generation.sampler.DirectSampler._compute_descriptors') as mock_compute:
                # 3 structures, 5 features
                mock_compute.return_value = np.array([
                    [1.0, 0.0], # 0
                    [1.1, 0.0], # 1 (close to 0)
                    [5.0, 5.0]  # 2 (far)
                ])

                sampler = DirectSampler(n_clusters=2)
                structures = [Atoms('H'), Atoms('H'), Atoms('H')]

                selected = sampler.sample(structures)

                # Should select one from cluster 0 (index 0 or 1) and one from cluster 1 (index 2)
                self.assertEqual(len(selected), 2)
                self.assertIn(structures[2], selected)

class TestSeedGenerationPipeline(unittest.TestCase):
    @patch('src.seed_generation.Config')
    @patch('src.seed_generation.RandomStructureGenerator')
    @patch('src.seed_generation.MACEFilter')
    @patch('src.seed_generation.DirectSampler')
    @patch('src.seed_generation.DeltaLabeler')
    @patch('src.seed_generation.PacemakerTrainer')
    @patch('src.seed_generation.Espresso') # Mock calculator init
    @patch('src.seed_generation.shutil')
    def test_main(self, mock_shutil, mock_espresso, mock_trainer_cls, mock_labeler_cls,
                  mock_sampler_cls, mock_filter_cls, mock_gen_cls, mock_config_cls):

        # Setup Config
        mock_config = MagicMock()
        mock_config.md_params.elements = ['Al']
        mock_config.dft_params.command = "pw.x"
        mock_config.dft_params.kpts = [1,1,1]
        mock_config.dft_params.pseudo_dir = "."
        mock_config.dft_params.ecutwfc = 30.0
        mock_config.lj_params.epsilon = 1.0
        mock_config.lj_params.sigma = 1.0
        mock_config.lj_params.cutoff = 2.5

        mock_config_cls.from_yaml.return_value = mock_config

        # Setup Generator
        mock_gen = mock_gen_cls.return_value
        mock_gen.generate.return_value = [Atoms('Al')] * 10

        # Setup Filter
        mock_filter = mock_filter_cls.return_value
        mock_filter.filter.return_value = [Atoms('Al')] * 8

        # Setup Sampler
        mock_sampler = mock_sampler_cls.return_value
        mock_sampler.sample.return_value = [Atoms('Al')] * 2 # 2 seeds

        # Setup Labeler
        mock_labeler = mock_labeler_cls.return_value
        mock_labeler.label.return_value = Atoms('Al') # successful label

        # Setup Trainer
        mock_trainer = mock_trainer_cls.return_value
        mock_trainer.prepare_dataset.return_value = "temp_data.pckl"
        mock_trainer.train.return_value = "temp_pot.yace"

        # Run main
        # We need to mock sys.argv
        with patch.object(sys, 'argv', ['seed_gen', '--config', 'config.yaml']):
             seed_main()

        # Verifications
        mock_gen.generate.assert_called()
        mock_filter.filter.assert_called()
        mock_sampler.sample.assert_called()

        # Labeler should be called for each sampled structure (2)
        self.assertEqual(mock_labeler.label.call_count, 2)

        # Trainer
        mock_trainer.train.assert_called_with(str(Path("data/seed/seed_data.pckl.gzip")), initial_potential=None)

        # Moves
        # Check if shutil.move was called for dataset and potential
        self.assertTrue(mock_shutil.move.called)

if __name__ == '__main__':
    unittest.main()
