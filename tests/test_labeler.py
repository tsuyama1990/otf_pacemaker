import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from src.labeler import DeltaLabeler

class MockCalculator(Calculator):
    """Mock Calculator that returns fixed energy, forces and stress."""
    implemented_properties = ['energy', 'forces', 'stress']

    def __init__(self, energy, forces, stress=None, **kwargs):
        super().__init__(**kwargs)
        self.fixed_energy = energy
        self.fixed_forces = forces
        self.fixed_stress = stress if stress is not None else np.zeros(6)

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results['energy'] = self.fixed_energy
        self.results['forces'] = self.fixed_forces
        self.results['stress'] = self.fixed_stress

def test_delta_labeling():
    # Set cell and pbc to allow stress calculation
    atoms = Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.74]], cell=[10, 10, 10], pbc=True)

    dft_energy = -10.0
    dft_forces = np.array([[0, 0, 0.1], [0, 0, -0.1]])
    mock_calc = MockCalculator(energy=dft_energy, forces=dft_forces)

    lj_params = {'epsilon': 0.0, 'sigma': 1.0, 'cutoff': 2.5}

    labeler = DeltaLabeler(qe_calculator=mock_calc, lj_params=lj_params)

    labeled_atoms = labeler.label(atoms)

    assert labeled_atoms is not None, "Labeling returned None unexpectedly"

    np.testing.assert_allclose(labeled_atoms.info['energy'], dft_energy)
    np.testing.assert_allclose(labeled_atoms.arrays['forces'], dft_forces)
    np.testing.assert_allclose(labeled_atoms.info['energy_dft_raw'], dft_energy)
    np.testing.assert_allclose(labeled_atoms.arrays['forces_dft_raw'], dft_forces)

def test_delta_labeling_with_lj():
    # Set cell and pbc to allow stress calculation
    atoms = Atoms('H2', positions=[[0, 0, 0], [0, 0, 1.0]], cell=[10, 10, 10], pbc=True)

    dft_energy = -5.0
    dft_forces = np.array([[0, 0, 0.5], [0, 0, -0.5]])
    mock_calc = MockCalculator(energy=dft_energy, forces=dft_forces)

    lj_params = {'epsilon': 1.0, 'sigma': 1.0, 'cutoff': 2.5}

    # Calculate expected LJ manually
    from ase.calculators.lj import LennardJones
    atoms_ref = atoms.copy()
    atoms_ref.calc = LennardJones(epsilon=1.0, sigma=1.0, rc=2.5)
    e_lj_expected = atoms_ref.get_potential_energy()
    f_lj_expected = atoms_ref.get_forces()
    s_lj_expected = atoms_ref.get_stress()

    labeler = DeltaLabeler(qe_calculator=mock_calc, lj_params=lj_params)
    labeled_atoms = labeler.label(atoms)

    assert labeled_atoms is not None

    expected_delta_e = dft_energy - e_lj_expected
    expected_delta_f = dft_forces - f_lj_expected
    expected_delta_s = np.zeros(6) - s_lj_expected # since mock stress is 0

    np.testing.assert_allclose(labeled_atoms.info['energy'], expected_delta_e)
    np.testing.assert_allclose(labeled_atoms.arrays['forces'], expected_delta_f)
    np.testing.assert_allclose(labeled_atoms.info['stress'], expected_delta_s)
