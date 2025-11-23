import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from src.labeling.strategies.delta_labeler import DeltaLabeler

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
    mock_dft = MockCalculator(energy=dft_energy, forces=dft_forces)

    lj_energy = -1.0
    lj_forces = np.array([[0, 0, 0.01], [0, 0, -0.01]])
    mock_lj = MockCalculator(energy=lj_energy, forces=lj_forces)

    labeler = DeltaLabeler(reference_calculator=mock_dft, baseline_calculator=mock_lj)

    labeled_atoms = labeler.label(atoms)

    assert labeled_atoms is not None, "Labeling returned None unexpectedly"

    np.testing.assert_allclose(labeled_atoms.info['energy'], dft_energy - lj_energy)
    np.testing.assert_allclose(labeled_atoms.arrays['forces'], dft_forces - lj_forces)
    np.testing.assert_allclose(labeled_atoms.info['energy_dft_raw'], dft_energy)
    np.testing.assert_allclose(labeled_atoms.arrays['forces_dft_raw'], dft_forces)

def test_delta_labeling_real_lj():
    # Test integration with actual ShiftedLennardJones if needed, or just rely on unit test of DeltaLabeler logic
    # Here we repeat the previous test logic but injecting ShiftedLennardJones as the baseline

    atoms = Atoms('H2', positions=[[0, 0, 0], [0, 0, 1.0]], cell=[10, 10, 10], pbc=True)

    dft_energy = -5.0
    dft_forces = np.array([[0, 0, 0.5], [0, 0, -0.5]])
    mock_dft = MockCalculator(energy=dft_energy, forces=dft_forces)

    from src.labeling.calculators.shifted_lj import ShiftedLennardJones
    lj_calc = ShiftedLennardJones(epsilon=1.0, sigma=1.0, rc=2.5)

    # Calculate expected LJ manually to verify
    atoms_ref = atoms.copy()
    atoms_ref.calc = ShiftedLennardJones(epsilon=1.0, sigma=1.0, rc=2.5)
    e_lj_expected = atoms_ref.get_potential_energy()
    f_lj_expected = atoms_ref.get_forces()
    s_lj_expected = atoms_ref.get_stress()

    labeler = DeltaLabeler(reference_calculator=mock_dft, baseline_calculator=lj_calc)
    labeled_atoms = labeler.label(atoms)

    assert labeled_atoms is not None

    expected_delta_e = dft_energy - e_lj_expected
    expected_delta_f = dft_forces - f_lj_expected
    # mock_dft has 0 stress by default (from init)
    expected_delta_s = np.zeros(6) - s_lj_expected

    np.testing.assert_allclose(labeled_atoms.info['energy'], expected_delta_e)
    np.testing.assert_allclose(labeled_atoms.arrays['forces'], expected_delta_f)
    np.testing.assert_allclose(labeled_atoms.info['stress'], expected_delta_s)
