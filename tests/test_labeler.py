import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from src.labeler import DeltaLabeler

class MockQECalculator(Calculator):
    """Mock Calculator that returns fixed energy and forces."""
    implemented_properties = ['energy', 'forces']

    def __init__(self, energy, forces, **kwargs):
        super().__init__(**kwargs)
        self.fixed_energy = energy
        self.fixed_forces = forces

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results['energy'] = self.fixed_energy
        self.results['forces'] = self.fixed_forces

def test_delta_calculation():
    # 1. Setup Dummy Atoms
    # H2 molecule
    atoms = Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.74]])

    # 2. Setup Mock QE
    # Let's define some arbitrary DFT values
    dft_energy = -10.0
    dft_forces = np.array([[0, 0, 0.1], [0, 0, -0.1]])

    mock_qe = MockQECalculator(energy=dft_energy, forces=dft_forces)

    # 3. Setup LJ Params
    # We use LJ params that give a known result or we calculate what LJ gives and verify the difference.
    # For H2 with standard params, LJ will give some value.
    # Let's use specific LJ params to make manual verification easier?
    # Or just trust the delta subtraction math.
    # If we set LJ epsilon=0, sigma=1, then E_LJ should be 0 (if standard LJ).
    # But LJ usually has 4*epsilon*((sigma/r)^12 - (sigma/r)^6).
    # If epsilon=0, E_LJ=0, F_LJ=0.
    # This makes validation very easy: Delta = DFT.

    lj_params = {'epsilon': 0.0, 'sigma': 1.0, 'cutoff': 2.5}

    # 4. Initialize Labeler
    labeler = DeltaLabeler(qe_calculator=mock_qe, lj_params=lj_params)

    # 5. Compute Delta
    labeled_atoms = labeler.compute_delta(atoms)

    # 6. Verify Results
    # Since LJ epsilon is 0, LJ energy/forces should be 0.
    # So delta should equal DFT values.

    np.testing.assert_allclose(labeled_atoms.info['energy'], dft_energy, err_msg="Delta Energy incorrect")
    np.testing.assert_allclose(labeled_atoms.arrays['forces'], dft_forces, err_msg="Delta Forces incorrect")

    # Also verify raw values were stored
    np.testing.assert_allclose(labeled_atoms.info['energy_dft_raw'], dft_energy)
    np.testing.assert_allclose(labeled_atoms.arrays['forces_dft_raw'], dft_forces)

def test_delta_calculation_with_nonzero_lj():
    """Test with non-zero LJ to ensure subtraction logic is actually working."""
    atoms = Atoms('H2', positions=[[0, 0, 0], [0, 0, 1.0]]) # r=1.0

    dft_energy = -5.0
    dft_forces = np.array([[0, 0, 0.5], [0, 0, -0.5]])
    mock_qe = MockQECalculator(energy=dft_energy, forces=dft_forces)

    # epsilon=1, sigma=1, r=1.0
    # LJ potential: 4*1 * ((1/1)^12 - (1/1)^6) = 4 * (1 - 1) = 0.0 at sigma.
    # Wait, r=sigma is where potential crosses 0? No.
    # V = 4eps((s/r)^12 - (s/r)^6). At r=s, V=0.
    # Let's use r = 2^(1/6) sigma (minimum) to get -epsilon?
    # 2^(1/6) approx 1.122.
    # Let's stick to r=1.0 (sigma=1.0) -> E_LJ should be 0.

    # Let's change sigma so r != sigma.
    # r=1.0. Let sigma=0.5.
    # s/r = 0.5.
    # (0.5)^12 is tiny.

    # Let's just check that E_delta = E_dft - E_lj
    # We don't need to predict E_lj exactly in the test, just that the math holds.
    # But we need to know E_lj to assert the result.
    # Alternatively, we can calculate E_lj using the same calculator separately.

    lj_params = {'epsilon': 1.0, 'sigma': 1.0, 'cutoff': 2.5}

    # Calculate expected LJ manually using ASE's LJ to be sure
    from ase.calculators.lj import LennardJones
    atoms_ref = atoms.copy()
    atoms_ref.calc = LennardJones(epsilon=1.0, sigma=1.0, rc=2.5)
    e_lj_expected = atoms_ref.get_potential_energy()
    f_lj_expected = atoms_ref.get_forces()

    labeler = DeltaLabeler(qe_calculator=mock_qe, lj_params=lj_params)
    labeled_atoms = labeler.compute_delta(atoms)

    expected_delta_e = dft_energy - e_lj_expected
    expected_delta_f = dft_forces - f_lj_expected

    np.testing.assert_allclose(labeled_atoms.info['energy'], expected_delta_e)
    np.testing.assert_allclose(labeled_atoms.arrays['forces'], expected_delta_f)
