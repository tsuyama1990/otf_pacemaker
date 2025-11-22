"""Tests for Labeler module."""

import pytest
import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from src.labeler import DeltaLabeler

class MockEspresso(Calculator):
    """Mock Espresso calculator that returns fixed energy and forces."""
    implemented_properties = ['energy', 'forces']

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results['energy'] = -100.0
        # Forces pointing in x direction
        self.results['forces'] = np.zeros((len(atoms), 3))
        self.results['forces'][:, 0] = 1.0

class MockEspressoFail(Calculator):
    """Mock Espresso calculator that raises an exception."""
    implemented_properties = ['energy', 'forces']

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        raise RuntimeError("DFT Convergence Failed")

def test_delta_labeler_compute_delta():
    atoms = Atoms('Ar2', positions=[[0, 0, 0], [0, 0, 3.0]])

    # LJ parameters that give small non-zero energy/forces at 3.0
    lj_params = {'epsilon': 1.0, 'sigma': 1.0, 'cutoff': 5.0}

    mock_qe = MockEspresso()
    labeler = DeltaLabeler(qe_calculator=mock_qe, lj_params=lj_params)

    labeled_atoms = labeler.compute_delta(atoms)

    # Check raw DFT values stored
    assert labeled_atoms.info['energy_dft_raw'] == -100.0
    assert np.allclose(labeled_atoms.arrays['forces_dft_raw'][:, 0], 1.0)

    # Check Delta values
    # We don't compute exact LJ here, but we know it's finite.
    # E_delta = E_dft - E_lj = -100.0 - E_lj
    assert 'energy' in labeled_atoms.info
    assert 'forces' in labeled_atoms.arrays

    # Ensure calculator is removed
    assert labeled_atoms.calc is None

def test_delta_labeler_failure():
    atoms = Atoms('Ar', positions=[[0, 0, 0]])
    lj_params = {'epsilon': 1.0, 'sigma': 1.0, 'cutoff': 2.5}

    mock_fail = MockEspressoFail()
    labeler = DeltaLabeler(qe_calculator=mock_fail, lj_params=lj_params)

    with pytest.raises(RuntimeError, match="DFT Convergence Failed"):
        labeler.compute_delta(atoms)
