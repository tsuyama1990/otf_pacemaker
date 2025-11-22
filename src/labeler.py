"""Labeling module for calculating target properties.

This module calculates the difference between DFT and Empirical Potential (LJ)
forces and energies to create training labels.
"""

from typing import Dict
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.calculators.lj import LennardJones

from src.interfaces import Labeler


class DeltaLabeler(Labeler):
    """Calculates the delta between DFT and reference (LJ) calculations."""

    def __init__(self, qe_calculator: Calculator, lj_params: Dict[str, float]):
        """Initialize the DeltaLabeler.

        Args:
            qe_calculator: A configured ASE calculator for the ground truth (DFT).
            lj_params: A dictionary containing LJ parameters (epsilon, sigma, cutoff).
        """
        self.qe_calculator = qe_calculator
        self.lj_params = lj_params

    def label(self, structure: Atoms) -> Atoms:
        """Compute delta energy and forces for a given structure.

        Args:
            structure: The atomic cluster to label.

        Returns:
            Atoms: The cluster with updated energy and forces representing the delta.
        """
        # Work on copies
        cluster_dft = structure.copy()
        cluster_lj = structure.copy()

        # 1. DFT Calculation
        cluster_dft.calc = self.qe_calculator
        e_dft = cluster_dft.get_potential_energy()
        f_dft = cluster_dft.get_forces()

        # 2. LJ Calculation
        lj_kwargs = {
            'epsilon': self.lj_params.get('epsilon', 1.0),
            'sigma': self.lj_params.get('sigma', 1.0),
            'rc': self.lj_params.get('cutoff', 2.5)
        }
        cluster_lj.calc = LennardJones(**lj_kwargs)
        e_lj = cluster_lj.get_potential_energy()
        f_lj = cluster_lj.get_forces()

        # 3. Compute Delta
        result_cluster = cluster_dft.copy()
        result_cluster.calc = None

        e_delta = e_dft - e_lj
        f_delta = f_dft - f_lj

        # 4. Store Results
        result_cluster.info['energy'] = e_delta
        result_cluster.arrays['forces'] = f_delta

        # Store Raw DFT
        result_cluster.info['energy_dft_raw'] = e_dft
        result_cluster.arrays['forces_dft_raw'] = f_dft

        result_cluster.info['energy_weight'] = 1.0

        return result_cluster
