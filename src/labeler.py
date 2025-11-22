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
        """Compute delta energy, forces, and stress for a given structure.

        Args:
            structure: The atomic cluster to label.

        Returns:
            Atoms: The cluster with updated energy, forces, and stress representing the delta.
        """
        # Work on copies
        cluster_dft = structure.copy()
        cluster_lj = structure.copy()

        # 1. DFT Calculation
        cluster_dft.calc = self.qe_calculator
        e_dft = cluster_dft.get_potential_energy()
        f_dft = cluster_dft.get_forces()
        # We assume the calculator supports stress since we need to label it.
        # If it doesn't, this will raise an error, which is appropriate given requirements.
        try:
             s_dft = cluster_dft.get_stress()
        except Exception:
             # Some calculators might not support stress or require specific params.
             # We assume the user provided a capable calculator.
             # Attempting to proceed without stress would violate the "validity" requirement.
             # Re-raise is appropriate.
             raise

        # 2. LJ Calculation
        # ASE LennardJones automatically shifts the potential if 'rc' is provided.
        lj_kwargs = {
            'epsilon': self.lj_params.get('epsilon', 1.0),
            'sigma': self.lj_params.get('sigma', 1.0),
            'rc': self.lj_params.get('cutoff', 2.5)
        }
        cluster_lj.calc = LennardJones(**lj_kwargs)
        e_lj = cluster_lj.get_potential_energy()
        f_lj = cluster_lj.get_forces()
        s_lj = cluster_lj.get_stress()

        # 3. Compute Delta
        result_cluster = cluster_dft.copy()
        result_cluster.calc = None

        e_delta = e_dft - e_lj
        f_delta = f_dft - f_lj
        s_delta = s_dft - s_lj

        # 4. Store Results
        result_cluster.info['energy'] = e_delta
        result_cluster.arrays['forces'] = f_delta
        result_cluster.info['stress'] = s_delta

        # Store Raw DFT
        result_cluster.info['energy_dft_raw'] = e_dft
        result_cluster.arrays['forces_dft_raw'] = f_dft
        # result_cluster.info['stress_dft_raw'] = s_dft # Optional

        # Weights
        result_cluster.info['energy_weight'] = 1.0
        result_cluster.info['virial_weight'] = 1.0

        return result_cluster
