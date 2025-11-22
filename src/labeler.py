"""Labeling module for calculating target properties.

This module calculates the difference between DFT and Empirical Potential (LJ)
forces and energies to create training labels.
"""

import logging
import numpy as np
from typing import Dict, Optional
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.calculators.lj import LennardJones

from src.interfaces import Labeler

logger = logging.getLogger(__name__)


class ShiftedLennardJones(LennardJones):
    """LennardJones calculator with potential shift to ensure V(rc) = 0."""

    def calculate(
        self,
        atoms: Optional[Atoms] = None,
        properties=None,
        system_changes=None
    ):
        """Calculate properties, applying energy shift."""
        if properties is None:
            properties = ['energy']
        if system_changes is None:
            system_changes = ['positions', 'numbers', 'cell', 'pbc', 'charges', 'magmom']

        super().calculate(atoms, properties, system_changes)

        if 'energy' in self.results:
            epsilon = self.parameters.get('epsilon', 1.0)
            sigma = self.parameters.get('sigma', 1.0)
            rc = self.parameters.get('rc')

            if rc is not None:
                # Calculate shift value at cutoff
                # V(rc) = 4*eps * ((sigma/rc)^12 - (sigma/rc)^6)
                sr_cut = sigma / rc
                v_cut = 4.0 * epsilon * (sr_cut**12 - sr_cut**6)

                # Count pairs within cutoff
                calc_atoms = atoms if atoms is not None else self.atoms
                dists = calc_atoms.get_all_distances(mic=True)
                np.fill_diagonal(dists, np.inf)

                # Number of pairs (matrix has double counts)
                n_pairs = np.sum(dists < rc) / 2.0

                # Subtract total shift
                self.results['energy'] -= n_pairs * v_cut


class DeltaLabeler(Labeler):
    """Calculates the delta between DFT and reference (LJ) calculations."""

    def __init__(self, reference_calculator: Calculator, baseline_calculator: Calculator):
        """Initialize the DeltaLabeler.

        Args:
            reference_calculator: A configured ASE calculator for the ground truth (DFT).
            baseline_calculator: A configured ASE calculator for the baseline (e.g. LJ).
        """
        self.reference_calculator = reference_calculator
        self.baseline_calculator = baseline_calculator

    def label(self, structure: Atoms) -> Optional[Atoms]:
        """Compute delta energy, forces, and stress for a given structure.

        Args:
            structure: The atomic cluster to label.

        Returns:
            Atoms: The cluster with updated energy, forces, and stress representing the delta.
                   Returns None if the DFT calculation fails.
        """
        # Work on copies
        cluster_ref = structure.copy()
        cluster_base = structure.copy()

        # 1. Reference Calculation (DFT)
        try:
            cluster_ref.calc = self.reference_calculator
            e_ref = cluster_ref.get_potential_energy()
            f_ref = cluster_ref.get_forces()
            s_ref = cluster_ref.get_stress()
        except Exception as e:
            logger.warning(f"Reference (DFT) Calculation failed: {e}")
            return None

        # 2. Baseline Calculation (LJ)
        try:
            cluster_base.calc = self.baseline_calculator
            e_base = cluster_base.get_potential_energy()
            f_base = cluster_base.get_forces()
            s_base = cluster_base.get_stress()
        except Exception as e:
            logger.error(f"Baseline (LJ) Calculation failed: {e}")
            return None

        # 3. Compute Delta
        result_cluster = cluster_ref.copy()
        result_cluster.calc = None

        e_delta = e_ref - e_base
        f_delta = f_ref - f_base
        s_delta = s_ref - s_base

        # 4. Store Results
        result_cluster.info['energy'] = e_delta
        result_cluster.arrays['forces'] = f_delta
        result_cluster.info['stress'] = s_delta

        # Store Raw DFT
        result_cluster.info['energy_dft_raw'] = e_ref
        result_cluster.arrays['forces_dft_raw'] = f_ref
        # result_cluster.info['stress_dft_raw'] = s_ref # Optional

        # Weights
        result_cluster.info['energy_weight'] = 1.0
        result_cluster.info['virial_weight'] = 1.0

        return result_cluster
