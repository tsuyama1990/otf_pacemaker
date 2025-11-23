"""Delta Labeler Strategy."""

import logging
from typing import Optional
import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator

from src.core.interfaces import Labeler

logger = logging.getLogger(__name__)

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
