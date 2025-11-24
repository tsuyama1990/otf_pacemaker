"""Delta Labeler Strategy."""

import logging
from typing import Optional, Dict
import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator

from src.core.interfaces import Labeler

logger = logging.getLogger(__name__)

class DeltaLabeler(Labeler):
    """Calculates the delta between DFT and reference (LJ) calculations."""

    def __init__(self, reference_calculator: Calculator, baseline_calculator: Calculator,
                 e0_dict: Optional[Dict[str, float]] = None, outlier_energy_max: float = 10.0):
        """Initialize the DeltaLabeler.

        Args:
            reference_calculator: A configured ASE calculator for the ground truth (DFT).
            baseline_calculator: A configured ASE calculator for the baseline (e.g. LJ).
            e0_dict: Dictionary of isolated atomic energies.
            outlier_energy_max: Max energy delta per atom (eV) allowed before discarding.
        """
        self.reference_calculator = reference_calculator
        self.baseline_calculator = baseline_calculator
        self.e0_dict = e0_dict or {}
        self.outlier_energy_max = outlier_energy_max

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
            s_ref = cluster_ref.get_stress() # Voigt form (6,) in eV/A^3
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

        # 3. Calculate Offset (E0)
        e_offset = 0.0
        if self.e0_dict:
            try:
                e_offset = sum(self.e0_dict[s] for s in structure.get_chemical_symbols())
            except KeyError as e:
                logger.error(f"Missing E0 for element: {e}")
                return None

        # 4. Compute Delta
        # Target = DFT - LJ - E0
        e_delta = e_ref - e_base - e_offset
        f_delta = f_ref - f_base
        s_delta = s_ref - s_base

        # [Filter] Outlier Check
        if abs(e_delta) / len(structure) > self.outlier_energy_max:
             logger.warning(f"Discarding outlier: Delta Energy {e_delta/len(structure):.2f} eV/atom exceeds threshold {self.outlier_energy_max}.")
             return None

        result_cluster = cluster_ref.copy()
        result_cluster.calc = None

        # Convert Stress to Virial (Extensive) [eV]
        # Virial W = - Sigma * V
        volume = structure.get_volume()
        virial_delta = -1.0 * s_delta * volume

        # 5. Store Results
        result_cluster.info['energy'] = e_delta
        result_cluster.arrays['forces'] = f_delta

        # Store as 'virial' and remove 'stress' to avoid ambiguity
        result_cluster.info['virial'] = virial_delta
        if 'stress' in result_cluster.info:
            del result_cluster.info['stress']

        # Store Raw DFT
        result_cluster.info['energy_dft_raw'] = e_ref
        result_cluster.arrays['forces_dft_raw'] = f_ref
        result_cluster.info['e0_offset'] = e_offset

        # Weights
        result_cluster.info['energy_weight'] = 1.0
        result_cluster.info['virial_weight'] = 1.0

        return result_cluster
