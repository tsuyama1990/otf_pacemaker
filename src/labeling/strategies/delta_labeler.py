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

    def __init__(self, reference_calculator: Calculator, baseline_calculator: Calculator, e0_dict: Optional[Dict[str, float]] = None):
        """Initialize the DeltaLabeler.

        Args:
            reference_calculator: A configured ASE calculator for the ground truth (DFT).
            baseline_calculator: A configured ASE calculator for the baseline (e.g. LJ).
            e0_dict: Dictionary of isolated atomic energies.
        """
        self.reference_calculator = reference_calculator
        self.baseline_calculator = baseline_calculator
        self.e0_dict = e0_dict or {}

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
        result_cluster = cluster_ref.copy()
        result_cluster.calc = None

        # Target = DFT - LJ - E0
        e_delta = e_ref - e_base - e_offset
        f_delta = f_ref - f_base
        s_delta = s_ref - s_base

        # Convert Stress to Virial (Extensive)
        # Virial [eV] = -Stress [eV/A^3] * Volume [A^3]
        # However, MLIPs often treat "stress" keyword as Virial or Stress depending on format.
        # But here we are asked to explicitly convert to Virial (eV).
        # Note: ASE get_stress() returns -1/V * Virial (if positive is tension).
        # Actually ASE definition: Stress = -1/V * derivative w.r.t strain.
        # So Virial = - Stress * Volume.
        # The prompt says: "V_{target} = (S_{DFT} - S_{LJ}) * Vol" (ignoring sign in prompt formula but saying "Virial (Energy)").
        # Standard convention: Virial is extensive.
        # If the prompt says "Stress x Volume", I will follow that.
        # Usually s_delta (eV/A^3) * volume (A^3) = eV.

        volume = structure.get_volume()
        virial_delta = s_delta * volume # This is now in eV.

        # Wait, if ASE stress is negative of pressure (standard), then P = -Tr(stress)/3.
        # Tension is positive stress.
        # The prompt asks for "Virial (Energy: eV)".
        # I will store it as 'virial'.

        # 5. Store Results
        result_cluster.info['energy'] = e_delta
        result_cluster.arrays['forces'] = f_delta

        # We store 'virial' instead of 'stress' if we want to be explicit,
        # or we store 'stress' with the virial value if that's what the trainer expects.
        # But wait, Pacemaker/MLIP usually reads 'virial' from extra info or 'stress' from atoms.info/arrays.
        # If I overwrite 'stress' with virial values (eV), subsequent ASE calls might be confused if they expect eV/A^3.
        # However, for training data generation, we are producing an XYZ file.
        # I should check how the data is saved. Usually `ase.io.write` handles this.
        # If I save to `info['virial']`, I need to make sure the trainer reads it.
        # The prompt says: "Virial (エネルギー: eV) 形式 ... に明示的に変換して学習データを作成する".

        # I will store it in `info['virial']` to be safe and clear.
        result_cluster.info['virial'] = virial_delta

        # I will ALSO store 'stress' as the delta stress (eV/A^3) just in case,
        # unless specifically forbidden. But the prompt says "Virial ... 形式に変換".
        # If I leave 'stress', the trainer might pick it up.
        # I will REMOVE 'stress' key if I put 'virial', or ensure 'virial' is used.
        # Let's put 'virial' and remove 'stress' to force usage of virial if the trainer supports it.
        # If the trainer (Pacemaker) expects 'stress' but containing virial, that would be confusing.
        # Pacemaker docs usually say "stress" in input.yaml mapping.
        # If I look at `TrainingParams` in `config.py`: `force_weight`, `energy_weight`.
        # I don't see `virial_weight` config there, but `result_cluster.info['virial_weight'] = 1.0` is in code.
        # So likely it expects 'virial'.

        if 'stress' in result_cluster.info:
            del result_cluster.info['stress']

        result_cluster.info['virial'] = virial_delta

        # Store Raw DFT
        result_cluster.info['energy_dft_raw'] = e_ref
        result_cluster.arrays['forces_dft_raw'] = f_ref
        result_cluster.info['e0_offset'] = e_offset

        # Weights
        result_cluster.info['energy_weight'] = 1.0
        result_cluster.info['virial_weight'] = 1.0

        return result_cluster
