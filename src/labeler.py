"""Labeling module for calculating target properties.

This module calculates the difference between DFT and Empirical Potential (LJ)
forces and energies to create training labels for the delta-learning model.
"""

from ase import Atoms
from ase.calculators.espresso import Espresso
from ase.calculators.lj import LennardJones


class DeltaLabeler:
    """Calculates the delta between DFT and reference (LJ) calculations.

    Attributes:
        qe_calculator: Configured ASE Espresso calculator for DFT.
        lj_params: Dictionary containing Lennard-Jones parameters.
    """

    def __init__(self, qe_calculator: Espresso, lj_params: dict[str, float]):
        """Initialize the DeltaLabeler.

        Args:
            qe_calculator: An instance of ase.calculators.espresso.Espresso.
            lj_params: A dictionary containing LJ parameters (epsilon, sigma, cutoff).
        """
        self.qe_calculator = qe_calculator
        self.lj_params = lj_params

    def compute_delta(self, cluster: Atoms) -> Atoms:
        """Compute delta energy and forces for a given cluster.

        This method performs two calculations:
        1. DFT calculation using the provided Espresso calculator.
        2. LJ calculation using the internal LJ calculator.

        The results are stored in the returned Atoms object with:
        - info['energy'] = E_DFT - E_LJ (Delta Energy)
        - arrays['forces'] = F_DFT - F_LJ (Delta Forces)
        - info['energy_dft_raw'] = E_DFT
        - arrays['forces_dft_raw'] = F_DFT
        - info['energy_weight'] = 1.0 (Energy learning enabled)

        Args:
            cluster: The atomic cluster to label (expected to be the relaxed Small-Cell).

        Returns:
            Atoms: The cluster with updated energy and forces representing the delta.

        Raises:
            Exception: If the DFT calculation fails (re-raises the underlying exception).
        """
        # Work on copies to ensure isolation and prevent side effects (e.g. reordering)
        cluster_dft = cluster.copy()
        cluster_lj = cluster.copy()

        # --- 1. DFT Calculation ---
        cluster_dft.calc = self.qe_calculator
        try:
            # Trigger calculation
            e_dft = cluster_dft.get_potential_energy()
            f_dft = cluster_dft.get_forces()
        except Exception as e:
            # Explicitly re-raise the exception to notify the caller of failure
            raise e

        # --- 2. LJ Calculation ---
        # LennardJones calculator uses 'rc' for cutoff
        lj_kwargs = {
            'epsilon': self.lj_params.get('epsilon', 1.0),
            'sigma': self.lj_params.get('sigma', 1.0),
            'rc': self.lj_params.get('cutoff', 2.5)
        }

        cluster_lj.calc = LennardJones(**lj_kwargs)

        e_lj = cluster_lj.get_potential_energy()
        f_lj = cluster_lj.get_forces()

        # --- 3. Compute Delta ---
        # Use the DFT cluster as the base for the result to preserve its structure/properties
        result_cluster = cluster_dft.copy()
        result_cluster.calc = None  # Detach calculator

        e_delta = e_dft - e_lj
        f_delta = f_dft - f_lj

        # --- 4. Store Results ---
        result_cluster.info['energy'] = e_delta
        result_cluster.arrays['forces'] = f_delta

        # Store Raw DFT
        result_cluster.info['energy_dft_raw'] = e_dft
        result_cluster.arrays['forces_dft_raw'] = f_dft

        # --- 5. Energy Learning ---
        # Small-Cell uses periodic boundary conditions and relaxed structures,
        # so energy is physically meaningful and should be trained.
        result_cluster.info['energy_weight'] = 1.0

        return result_cluster
