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

        Args:
            cluster: The atomic cluster to label.

        Returns:
            Atoms: The cluster with updated energy and forces representing the delta.

        Raises:
            Exception: If the DFT calculation fails (re-raises the underlying exception).
        """
        # --- 1. DFT Calculation ---
        cluster.calc = self.qe_calculator
        try:
            # Trigger calculation
            e_dft = cluster.get_potential_energy()
            f_dft = cluster.get_forces()
        except Exception as e:
            # Explicitly re-raise the exception to notify the caller of failure
            raise e

        # Backup raw DFT values
        cluster.info['energy_dft_raw'] = e_dft
        cluster.arrays['forces_dft_raw'] = f_dft.copy()

        # --- 2. LJ Calculation ---
        # Initialize LJ calculator with stored parameters
        # LennardJones calculator uses 'rc' for cutoff usually, but let's check standard usage.
        # ase.calculators.lj.LennardJones(epsilon=1.0, sigma=1.0, rc=None, ro=None)
        # Our config uses 'cutoff', which likely maps to 'rc'.

        lj_kwargs = {
            'epsilon': self.lj_params.get('epsilon', 1.0),
            'sigma': self.lj_params.get('sigma', 1.0),
            'rc': self.lj_params.get('cutoff', 2.5)
        }

        cluster.calc = LennardJones(**lj_kwargs)

        e_lj = cluster.get_potential_energy()
        f_lj = cluster.get_forces()

        # --- 3. Compute Delta ---
        e_delta = e_dft - e_lj
        f_delta = f_dft - f_lj

        # --- 4. Store Delta as Primary Labels ---
        # We do not set atoms.calc to None or a specific calculator that holds these,
        # instead we store them directly in info/arrays as if they were results.
        # However, calling get_potential_energy() again might trigger a recalc if a calculator is attached.
        # Pacemaker typically reads from info/arrays directly if loaded from pickle.
        # To be safe, we can remove the calculator so ASE doesn't try to recompute.
        cluster.calc = None

        cluster.info['energy'] = e_delta
        cluster.arrays['forces'] = f_delta

        return cluster
