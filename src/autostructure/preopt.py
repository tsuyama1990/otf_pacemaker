import numpy as np
from typing import Dict, List, Set, Optional
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.calculators.emt import EMT
from ase.calculators.lj import LennardJones
from ase.optimize import BFGS

class PreOptimizer:
    """
    Safety valve to perform geometric sanity checks and basic relaxation
    before expensive DFT calculations.

    NOTE: Default behavior uses ase.optimize.BFGS, which performs Fixed-Cell relaxation
    (only atomic positions are relaxed, lattice vectors remain constant).
    Strategies like MolecularGenerator's 'high_pressure_packing' rely on this behavior
    to maintain compressed cell volumes while relaxing atomic overlaps.
    """

    def __init__(
        self,
        lj_params: Dict[str, float],
        emt_elements: Optional[Set[str]] = None,
        fmax: float = 0.1,
        steps: int = 200,
        mic_distance: float = 0.8
    ):
        """
        Args:
            lj_params (Dict[str, float]): LJ parameters (sigma, epsilon, cutoff).
            emt_elements (Set[str], optional): Elements allowed for EMT.
                                               Defaults to standard ASE EMT elements.
            fmax (float): Maximum force threshold for relaxation.
            steps (int): Maximum number of relaxation steps.
            mic_distance (float): Minimum interatomic distance threshold (Å) for discarding.
        """
        self.lj_params = lj_params
        self.fmax = fmax
        self.steps = steps
        self.mic_distance = mic_distance

        # Elements supported by ASE's EMT calculator (Standard set)
        if emt_elements is None:
            self.emt_elements = {"Al", "Cu", "Ag", "Au", "Ni", "Pd", "Pt", "C", "N", "O", "H"}
        else:
            self.emt_elements = set(emt_elements)

    def get_calculator(self, atoms: Atoms) -> Calculator:
        """
        Returns an appropriate lightweight calculator (EMT or LJ) for the given structure.
        """
        unique_elements = set(atoms.get_chemical_symbols())

        if unique_elements.issubset(self.emt_elements):
            return EMT()
        else:
            # Use injected LJ parameters
            return LennardJones(
                epsilon=self.lj_params['epsilon'],
                sigma=self.lj_params['sigma'],
                rc=self.lj_params['cutoff']
            )

    def run_pre_optimization(self, atoms: Atoms) -> Atoms:
        """
        Run pre-optimization on the given structure.

        Returns:
            Atoms: The relaxed structure.

        Raises:
            ValueError: If the structure is physically unsound (too close atoms) after relaxation.
        """
        # Work on a copy
        atoms = atoms.copy()

        # Attach calculator
        atoms.calc = self.get_calculator(atoms)

        try:
            dyn = BFGS(atoms, logfile=None) # Suppress log output
            dyn.run(fmax=self.fmax, steps=self.steps)
        except Exception:
            # If relaxation fails (e.g., explosion), we catch it
            # But we proceed to check distances. If it exploded, distances might be weird or fine.
            pass

        # Final Sanity Check: Distance Matrix
        # We need to account for PBC.
        if len(atoms) > 1:
            # get_all_distances with mic=True handles PBC
            dists = atoms.get_all_distances(mic=True)
            # Mask diagonal (self-distance is 0)
            np.fill_diagonal(dists, np.inf)
            min_dist = np.min(dists)

            if min_dist < self.mic_distance:
                # Discard
                raise ValueError(f"Structure discarded: Atoms too close ({min_dist:.2f} Å < {self.mic_distance} Å)")

        return atoms
