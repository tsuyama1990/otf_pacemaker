import numpy as np
from ase import Atoms
from ase.calculators.emt import EMT
from ase.calculators.lj import LennardJones
from ase.optimize import BFGS
from ase.data import atomic_numbers, covalent_radii

class PreOptimizer:
    """
    Safety valve to perform geometric sanity checks and basic relaxation
    before expensive DFT calculations.
    """

    def __init__(self, fmax=0.1, steps=200, mic_distance=0.8):
        """
        Args:
            fmax (float): Maximum force threshold for relaxation.
            steps (int): Maximum number of relaxation steps.
            mic_distance (float): Minimum interatomic distance threshold (Å) for discarding.
        """
        self.fmax = fmax
        self.steps = steps
        self.mic_distance = mic_distance

        # Elements supported by ASE's EMT calculator
        self.emt_elements = {"Al", "Cu", "Ag", "Au", "Ni", "Pd", "Pt", "C", "N", "O", "H"}

    def _get_lj_parameters(self, atoms: Atoms):
        """
        Generate rough Lennard-Jones parameters based on covalent radii.
        This is not for physical accuracy but for purely repulsive overlap removal.
        """
        # Universal coarse parameters
        # epsilon: depth of the potential well (eV) - kept small to avoid strong binding artifacts
        # sigma: distance where potential is zero (Å) - roughly 2 * radius

        epsilon = 0.1 # Weak interaction

        # Determine average sigma for the system or per-pair?
        # ASE LennardJones is a simple pair potential. It takes sigma/epsilon as scalars
        # (applying to all pairs) or we can use the 'rc' argument effectively.
        # However, standard ASE LJ assumes single sigma/epsilon for the whole system.
        # For mixed systems, this is poor, but for "overlap removal", we mainly care about
        # the largest atoms not overlapping.

        # A better approach for mixed species safety valve is to use a repulsive-only potential
        # or just pick the smallest sigma to allow some closeness, or average.
        # Let's use the average covalent radius to estimate sigma.

        radii = [covalent_radii[Z] for Z in atoms.numbers]
        avg_radius = np.mean(radii)
        sigma = 2.0 * avg_radius * 0.8909 # sigma is roughly r_min / 2^(1/6)

        return sigma, epsilon

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

        # Check if we can use EMT (all elements supported)
        unique_elements = set(atoms.get_chemical_symbols())

        if unique_elements.issubset(self.emt_elements):
            calc = EMT()
        else:
            # Fallback to LJ
            sigma, epsilon = self._get_lj_parameters(atoms)
            # rc cut-off: 3*sigma is standard
            calc = LennardJones(epsilon=epsilon, sigma=sigma, rc=3.0*sigma)

        atoms.calc = calc

        try:
            dyn = BFGS(atoms, logfile=None) # Suppress log output
            dyn.run(fmax=self.fmax, steps=self.steps)
        except Exception as e:
            # If relaxation fails (e.g., explosion), we catch it
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
