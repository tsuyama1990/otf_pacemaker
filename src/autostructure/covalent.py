import numpy as np
import random
from typing import List
from pymatgen.core import Structure
from ase import Atoms
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.transformations.standard_transformations import DeformStructureTransformation

from .base import BaseGenerator

class CovalentGenerator(BaseGenerator):
    """
    Strategy for Covalent Materials (Si, C, GaN).
    Focus: Shear strain, Interstitials, Amorphous/Quench.
    """

    def generate_shear_strain(self):
        """
        Applies strong shear strain to alter bond angles.
        """
        # Shear matrix
        # [[1, s, 0], [0, 1, 0], [0, 0, 1]]
        shears = [0.1, 0.2, 0.3] # Radical shears

        for s in shears:
            deformation = [[1.0, s, 0.0],
                           [0.0, 1.0, 0.0],
                           [0.0, 0.0, 1.0]]

            t = DeformStructureTransformation(deformation=deformation)
            sheared = t.apply_transformation(self.structure.copy())
            self._add_structure(sheared, meta={"type": "shear_strain", "magnitude": s})

    def generate_interstitials(self):
        """
        Inserts atoms into interstitial sites (Voronoi voids).
        """
        # from pymatgen.analysis.defects.generators import InterstitialGenerator

        # This is expensive, use simple Voronoi finding or just random placement
        # The user requested "Interstitials... high energy state"
        # We can try to put atoms in holes.

        # Simple approach: Find biggest void
        # To avoid heavy dependencies for Voronoi, we can use a randomized probe approach
        # or just use pymatgen if available.

        # Pymatgen Defect generators are evolving.
        # Let's use a simpler heuristic: Place atom at random position, check distance.
        # If distance > threshold, it's a valid interstitial.

        sc = self.structure.copy()
        sc.make_supercell([2,2,2])

        # Attempt to insert 1 to 5 atoms
        attempts = 0
        inserted = 0
        max_inserts = 5

        temp_struct = sc.copy()

        while inserted < max_inserts and attempts < 100:
            attempts += 1
            # Random fractional coord
            f_coords = np.random.rand(3)

            # Check distances
            # Create a dummy site to check
            temp_struct.append("H", f_coords, coords_are_cartesian=False) # Dummy H

            # Check overlap
            # Use distance matrix: last row, exclude self (last col)
            # Actually get_distance is cleaner for single site
            # dists = [temp_struct.get_distance(len(temp_struct)-1, i) for i in range(len(temp_struct)-1)]

            # Using distance_matrix (N, N)
            dm = temp_struct.distance_matrix
            dists = dm[-1][:-1]

            temp_struct.pop() # Remove dummy

            min_dist = np.min(dists)

            if min_dist > 1.5: # Valid hole (approx)
                # Insert a host atom (e.g. the first element type)
                el = self.structure.composition.elements[0]
                temp_struct.append(el, f_coords, coords_are_cartesian=False)
                inserted += 1

                # Save intermediate
                self._add_structure(temp_struct.copy(), meta={"type": "interstitial", "count": inserted})

    def generate_amorphous_quench(self):
        """
        Simulates amorphous structure by high-T randomization + quench (Pre-opt).
        """
        # "Melt" by randomizing positions strongly
        atoms = AseAtomsAdaptor.get_atoms(self.structure).repeat((2,2,2))

        # Random displacement (Rattle) with large amplitude
        # For Covalent, bonds break at > 0.5-1.0 A usually.
        # We want to scramble it.

        rattle_amplitude = 0.8 # Very high
        atoms.rattle(stdev=rattle_amplitude)

        # The "Quench" is effectively the Pre-optimization step (Relaxation)
        # which pulls atoms back to local minima (amorphous state).

        # We assume the pre-optimizer will be called in _add_structure
        self._add_structure(atoms, meta={"type": "amorphous_quench", "rattle_amp": rattle_amplitude})

    def generate_all(self) -> List[Atoms]:
        self.generate_shear_strain()
        self.generate_interstitials()
        self.generate_amorphous_quench()
        return self.generated_structures
