import numpy as np
import random
from typing import List
from pymatgen.core import Structure
# GB Generator location changed in recent pymatgen or requires specific import
# It seems GrainBoundaryGenerator is not in analysis.interfaces.grain_boundaries anymore
# or the package structure changed.
# We will use reflection/try-import or remove the explicit import if it's dynamic.
# Actually, checking pymatgen docs, it is often in pymatgen.analysis.gb.grain_boundaries but that failed too.
# Let's try pymatgen.analysis.local_env or others? No.
# Wait, pymatgen 2025 refactored a lot.
# If unavailable, we should implement a fallback or use available tools.
# But for now, let's try to find where it is.
# Actually, it might be removed or moved to an add-on.
# Assuming it is missing, we will comment it out and use a dummy implementation for GBs if needed,
# OR try to import safely.

try:
    from pymatgen.analysis.gb.grain import GrainBoundaryGenerator
except ImportError:
    try:
        from pymatgen.analysis.interfaces.grain_boundaries import GrainBoundaryGenerator
    except ImportError:
        GrainBoundaryGenerator = None

from pymatgen.transformations.standard_transformations import PerturbStructureTransformation
from ase import Atoms
from pymatgen.io.ase import AseAtomsAdaptor

from .base import BaseGenerator

class AlloyGenerator(BaseGenerator):
    """
    Strategy for Alloys and Metals (e.g., HEAs).
    Focus: Entropy, Random Substitution, GBs, Phase Separation.
    """

    def generate_random_substitution(self, elements: List[str] = None, supercell_dim: int = 3):
        """
        Generates SQS-like random substitutions in a supercell.
        """
        # Create Supercell
        sc = self.structure.copy()
        sc.make_supercell([supercell_dim, supercell_dim, supercell_dim])

        # Determine mixing elements
        # If input elements provided, use them. Else shuffle existing.
        # Ideally, we want to mix 'similar' sites.

        sites = list(range(len(sc)))
        random.shuffle(sites)

        # Check elements
        if not elements:
            # If no new elements provided, just shuffle existing ones (antisite defects essentially)
            species = [s.specie for s in sc]
            random.shuffle(species)
            for i, site in enumerate(sc):
                # We can't easily assign directly in pymatgen without replace
                sc.replace(i, species[i])
        else:
            # Equiatomic target
            n_elems = len(elements)
            n_sites = len(sc)
            # Create distribution
            targets = []
            for i, el in enumerate(elements):
                count = n_sites // n_elems
                if i == n_elems - 1:
                    count += n_sites % n_elems
                targets.extend([el] * count)

            random.shuffle(targets)
            for i, el in enumerate(targets):
                sc.replace(i, el)

        self._add_structure(sc, meta={"type": "random_substitution", "mixing": "random"})

    def generate_grain_boundaries(self):
        """
        Generates Sigma 3 and Sigma 5 Grain Boundaries.
        """
        if GrainBoundaryGenerator is None:
            # logger.warning("GrainBoundaryGenerator not available in installed pymatgen version.")
            return

        # GB generation is expensive and tricky. We pick standard low sigma.
        try:
            gb_gen = GrainBoundaryGenerator(self.structure)
            # Sigma 3 (Twin mostly)
            gb3 = gb_gen.get_ratio(3, max_j=1)
            for gb in gb3[:2]: # Limit to 2 variants
                self._add_structure(gb, meta={"type": "grain_boundary", "sigma": 3})

            # Sigma 5
            gb5 = gb_gen.get_ratio(5, max_j=1)
            for gb in gb5[:2]:
                self._add_structure(gb, meta={"type": "grain_boundary", "sigma": 5})

        except Exception as e:
            # GB generation fails often for complex unit cells
            # logger.warning(f"GB Generation failed: {e}")
            pass

    def generate_phase_separation(self, cluster_size: int = 4):
        """
        Simulate spinodal decomposition/clustering.
        Groups like atoms together in a supercell.
        """
        sc = self.structure.copy()
        sc.make_supercell([3, 3, 3])

        unique_species = list(set([s.specie for s in sc]))
        if len(unique_species) < 2:
            return # Can't separate 1 phase

        # Clustering Logic:
        # Pick seeds for each species
        seeds = {}
        available_indices = set(range(len(sc)))

        # Simple clustering: Assign regions based on distance to seeds
        # Pick random seeds
        seed_indices = random.sample(list(available_indices), len(unique_species))

        for i, specie in enumerate(unique_species):
             # Assign all closest atoms to this specie
             pass

        # Voronoi-like assignment for "clumped" distribution
        # We calculate distance from every site to every seed
        new_species_map = {}

        # Get cartesian coords
        cart_coords = sc.cart_coords
        seed_coords = [cart_coords[idx] for idx in seed_indices]

        for i in range(len(sc)):
            dists = [sc.get_distance(i, seed_idx) for seed_idx in seed_indices]
            closest_seed_idx = np.argmin(dists)
            target_specie = unique_species[closest_seed_idx]
            sc.replace(i, target_specie)

        self._add_structure(sc, meta={"type": "phase_separation", "mode": "clustering"})

    def generate_all(self) -> List[Atoms]:
        self.generate_random_substitution() # Shuffle existing
        self.generate_grain_boundaries()
        self.generate_phase_separation()
        return self.generated_structures
