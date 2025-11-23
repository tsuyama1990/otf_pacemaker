import numpy as np
from typing import List
from pymatgen.core import Structure, Molecule
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import CrystalNN
from ase import Atoms
from pymatgen.io.ase import AseAtomsAdaptor

from .base import BaseGenerator

class MolecularGenerator(BaseGenerator):
    """
    Strategy for Molecular Crystals.
    Focus: Intramolecular distortion vs Intermolecular packing.
    """

    def _get_molecules(self, structure: Structure) -> List[List[int]]:
        """
        Identify molecules using Graph Analysis.
        Returns list of lists of site indices.
        """
        # CrystalNN is robust for bonding
        strategy = CrystalNN()
        sg = StructureGraph.with_local_env_strategy(structure, strategy)

        # Get subgraphs (molecules)
        # StructureGraph.get_subgraphs_as_molecules() returns Molecule objects
        # But we need indices to manipulate the original structure IN PLACE (or copy)
        # properly respecting PBC is hard with indices if molecules cross boundaries.

        # However, pymatgen handles this well.
        # Let's extract molecules and their original indices.
        # Because 'get_subgraphs_as_molecules' might not preserve mapping easily,
        # we stick to networkx connected components on the graph.

        import networkx as nx
        graph = sg.graph

        # Connected components are molecules
        molecules_indices = list(nx.connected_components(graph))
        return [list(m) for m in molecules_indices]

    def generate_intramolecular_distortion(self):
        """
        Distorts bonds/angles WITHIN molecules only.
        """
        # 1. Identify Molecules
        try:
            mol_indices_list = self._get_molecules(self.structure)
        except Exception:
            # Fallback if graph fails (e.g. not molecular)
            return

        sc = self.structure.copy()
        cart_coords = sc.cart_coords

        # 2. Apply distortion
        # We rattle atoms, but we want to shift them relative to the molecule center,
        # not the cell.

        # Simple approach: Rattle everything, but check connectivity?
        # No, "Intramolecular" means changing the molecule's shape.
        # "Intermolecular" means changing packing.

        # If we rattle all atoms, we do both.
        # To separate, we can:
        # A. Rattle atoms (Intra + Inter) -> Easy
        # B. Rigid body rotation/translation (Inter only) -> Harder
        # C. Scale molecular coordinates relative to COM (Intra only) -> This is what we want.

        distortion_factors = [0.95, 1.05] # Compress/Expand molecule

        for factor in distortion_factors:
            new_struct = sc.copy()
            new_coords = new_struct.cart_coords # (N, 3)

            for mol_indices in mol_indices_list:
                # Get COM of molecule
                # Need to handle PBC for COM calculation?
                # Pymatgen usually unwraps for molecules, but here we are in a periodic struct.
                # Simplification: Assume atoms in 'mol_indices' are relatively close
                # (CrystalNN usually handles connectivity).

                # Get coords of this molecule
                mol_coords = new_coords[mol_indices]

                # Calculate centroid
                centroid = np.mean(mol_coords, axis=0)

                # Scale distance from centroid
                vecs = mol_coords - centroid
                vecs *= factor

                # Update
                new_coords[mol_indices] = centroid + vecs

            # Update structure
            # modifying .cart_coords directly is not safe in pymatgen, use specialized setters or recreate
            for i, coord in enumerate(new_coords):
                new_struct[i].coords = coord

            self._add_structure(new_struct, meta={"type": "intramolecular_distortion", "factor": factor})

    def generate_high_pressure_packing(self):
        """
        Compresses cell volume to reduce intermolecular distances.

        NOTE: This strategy relies on the PreOptimizer using a Fixed-Cell relaxation (like standard BFGS).
        The cell is manually compressed here, and the subsequent optimization relaxes the atomic positions
        to accommodate the new density without expanding the cell back.
        """
        # Scaling the lattice vectors reduces ALL distances (Intra and Inter).
        # But Intra bonds are stiff, Inter are soft.
        # In reality, under pressure, Inter compresses more.
        # We simulate "High Pressure" simply by volumetric scaling.
        # The Pre-Optimization (if using a good potential) would relax the stiff bonds back
        # while keeping the soft contacts compressed (if we constrain cell).

        # However, our PreOptimizer relaxes everything including cell if we aren't careful.
        # But 'ase.optimize.BFGS' (used in PreOptimizer) by default only relaxes positions (Cell fixed).
        # So compressing the cell and relaxing positions is EXACTLY what we want.
        # The stiff molecular bonds will relax to equilibrium length,
        # but the molecules will be jammed closer together (high pressure packing).

        scales = [0.95, 0.90, 0.85] # 15% compression is huge

        for s in scales:
            new_struct = self.structure.copy()
            new_struct.scale_lattice(new_struct.volume * s)
            self._add_structure(new_struct, meta={"type": "high_pressure_packing", "scale": s})

    def generate_all(self) -> List[Atoms]:
        self.generate_intramolecular_distortion()
        self.generate_high_pressure_packing()
        return self.generated_structures
