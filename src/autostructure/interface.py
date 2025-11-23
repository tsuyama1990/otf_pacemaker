import numpy as np
import copy
import random
from typing import List, Union
from ase import Atoms
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from ase.build import stack, rotate

from .base import BaseGenerator

class InterfaceBuilder(BaseGenerator):
    """
    Builder for Heterostructures and Interfaces.
    Modes: Epitaxial Strain Scan, Incoherent/Twist, Gradient Intermix.
    """

    def __init__(self, structure_a: Union[Atoms, Structure], structure_b: Union[Atoms, Structure]):
        # We need two structures. BaseGenerator init takes one.
        # We'll use A as the "base" but store B.
        super().__init__(structure_a)

        if isinstance(structure_b, Atoms):
            self.structure_b = AseAtomsAdaptor.get_structure(structure_b)
        else:
            self.structure_b = structure_b

        self.atoms_a = AseAtomsAdaptor.get_atoms(self.structure)
        self.atoms_b = AseAtomsAdaptor.get_atoms(self.structure_b)

    def epitaxial_strain_scan(self):
        """
        Stacks A and B and scans separation distance.
        Averages lattice constants (simplistic) or strains one to match other.
        """
        # Simplest epitaxial match: Strain B to match A's cell in X/Y
        # Note: This requires cells to be commensurate.
        # For an "Auto" builder, we assume the user provided commensurate cells
        # or we just stack them if they are close.
        # If not, we might generate garbage, but 'PreOptimizer' cleans up collisions.

        # We use ASE stack. It generally requires equal cell dimensions in non-stack directions
        # to produce a valid periodic system, but 'stack' function might average or fail.
        # Actually ase.build.stack simply stacks them. If cell differs, it might be weird.
        # Let's force B's cell (0, 1) to match A's.

        a_cell = self.atoms_a.get_cell()
        b_cell = self.atoms_b.get_cell()

        # Create strained B
        atoms_b_strained = self.atoms_b.copy()

        # Naive scaling of B to match A in x and y (indices 0 and 1)
        # Assuming Z (index 2) is the stacking direction
        # We scale B's atomic positions and cell vectors

        # Ratios
        # Use lengths() for safer scaling factors
        a_lengths = a_cell.lengths()
        b_lengths = b_cell.lengths()

        # Avoid div by zero
        bx = b_lengths[0] if b_lengths[0] > 1e-3 else 1.0
        by = b_lengths[1] if b_lengths[1] > 1e-3 else 1.0

        scale_x = a_lengths[0] / bx
        scale_y = a_lengths[1] / by

        # Construct new cell for B
        # We scale the basis vectors directly
        new_cell_b = np.array(b_cell)
        new_cell_b[0] *= scale_x
        new_cell_b[1] *= scale_y

        atoms_b_strained.set_cell(new_cell_b, scale_atoms=True)

        # Scan separation distance
        d0 = 2.0 # Nominal interface dist
        separations = [0.8*d0, 1.0*d0, 1.5*d0]

        for d in separations:
            try:
                # Stack along z (axis=2)
                combined = stack(self.atoms_a, atoms_b_strained, axis=2, distance=d)
                self._add_structure(combined, meta={"type": "interface_epitaxial", "separation": d})
            except Exception as e:
                # Stack failed (incompatible cells)
                pass

    def incoherent_twist(self):
        """
        Rotates B relative to A and stacks (Twist Boundary).
        """
        angles = [30.0, 45.0, 90.0]

        for angle in angles:
            atoms_b_rotated = self.atoms_b.copy()
            # Rotate B around Z axis
            atoms_b_rotated.rotate(angle, 'z', center='COM')

            # Stacking incoherent lattices usually requires a large supercell to be periodic.
            # If we just stack them, the periodic boundary conditions will be broken at the edges.
            # However, for training data (local environments), a broken PBC at the cell edge
            # might be acceptable if the interface area is valid.
            # OR we treat it as a cluster (vacuum in all directions).
            # Let's add vacuum to be safe.

            atoms_b_rotated.center(vacuum=5.0, axis=2)
            atoms_a_centered = self.atoms_a.copy()
            atoms_a_centered.center(vacuum=5.0, axis=2)

            # Just place them near each other
            # We can't use 'stack' easily if cells don't match.
            # We'll create a new atoms object combining both.

            combined = atoms_a_centered.copy()
            # Shift B to sit on top of A
            z_offset = atoms_a_centered.positions[:, 2].max() + 2.5
            atoms_b_rotated.translate([0, 0, z_offset])

            combined.extend(atoms_b_rotated)
            combined.set_pbc([True, True, True]) # Technically false periodicity in XY if twisted

            # This effectively makes a Moire pattern if viewed locally
            self._add_structure(combined, meta={"type": "interface_twist", "angle": angle})

    def gradient_intermix(self, layers: int = 2):
        """
        Swaps atoms near the interface to create a gradient.
        """
        # First create a clean stack
        try:
            # Re-use epitaxial logic or just stack
            atoms_b_strained = self.atoms_b.copy()
            # Naive match
            a_cell = self.atoms_a.get_cell()
            b_cell = self.atoms_b.get_cell()

            a_lengths = a_cell.lengths()
            b_lengths = b_cell.lengths()

            bx = b_lengths[0] if b_lengths[0] > 1e-3 else 1.0
            by = b_lengths[1] if b_lengths[1] > 1e-3 else 1.0

            scale_x = a_lengths[0] / bx
            scale_y = a_lengths[1] / by

            new_cell_b = np.array(b_cell)
            new_cell_b[0] *= scale_x
            new_cell_b[1] *= scale_y
            atoms_b_strained.set_cell(new_cell_b, scale_atoms=True)

            combined = stack(self.atoms_a, atoms_b_strained, axis=2, distance=2.0)
        except:
            return

        # Identify Interface Z
        # A is at bottom, B is top.
        z_coords = combined.positions[:, 2]
        # Interface is roughly where A ends and B begins
        n_a = len(self.atoms_a)
        z_interface = (z_coords[n_a-1] + z_coords[n_a]) / 2.0

        # Select atoms within 'layers' distance
        # Assuming layers ~ 2-3 Angstroms per layer
        width = layers * 2.5

        indices_near = [i for i, z in enumerate(z_coords) if abs(z - z_interface) < width]

        # Swap 50% of them randomly?
        # To make a gradient, we should swap based on probability P(z)
        # But random swaps in the zone is "Gradient Intermix" enough for training data.

        n_swap = len(indices_near) // 2
        if n_swap > 0:
            to_swap = random.sample(indices_near, n_swap)
            # Shuffle their positions (effectively swapping species if we assign positions)
            # Or better: shuffle the species (atomic numbers) of these atoms

            # Extract symbols/numbers
            nums = combined.numbers
            subset_nums = nums[to_swap]
            np.random.shuffle(subset_nums) # Shuffle in place
            nums[to_swap] = subset_nums
            combined.numbers = nums

        self._add_structure(combined, meta={"type": "interface_gradient", "width": width})

    def generate_all(self) -> List[Atoms]:
        self.epitaxial_strain_scan()
        self.incoherent_twist()
        self.gradient_intermix()
        return self.generated_structures
