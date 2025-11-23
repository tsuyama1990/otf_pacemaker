"""Shifted Lennard-Jones Calculator."""

import numpy as np
from typing import Optional
from ase import Atoms
from ase.calculators.lj import LennardJones

class ShiftedLennardJones(LennardJones):
    """LennardJones calculator with potential shift to ensure V(rc) = 0."""

    def calculate(
        self,
        atoms: Optional[Atoms] = None,
        properties=None,
        system_changes=None
    ):
        """Calculate properties, applying energy shift."""
        if properties is None:
            properties = ['energy']
        if system_changes is None:
            system_changes = ['positions', 'numbers', 'cell', 'pbc', 'charges', 'magmom']

        super().calculate(atoms, properties, system_changes)

        if 'energy' in self.results:
            epsilon = self.parameters.get('epsilon', 1.0)
            sigma = self.parameters.get('sigma', 1.0)
            rc = self.parameters.get('rc')

            if rc is not None:
                # Calculate shift value at cutoff
                # V(rc) = 4*eps * ((sigma/rc)^12 - (sigma/rc)^6)
                sr_cut = sigma / rc
                v_cut = 4.0 * epsilon * (sr_cut**12 - sr_cut**6)

                # Count pairs within cutoff
                calc_atoms = atoms if atoms is not None else self.atoms
                dists = calc_atoms.get_all_distances(mic=True)
                np.fill_diagonal(dists, np.inf)

                # Number of pairs (matrix has double counts)
                n_pairs = np.sum(dists < rc) / 2.0

                # Subtract total shift
                self.results['energy'] -= n_pairs * v_cut
