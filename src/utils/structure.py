"""Structure manipulation utilities."""

import numpy as np
from typing import Tuple, Optional, List
from ase import Atoms
from ase.constraints import FixAtoms

def carve_cubic_cluster(
    atoms: Atoms,
    center_pos: np.ndarray,
    box_size: float,
    buffer_width: Optional[float] = None,
    apply_pbc: bool = False
) -> Tuple[Atoms, np.ndarray]:
    """Extract a cubic cluster from a larger structure (with PBC handling).

    Args:
        atoms: The full source structure.
        center_pos: The position (Cartesian) to center the cluster on.
        box_size: The side length of the cubic box.
        buffer_width: If set, fixes atoms within this buffer distance from the box edge.
                      (Used for fixed boundary conditions).
        apply_pbc: If True, the resulting cluster will have PBC=True and be wrapped.
                   If False, PBC=False.

    Returns:
        Tuple[Atoms, np.ndarray]:
            - The carved cluster atoms.
            - The indices of these atoms in the original structure.
    """
    if atoms.pbc.any():
        # Create a ghost atom at center_pos to use ASE's MIC distance tool
        # Or simpler: just use get_distances from a reference point if we had an index.
        # But we have center_pos.
        # We need to find distances from center_pos respecting PBC.

        # We can't use get_distances directly with an arbitrary position easily without adding a dummy atom.
        # Let's iterate or assume the user handles MIC if providing pos.
        # Actually, standard approach:
        # 1. Calculate vector from center_pos to all atoms.
        # 2. Apply MIC to these vectors relative to the *full* cell.

        diff = atoms.positions - center_pos
        # MIC:
        cell = atoms.get_cell()
        # ASE's find_mic requires cell.
        # But for an orthogonal box it is simple. For general triclinic, use ase.geometry.find_mic

        from ase.geometry import find_mic
        vectors, _ = find_mic(diff, cell, pbc=atoms.pbc)

    else:
        vectors = atoms.positions - center_pos

    half_box = box_size / 2.0

    # Mask for atoms inside the cubic box [-L/2, L/2]
    # We check if the vector components are within the half box.
    mask = (np.abs(vectors) <= half_box).all(axis=1)

    # Indices in the original system
    indices = np.where(mask)[0]

    if len(indices) == 0:
        raise ValueError(f"No atoms found in box of size {box_size} around {center_pos}")

    cluster = atoms[mask].copy()

    # Recenter so that center_pos maps to [half_box, half_box, half_box]
    cluster.positions = vectors[mask] + half_box

    cluster.set_cell([box_size, box_size, box_size])
    cluster.set_pbc(apply_pbc)

    if apply_pbc:
        cluster.wrap()

    # Apply Fixed Boundary Conditions (buffer) if requested
    if buffer_width is not None and buffer_width > 0:
        rel_pos = cluster.positions - np.array([half_box, half_box, half_box])
        # Recalculate relative position. If PBC is on, we should be careful.
        # Usually buffer is used with Fixed BC (PBC=False).
        # But if PBC=True, fixing atoms at the edge might be weird unless it's for some specific reason.
        # Assuming Fixed BC context here.

        # Logic: if atom is close to the box wall (distance to center > half_box - buffer)
        inner_limit = half_box - buffer_width

        # Note: box is cubic aligned with axes.
        fixed_mask = (np.abs(rel_pos) > inner_limit).any(axis=1)
        fixed_indices = np.where(fixed_mask)[0]

        if len(fixed_indices) > 0:
            # Check existing constraints?
            existing_constraints = cluster.constraints
            new_constraint = FixAtoms(indices=fixed_indices)
            cluster.set_constraint(existing_constraints + [new_constraint])

    return cluster, indices
