
import pytest
import numpy as np
from ase import Atoms
from src.utils.structure import carve_cubic_cluster

def test_carve_cubic_cluster_simple():
    """Test carving from a simple cubic lattice."""
    # Create 3x3x3 simple cubic
    atoms = Atoms('H27', positions=np.indices((3, 3, 3)).reshape(3, -1).T, cell=[3, 3, 3], pbc=True)
    center_pos = np.array([1.0, 1.0, 1.0])

    # Carve a 2.0 box (should capture center and neighbors at distance 1? No, box is 2.0 side.)
    # Half box = 1.0.
    # Points at 1.0 +/- 1.0 => [0, 2].
    # So it should include 0, 1, 2 in each dimension.

    cluster, indices = carve_cubic_cluster(atoms, center_pos, box_size=2.1, apply_pbc=False)

    # Expected: 27 atoms if box covers everything?
    # center is 1,1,1. box size 2.1.
    # limits: 1 +/- 1.05 -> [-0.05, 2.05].
    # So indices 0, 1, 2 fit.
    # All 27 atoms should be in.
    assert len(cluster) == 27

def test_carve_cubic_cluster_small():
    # 4x4x4. center at 2,2,2.
    atoms = Atoms('H64', positions=np.indices((4, 4, 4)).reshape(3, -1).T, cell=[4, 4, 4], pbc=True)
    center_pos = np.array([1.5, 1.5, 1.5])

    # Box size 1.0. Half box 0.5.
    # range: [1.0, 2.0].
    # points are integers. 0, 1, 2, 3.
    # inside: 1 and 2 are on the boundary?
    # mask: abs(dist) <= 0.5.
    # if atom at 1.0: dist = -0.5. abs=0.5. <= 0.5 is True.
    # if atom at 2.0: dist = 0.5. abs=0.5. True.
    # So 1 and 2 should be included in all dims.
    # 2x2x2 = 8 atoms.

    cluster, indices = carve_cubic_cluster(atoms, center_pos, box_size=1.0, apply_pbc=False)
    assert len(cluster) == 8

def test_carve_pbc_wrapping():
    """Test that carving wraps around PBC correctly."""
    # 1D line for simplicity: 0, 1, 2, 3, 4. Cell=5.
    # Center at 0. Box size = 2.0 (radius 1).
    # Should get 4 (dist -1), 0 (dist 0), 1 (dist 1).
    atoms = Atoms('H5', positions=[[0,0,0], [1,0,0], [2,0,0], [3,0,0], [4,0,0]], cell=[5, 5, 5], pbc=True)
    center_pos = np.array([0.0, 0.0, 0.0])

    cluster, indices = carve_cubic_cluster(atoms, center_pos, box_size=2.1, apply_pbc=True)

    assert len(cluster) == 3
    # Check original indices
    expected_indices = {0, 1, 4}
    assert set(indices) == expected_indices

    # Check centering: center atom should be at box/2 = 1.05
    # The atom 0 was at 0. It is the center. So it should be at 1.05.
    # The atom 1 was at 1. Vector 1-0 = 1. Pos = 1 + 1.05 = 2.05.
    # The atom 4 was at 4. Vector 4-0 = -1 (MIC). Pos = -1 + 1.05 = 0.05.

    positions = cluster.positions
    # Sort by x
    x_pos = sorted(positions[:, 0])
    assert np.allclose(x_pos, [0.05, 1.05, 2.05])

def test_fixed_buffer():
    """Test that buffer_width creates constraints."""
    atoms = Atoms('H3', positions=[[0,0,0], [0.9,0,0], [0.1,0,0]], cell=[10,10,10], pbc=True)
    center_pos = np.array([0,0,0])
    # Box size 2.0. Half box 1.0.
    # All 3 atoms are inside.

    # Buffer 0.2. Inner limit = 1.0 - 0.2 = 0.8.
    # Atom 0: rel_pos 0. Inside.
    # Atom 2 (0.1): rel_pos 0.1. Inside.
    # Atom 1 (0.9): rel_pos 0.9. > 0.8. Fixed.

    cluster, indices = carve_cubic_cluster(atoms, center_pos, box_size=2.0, buffer_width=0.15, apply_pbc=False)

    assert len(cluster) == 3

    # Check constraints
    from ase.constraints import FixAtoms
    constraints = cluster.constraints
    assert len(constraints) > 0
    fixed_indices = constraints[0].index

    # Only the atom originally at 0.9 should be fixed.
    # Original indices: 0 (center), 1 (0.9), 2 (0.1).
    # In cluster, order depends on mask, usually preserves order.
    # 0 -> 0, 1 -> 1, 2 -> 2.
    # 0.9 is at index 1.

    assert 1 in fixed_indices
    assert 0 not in fixed_indices
    assert 2 not in fixed_indices
