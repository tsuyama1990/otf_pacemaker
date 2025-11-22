"""Tests for Active Learning module."""

import pytest
import numpy as np
import pandas as pd
from ase import Atoms
from src.active_learning import ClusterCarver, convert_to_pacemaker_format

def test_cluster_carver_init():
    carver = ClusterCarver(r_core=2.0, r_buffer=3.0)
    assert carver.r_core == 2.0
    assert carver.r_buffer == 3.0

def test_cluster_carver_extract():
    # Create a simple cubic lattice
    # 3x3x3 = 27 atoms. Distance between neighbors is 1.0.
    atoms = Atoms('Ar27',
                  positions=[(x, y, z) for x in range(3) for y in range(3) for z in range(3)],
                  cell=[3, 3, 3],
                  pbc=True)

    # Center at (1, 1, 1) -> index 13
    center_id = 13

    # r_core = 0.5 (only center)
    # r_buffer = 1.1 (center + 6 nearest neighbors)
    carver = ClusterCarver(r_core=0.5, r_buffer=1.1)
    cluster = carver.extract_cluster(atoms, center_id)

    # Verify cluster size
    # 1 center + 6 neighbors = 7 atoms
    assert len(cluster) == 7

    # Verify weights
    weights = cluster.arrays['forces_weight']
    assert np.sum(weights == 1.0) == 1
    assert np.sum(weights == 0.0) == 6

    # Verify the center atom has weight 1.0
    # Since we don't know the exact order (though extract_cluster implementation uses indices order),
    # we check if the atom at distance 0 from center (in the cluster frame) has weight 1.0.
    # But cluster positions are absolute. We know center was at (1,1,1).
    # Let's check distances from (1,1,1).
    dists = np.linalg.norm(cluster.positions - np.array([1,1,1]), axis=1)

    # Find index where dist is ~0
    center_idx = np.argmin(dists)
    assert weights[center_idx] == 1.0

def test_convert_to_pacemaker_format():
    atoms1 = Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.74]])
    atoms2 = Atoms('O2', positions=[[0, 0, 0], [0, 0, 1.2]])

    df = convert_to_pacemaker_format([atoms1, atoms2])

    assert isinstance(df, pd.DataFrame)
    assert 'ase_atoms' in df.columns
    assert len(df) == 2
    assert df.iloc[0]['ase_atoms'] == atoms1
    assert df.iloc[1]['ase_atoms'] == atoms2
