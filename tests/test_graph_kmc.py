import pytest
import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms
from src.engines.kmc import OffLatticeKMCEngine
from src.core.config import KMCParams, ALParams
from ase.neighborlist import NeighborList

def create_test_system():
    # Create a simplified slab: 3x3x2
    a = 2.5
    positions = []

    # Layer 1 (z=0)
    for x in range(3):
        for y in range(3):
            positions.append([x*a, y*a, 0.0])

    # Layer 2 (z=a)
    for x in range(3):
        for y in range(3):
            positions.append([x*a, y*a, a])

    substrate_len = len(positions)

    dimer_z = 2.0 * a
    positions.append([1.0*a, 1.0*a, dimer_z]) # Adsorbate 1
    positions.append([1.0*a + 2.0, 1.0*a, dimer_z]) # Adsorbate 2

    atoms = Atoms('Cu' * len(positions), positions=positions)
    atoms.set_cell([3*a, 3*a, 4*a])
    atoms.set_pbc(True)

    return atoms, substrate_len

def get_engine():
    kmc_params = KMCParams(
        active=True,
        adsorbate_cn_cutoff=4,
        cluster_connectivity_cutoff=3.0,
        box_size=6.0
    )
    al_params = ALParams(
        gamma_threshold=0.1, n_clusters=1, r_core=1.0, box_size=10.0,
        initial_potential="dummy", potential_yaml_path="dummy"
    )
    return OffLatticeKMCEngine(kmc_params, al_params)

def test_identify_moving_cluster():
    atoms, sub_len = create_test_system()
    adsorbate_idx_1 = sub_len
    adsorbate_idx_2 = sub_len + 1

    engine = get_engine()
    cutoff = engine.kmc_params.cluster_connectivity_cutoff

    # Prepare CSR
    nl = NeighborList([cutoff/2.0]*len(atoms), self_interaction=False, bothways=True, skin=0.0)
    nl.update(atoms)
    matrix = nl.get_connectivity_matrix(sparse=True)
    csr_matrix = matrix.tocsr()
    indices = csr_matrix.indices
    indptr = csr_matrix.indptr
    cns = np.diff(indptr)

    # Verify CNs
    assert cns[adsorbate_idx_1] == 2
    assert cns[13] >= 5

    # Test Cluster Identification
    cluster = engine._identify_moving_cluster(
        atoms,
        center_idx=adsorbate_idx_1,
        indices=indices,
        indptr=indptr,
        cns=cns
    )

    assert len(cluster) == 2
    assert adsorbate_idx_1 in cluster
    assert adsorbate_idx_2 in cluster
    assert 13 not in cluster

def test_carve_cluster_strict_fixing():
    atoms, sub_len = create_test_system()
    adsorbate_indices = [sub_len, sub_len + 1]

    engine = get_engine()

    cluster, index_map = engine._carve_cluster(atoms, adsorbate_indices)

    # Verify cluster contains the adsorbates
    local_adsorbates = []
    for i, global_idx in enumerate(index_map):
        if global_idx in adsorbate_indices:
            local_adsorbates.append(i)

    assert len(local_adsorbates) == 2

    # Verify Constraints
    constraints = cluster.constraints
    assert len(constraints) > 0

    fixed_indices = []
    for c in constraints:
        if isinstance(c, FixAtoms):
            fixed_indices.extend(c.index)

    # Check that all non-adsorbate atoms are fixed
    for i in range(len(cluster)):
        if i not in local_adsorbates:
            assert i in fixed_indices, f"Substrate atom {i} was not fixed!"
        else:
            assert i not in fixed_indices, f"Adsorbate atom {i} was fixed!"

if __name__ == "__main__":
    test_identify_moving_cluster()
    test_carve_cluster_strict_fixing()
