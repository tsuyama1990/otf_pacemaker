"""KMC Engine Implementation.

This module implements the Off-Lattice KMC Engine using ASE Dimer method
and Pacemaker potential with k-step uncertainty checking.
It uses Graph-Based Local Cluster identification (Numba-optimized) to move molecules/clusters naturally.
"""

import logging
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional, List, Tuple, Dict, Any, Union
from scipy import constants
from scipy import sparse

from ase import Atoms
from ase.mep import MinModeAtoms, DimerControl
from ase.optimize import FIRE
from ase.neighborlist import NeighborList
from ase.constraints import FixAtoms

try:
    from pyace import PyACECalculator
except ImportError:
    # Fallback for environments where pyace is not installed (e.g. testing)
    PyACECalculator = None

# Numba Handling
try:
    from numba import jit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

from src.core.interfaces import KMCEngine, KMCResult
from src.core.enums import KMCStatus
from src.core.config import KMCParams, ALParams

logger = logging.getLogger(__name__)

# Constants (Externalized)
KB_EV = constants.k / constants.e  # Boltzmann constant in eV/K ~ 8.617e-5

def _setup_calculator(atoms: Atoms, potential_path: str):
    """Attach the Pacemaker calculator to the atoms.

    Helper function for workers.
    """
    if PyACECalculator is None:
        if hasattr(atoms, "calc") and atoms.calc is not None:
            return
        raise ImportError("pyace module is required for KMC Engine.")

    calc = PyACECalculator(potential_path)
    atoms.calc = calc

if HAS_NUMBA:
    @jit(nopython=True)
    def _bfs_traversal_numba(
        start_node: int,
        indices: np.ndarray,
        indptr: np.ndarray,
        cns: np.ndarray,
        cn_cutoff: int,
        num_atoms: int
    ) -> np.ndarray:
        """
        BFS traversal using Numba.
        """
        visited = np.zeros(num_atoms, dtype=np.bool_)
        visited[start_node] = True

        queue = np.empty(num_atoms, dtype=np.int32)
        q_head = 0
        q_tail = 0

        queue[q_tail] = start_node
        q_tail += 1

        cluster_buffer = np.empty(num_atoms, dtype=np.int32)
        cluster_count = 0

        cluster_buffer[cluster_count] = start_node
        cluster_count += 1

        while q_head < q_tail:
            current = queue[q_head]
            q_head += 1

            start_idx = indptr[current]
            end_idx = indptr[current + 1]

            for k in range(start_idx, end_idx):
                neighbor = indices[k]

                if not visited[neighbor]:
                    if cns[neighbor] < cn_cutoff:
                        visited[neighbor] = True
                        queue[q_tail] = neighbor
                        q_tail += 1

                        cluster_buffer[cluster_count] = neighbor
                        cluster_count += 1

        return cluster_buffer[:cluster_count]


def _bfs_traversal_scipy(
    start_node: int,
    indices: np.ndarray,
    indptr: np.ndarray,
    cns: np.ndarray,
    cn_cutoff: int,
    num_atoms: int
) -> np.ndarray:
    """
    BFS traversal using Scipy (Fallback).
    Explicit implementation to ensure performance when Numba is missing.
    """
    # Create valid mask
    valid_mask = cns < cn_cutoff
    # Start node is assumed valid by caller (or we check)
    if not valid_mask[start_node]:
        return np.array([start_node], dtype=np.int32)

    data = np.ones(len(indices), dtype=np.int8)
    adj = sparse.csr_matrix((data, indices, indptr), shape=(num_atoms, num_atoms))

    # We want to restrict traversal to 'valid_mask' nodes.
    # M' = D * M * D where D is diagonal matrix with 1 for valid, 0 for invalid.
    diag_data = valid_mask.astype(np.int8)
    D = sparse.diags(diag_data)

    # Filtered Adjacency
    adj_filtered = D @ adj @ D

    # Now BFS
    # breadth_first_order returns (node_array, predecessors)
    nodes, _ = sparse.csgraph.breadth_first_order(
        adj_filtered,
        i_start=start_node,
        directed=False,
        return_predecessors=True
    )

    return nodes.astype(np.int32)


class OffLatticeKMCEngine(KMCEngine):
    """Off-Lattice KMC Engine with k-step uncertainty checking and Numba optimization."""

    def __init__(self, kmc_params: KMCParams, al_params: ALParams):
        """Initialize the KMC Engine.

        Args:
            kmc_params: KMC configuration parameters.
            al_params: Active Learning parameters (for gamma threshold).
        """
        self.kmc_params = kmc_params
        self.al_params = al_params

        if not HAS_NUMBA:
            # As per requirements: "Numba非依存時の明示的なエラーハンドリングまたはScipy代替実装"
            # We are providing the Scipy alternative, but let's log explicitly.
            logger.warning("Numba not installed. Using Scipy fallback for graph traversal.")
            # Verify scipy is available
            try:
                import scipy.sparse.csgraph
            except ImportError:
                 raise RuntimeError("Numba is missing AND Scipy is missing. One is required for performance.")

    def _identify_moving_cluster(
        self,
        atoms: Atoms,
        center_idx: int,
        indices: np.ndarray,
        indptr: np.ndarray,
        cns: np.ndarray
    ) -> List[int]:
        """Identify the connected cluster using Numba-optimized BFS or Scipy fallback."""

        # Check center first
        if cns[center_idx] >= self.kmc_params.adsorbate_cn_cutoff:
             return [center_idx]

        if HAS_NUMBA:
            cluster_arr = _bfs_traversal_numba(
                center_idx,
                indices,
                indptr,
                cns,
                self.kmc_params.adsorbate_cn_cutoff,
                len(atoms)
            )
        else:
            # Fallback to Scipy implementation
            cluster_arr = _bfs_traversal_scipy(
                center_idx,
                indices,
                indptr,
                cns,
                self.kmc_params.adsorbate_cn_cutoff,
                len(atoms)
            )

        return sorted(cluster_arr.tolist())

    def _carve_cluster(
        self,
        full_atoms: Atoms,
        moving_indices: List[int]
    ) -> Tuple[Atoms, List[int]]:
        """Carve a cubic cluster around the moving molecule and apply strict fixing."""
        box_size = self.kmc_params.box_size

        # 1. Calculate Geometric Center
        cluster_pos = full_atoms.positions[moving_indices]
        if full_atoms.pbc.any():
            ref_pos = cluster_pos[0]
            d = cluster_pos - ref_pos
            cell = full_atoms.get_cell()
            pbc = full_atoms.get_pbc()
            for i in range(3):
                if pbc[i]:
                    l = cell[i, i]
                    d[:, i] -= np.round(d[:, i] / l) * l
            cluster_pos = ref_pos + d

        center_pos = np.mean(cluster_pos, axis=0)

        # 2. Extract Atoms
        vectors = full_atoms.positions - center_pos

        if full_atoms.pbc.any():
            cell = full_atoms.get_cell()
            inv_cell = np.linalg.inv(cell)
            scaled = np.dot(vectors, inv_cell)
            scaled -= np.round(scaled)
            vectors = np.dot(scaled, cell)

        half_box = box_size / 2.0
        mask = (np.abs(vectors) <= half_box).all(axis=1)
        mask[moving_indices] = True

        cluster_indices = np.where(mask)[0]
        cluster = full_atoms[mask].copy()

        # 3. Center and Disable PBC
        extracted_vectors = vectors[mask]
        cluster.positions = extracted_vectors + half_box
        cluster.set_cell([box_size, box_size, box_size])
        cluster.set_pbc(False)

        # 4. Strict Fixed Boundary
        global_to_local = {global_idx: local_idx for local_idx, global_idx in enumerate(cluster_indices)}

        indices_to_fix = []
        for global_idx in cluster_indices:
            if global_idx not in moving_indices:
                indices_to_fix.append(global_to_local[global_idx])

        if indices_to_fix:
            cluster.set_constraint(FixAtoms(indices=indices_to_fix))

        return cluster, cluster_indices.tolist()

    def _run_single_search(
        self,
        full_atoms_snapshot: Atoms,
        potential_path: str,
        target_idx: int,
        indices: np.ndarray,
        indptr: np.ndarray,
        cns: np.ndarray,
        seed: int
    ) -> Union[KMCResult, Tuple[float, np.ndarray, List[int]]]:
        """Worker function for single search."""
        np.random.seed(seed)

        # 1. Identify Cluster
        moving_indices = self._identify_moving_cluster(
            full_atoms_snapshot, target_idx, indices, indptr, cns
        )

        # 2. Carve
        cluster, index_map = self._carve_cluster(full_atoms_snapshot, moving_indices)

        local_moving_indices = []
        for i, global_idx in enumerate(index_map):
            if global_idx in moving_indices:
                local_moving_indices.append(i)

        if not local_moving_indices:
             return KMCResult(status=KMCStatus.NO_EVENT, structure=full_atoms_snapshot, metadata={"reason": "Carving Error"})

        _setup_calculator(cluster, potential_path)

        # 3. Perturb (Cooperative Move)
        search_atoms = cluster.copy()
        _setup_calculator(search_atoms, potential_path)

        coherent_disp = np.random.normal(0, self.kmc_params.search_radius, 3)

        for idx in local_moving_indices:
            noise = np.random.normal(0, self.kmc_params.search_radius * 0.2, 3)
            search_atoms.positions[idx] += coherent_disp + noise

        # 4. Dimer Search
        dimer_control = DimerControl(
            logfile=None,
            eigenmode_method='displacement',
            f_rot_min=0.1,
            f_rot_max=1.0
        )

        dimer_atoms = MinModeAtoms(search_atoms, dimer_control)
        dimer_atoms.displace()

        opt = FIRE(dimer_atoms, logfile=None)

        converged = False
        uncertain = False
        max_steps = 1000
        current_step = 0
        max_gamma = 0.0

        while current_step < max_steps:
            opt.run(steps=self.kmc_params.check_interval)
            current_step += self.kmc_params.check_interval

            if opt.converged():
                converged = True
                break

            # Check Uncertainty
            try:
                calc = search_atoms.calc
                gamma_vals = None
                if hasattr(calc, "results"):
                    gamma_vals = calc.results.get('gamma')
                if gamma_vals is None and hasattr(calc, 'get_property'):
                    try: gamma_vals = calc.get_property('gamma', search_atoms)
                    except: pass

                if gamma_vals is not None:
                    max_gamma = np.max(gamma_vals)
                    if max_gamma > self.al_params.gamma_threshold:
                        uncertain = True
                        break
            except Exception:
                pass

        if uncertain:
            return KMCResult(
                status=KMCStatus.UNCERTAIN,
                structure=search_atoms,
                metadata={"reason": "High Gamma Saddle", "gamma": max_gamma}
            )

        if converged:
            product_atoms = search_atoms.copy()
            _setup_calculator(product_atoms, potential_path)

            prod_opt = FIRE(product_atoms, logfile=None)
            prod_opt.run(fmax=self.kmc_params.dimer_fmax, steps=500)

            e_saddle = dimer_atoms.get_potential_energy()
            e_initial = cluster.get_potential_energy()
            barrier = e_saddle - e_initial

            if barrier > 0.01:
                displacement = product_atoms.positions - cluster.positions
                return (barrier, displacement, index_map)

        return KMCResult(status=KMCStatus.NO_EVENT, structure=full_atoms_snapshot)


    def select_active_candidates(self, atoms: Atoms, cns: np.ndarray) -> np.ndarray:
        """Select candidate atoms for KMC searches based on configuration."""
        species_mask = np.ones(len(atoms), dtype=bool)
        surface_mask = np.ones(len(atoms), dtype=bool)

        mode = self.kmc_params.active_region_mode

        if mode in ["species", "surface_and_species"]:
            if self.kmc_params.active_species:
                 symbols = np.array(atoms.get_chemical_symbols())
                 species_mask = np.isin(symbols, self.kmc_params.active_species)

        if mode in ["surface", "surface_and_species"]:
            z_coords = atoms.positions[:, 2]
            surface_mask = z_coords > self.kmc_params.active_z_cutoff

        cn_mask = cns < self.kmc_params.adsorbate_cn_cutoff

        valid_mask = species_mask & surface_mask & cn_mask
        candidate_indices = np.where(valid_mask)[0]
        return candidate_indices

    def run_step(self, initial_atoms: Atoms, potential_path: str) -> KMCResult:
        """Run a single KMC step."""

        atoms = initial_atoms.copy()
        _setup_calculator(atoms, potential_path)

        try:
            opt = FIRE(atoms, logfile=None)
            opt.run(fmax=self.kmc_params.dimer_fmax, steps=200)
        except Exception:
            pass

        atoms.calc = None

        cutoff = self.kmc_params.cluster_connectivity_cutoff
        nl = NeighborList(
            [cutoff/2.0]*len(atoms),
            self_interaction=False,
            bothways=True,
            skin=0.0
        )
        nl.update(atoms)

        matrix = nl.get_connectivity_matrix(sparse=True)
        csr_matrix = matrix.tocsr()
        indices = csr_matrix.indices
        indptr = csr_matrix.indptr
        cns = np.diff(indptr)

        candidate_indices = self.select_active_candidates(atoms, cns)

        if len(candidate_indices) == 0:
            logger.warning("No valid KMC candidates.")
            return KMCResult(status=KMCStatus.NO_EVENT, structure=initial_atoms)

        n_select = min(self.kmc_params.n_searches, len(candidate_indices))
        selected_targets = np.random.choice(candidate_indices, size=n_select, replace=False)

        logger.info(f"Dispatching {n_select} parallel searches...")

        found_events = []
        uncertain_results = []

        with ProcessPoolExecutor(max_workers=self.kmc_params.n_workers) as executor:
            futures = []
            for target_idx in selected_targets:
                seed = np.random.randint(0, 1000000)
                futures.append(executor.submit(
                    self._run_single_search,
                    atoms,
                    potential_path,
                    target_idx,
                    indices,
                    indptr,
                    cns,
                    seed
                ))

            for future in as_completed(futures):
                try:
                    res = future.result()
                    if isinstance(res, KMCResult) and res.status == KMCStatus.UNCERTAIN:
                        uncertain_results.append(res)
                    elif isinstance(res, tuple):
                        found_events.append(res)
                except Exception as e:
                    logger.error(f"Worker failed: {e}")

        if uncertain_results:
             return uncertain_results[0]

        if not found_events:
             return KMCResult(status=KMCStatus.NO_EVENT, structure=initial_atoms)

        # Rate Calculation
        k_B = KB_EV
        T = self.kmc_params.temperature
        v = self.kmc_params.prefactor

        rates = []
        for barrier, _, _ in found_events:
            r = v * np.exp(-barrier / (k_B * T))
            rates.append(r)

        total_rate = sum(rates)
        if total_rate == 0:
             return KMCResult(status=KMCStatus.NO_EVENT, structure=initial_atoms)

        dt = -np.log(np.random.random()) / total_rate
        selected_idx = np.random.choice(len(rates), p=np.array(rates)/total_rate)

        barrier, displacement, index_map = found_events[selected_idx]

        final_atoms = atoms.copy()
        final_atoms.positions[index_map] += displacement
        if final_atoms.pbc.any():
            final_atoms.wrap()

        return KMCResult(
            status=KMCStatus.SUCCESS,
            structure=final_atoms,
            time_step=dt,
            metadata={"barrier": barrier}
        )
