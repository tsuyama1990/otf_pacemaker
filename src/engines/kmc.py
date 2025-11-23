"""KMC Engine Implementation.

This module implements the Off-Lattice KMC Engine using ASE Dimer method
and Pacemaker potential with k-step uncertainty checking.
"""

import logging
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional, List, Tuple, Dict, Any, Union

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

from src.core.interfaces import KMCEngine, KMCResult
from src.core.enums import KMCStatus
from src.core.config import KMCParams, ALParams
from src.utils.structure import carve_cubic_cluster

logger = logging.getLogger(__name__)

def _setup_calculator(atoms: Atoms, potential_path: str):
    """Attach the Pacemaker calculator to the atoms.

    Helper function for workers.
    """
    if PyACECalculator is None:
        # Check if we are mocking/testing
        if hasattr(atoms, "calc") and atoms.calc is not None:
            return
        raise ImportError("pyace module is required for KMC Engine.")

    # We assume potential_path is a .yace or .yaml file
    calc = PyACECalculator(potential_path)
    atoms.calc = calc

def _select_active_atoms(atoms: Atoms, params: KMCParams) -> List[int]:
    """Select active atoms for KMC displacement based on strategy."""
    indices = []

    # 1. Surface selection
    if "surface" in params.active_region_mode:
        z_coords = atoms.positions[:, 2]
        surface_indices = np.where(z_coords > params.active_z_cutoff)[0]
    else:
        surface_indices = np.arange(len(atoms))

    # 2. Species selection
    if "species" in params.active_region_mode:
        symbols = np.array(atoms.get_chemical_symbols())
        species_mask = np.isin(symbols, params.active_species)
        species_indices = np.where(species_mask)[0]
    else:
        species_indices = np.arange(len(atoms))

    # Intersection
    active_indices = np.intersect1d(surface_indices, species_indices)

    if len(active_indices) == 0:
        return []

    return active_indices.tolist()


def _run_single_search(
    initial_full_atoms: Atoms,
    potential_path: str,
    kmc_params: KMCParams,
    al_params: ALParams,
    seed: int
) -> Union[KMCResult, Tuple[float, np.ndarray, List[int]]]:
    """Run a single dimer search on a LOCAL CLUSTER.

    Returns:
        KMCResult if UNCERTAIN or FAILED.
        Tuple[float, np.ndarray, List[int]] (barrier, displacement_vector, map) if SUCCESS.
    """
    np.random.seed(seed)

    # 1. Select Target Atom
    active_indices = _select_active_atoms(initial_full_atoms, kmc_params)
    if not active_indices:
        return KMCResult(status=KMCStatus.NO_EVENT, structure=initial_full_atoms, metadata={"reason": "No active atoms found"})

    if kmc_params.selection_bias == "coordination":
        cutoff = 3.5
        nl = NeighborList([cutoff/2]*len(initial_full_atoms), self_interaction=False, bothways=True)
        nl.update(initial_full_atoms)
        cns = np.array([len(nl.get_neighbors(i)[0]) for i in active_indices])
        weights = (1.0 / (cns + 0.1)) ** kmc_params.bias_strength
        probs = weights / np.sum(weights)
        target_idx = np.random.choice(active_indices, p=probs)
    else:
        target_idx = np.random.choice(active_indices)

    # 2. Carve Cluster
    center_pos = initial_full_atoms.positions[target_idx]
    cluster_atoms, index_array = carve_cubic_cluster(
        atoms=initial_full_atoms,
        center_pos=center_pos,
        box_size=kmc_params.box_size,
        buffer_width=kmc_params.buffer_width,
        apply_pbc=False
    )
    index_map = index_array.tolist()

    try:
        cluster_target_idx = index_map.index(target_idx)
    except ValueError:
        return KMCResult(status=KMCStatus.NO_EVENT, structure=initial_full_atoms, metadata={"reason": "Target atom lost in carving"})

    # Setup calculator
    _setup_calculator(cluster_atoms, potential_path)

    # 3. Perturb
    search_atoms = cluster_atoms.copy()
    _setup_calculator(search_atoms, potential_path)

    if kmc_params.move_type == "cluster":
        dists = search_atoms.get_distances(cluster_target_idx, range(len(search_atoms)), mic=False)
        local_cluster_indices = np.where(dists < kmc_params.cluster_radius)[0]

        fixed_indices = set()
        for c in search_atoms.constraints:
            if isinstance(c, FixAtoms):
                fixed_indices.update(c.index)

        coherent_disp = np.random.normal(0, kmc_params.search_radius, 3)

        for idx in local_cluster_indices:
            if idx in fixed_indices:
                continue
            noise = np.random.normal(0, kmc_params.search_radius * 0.2, 3)
            search_atoms.positions[idx] += coherent_disp + noise

    else:
        displacement = np.random.normal(0, kmc_params.search_radius, 3)
        search_atoms.positions[cluster_target_idx] += displacement

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
        opt.run(steps=kmc_params.check_interval)
        current_step += kmc_params.check_interval

        if opt.converged():
            converged = True
            break

        try:
            gamma_vals = search_atoms.calc.results.get('gamma')
            if gamma_vals is None and hasattr(search_atoms.calc, 'get_property'):
                 try:
                     gamma_vals = search_atoms.calc.get_property('gamma', search_atoms)
                 except:
                     pass

            if gamma_vals is not None:
                max_gamma = np.max(gamma_vals)
                if max_gamma > al_params.gamma_threshold:
                    uncertain = True
                    break
        except Exception:
            pass

    if uncertain:
        return KMCResult(
            status=KMCStatus.UNCERTAIN,
            structure=search_atoms,
            metadata={"reason": "High Gamma in Saddle Search", "gamma": max_gamma}
        )

    if converged:
        # Relax to product
        product_atoms = search_atoms.copy()
        _setup_calculator(product_atoms, potential_path)

        prod_opt = FIRE(product_atoms, logfile=None)
        prod_converged = False
        prod_uncertain = False
        p_step = 0

        while p_step < max_steps:
            prod_opt.run(steps=kmc_params.check_interval)
            p_step += kmc_params.check_interval

            if prod_opt.converged():
                prod_converged = True
                break

            try:
                g_vals = product_atoms.calc.results.get('gamma')
                if g_vals is not None and np.max(g_vals) > al_params.gamma_threshold:
                     prod_uncertain = True
                     break
            except:
                pass

        if prod_uncertain:
             return KMCResult(
                status=KMCStatus.UNCERTAIN,
                structure=product_atoms,
                metadata={"reason": "High Gamma in Product Relaxation"}
             )

        if prod_converged:
            e_saddle = dimer_atoms.get_potential_energy()
            e_initial = cluster_atoms.get_potential_energy()

            barrier = e_saddle - e_initial

            if barrier > 0.01:
                # Calculate displacement vector
                displacement = product_atoms.positions - cluster_atoms.positions
                return (barrier, displacement, index_map)

    return KMCResult(status=KMCStatus.NO_EVENT, structure=initial_full_atoms)


class OffLatticeKMCEngine(KMCEngine):
    """Off-Lattice KMC Engine with k-step uncertainty checking."""

    def __init__(self, kmc_params: KMCParams, al_params: ALParams):
        """Initialize the KMC Engine.

        Args:
            kmc_params: KMC configuration parameters.
            al_params: Active Learning parameters (for gamma threshold).
        """
        self.kmc_params = kmc_params
        self.al_params = al_params

    def _setup_calculator(self, atoms: Atoms, potential_path: str):
        """Attach the Pacemaker calculator to the atoms."""
        _setup_calculator(atoms, potential_path)

    def run_step(self, initial_atoms: Atoms, potential_path: str) -> KMCResult:
        """Run a single KMC step: Saddle point search and event selection.
        """

        # Working copy for minimization (Full System)
        atoms = initial_atoms.copy()
        self._setup_calculator(atoms, potential_path)

        # 1. Minimize current basin (ensure we are at a minimum)
        try:
            opt = FIRE(atoms, logfile=None)
            opt.run(fmax=self.kmc_params.dimer_fmax, steps=200)
        except Exception as e:
            logger.warning(f"Initial minimization failed: {e}")

        # Prepare for parallel execution
        atoms.calc = None

        found_events: List[Tuple[float, np.ndarray, List[int]]] = []
        uncertain_results: List[KMCResult] = []

        logger.info(f"Starting {self.kmc_params.n_searches} parallel KMC searches with {self.kmc_params.n_workers} workers.")

        with ProcessPoolExecutor(max_workers=self.kmc_params.n_workers) as executor:
            futures = []
            for i in range(self.kmc_params.n_searches):
                seed = np.random.randint(0, 1000000)
                futures.append(
                    executor.submit(
                        _run_single_search,
                        atoms, # Full atoms
                        potential_path,
                        self.kmc_params,
                        self.al_params,
                        seed
                    )
                )

            for future in as_completed(futures):
                try:
                    result = future.result()
                    if isinstance(result, KMCResult):
                        if result.status == KMCStatus.UNCERTAIN:
                            uncertain_results.append(result)
                    elif isinstance(result, tuple):
                        # (barrier, displacement, index_map)
                        found_events.append(result)
                except Exception as e:
                    logger.error(f"KMC Search worker failed: {e}")

        if uncertain_results:
            logger.info(f"Collected {len(uncertain_results)} uncertain searches. Triggering AL.")
            return uncertain_results[0]

        if not found_events:
            return KMCResult(status=KMCStatus.NO_EVENT, structure=initial_atoms)

        # Rate R_i = v * exp(-E_a / k_B T)
        k_B = 8.617333262e-5
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
        selected_event = found_events[selected_idx]

        barrier, displacement, index_map = selected_event

        logger.info(f"KMC Event Selected: Barrier={barrier:.3f} eV, Time Step={dt:.3e} s")

        final_full_structure = atoms.copy()

        # Apply displacement
        # index_map maps cluster_idx -> full_idx
        # displacement is array of shape (n_cluster_atoms, 3)
        full_indices = index_map
        final_full_structure.positions[full_indices] += displacement

        # Wrap? No, Fixed Boundary logic implies we didn't cross boundaries of the box.
        # But we should wrap the full system if it's periodic?
        # Standard to wrap.
        if final_full_structure.pbc.any():
            final_full_structure.wrap()

        final_full_structure.calc = None

        return KMCResult(
            status=KMCStatus.SUCCESS,
            structure=final_full_structure,
            time_step=dt,
            metadata={"barrier": barrier, "rates": rates}
        )
