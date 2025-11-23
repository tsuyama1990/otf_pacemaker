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

try:
    from pyace import PyACECalculator
except ImportError:
    # Fallback for environments where pyace is not installed (e.g. testing)
    PyACECalculator = None

from src.core.interfaces import KMCEngine, KMCResult
from src.core.enums import KMCStatus
from src.core.config import KMCParams, ALParams

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
        # Select atoms with z > cutoff
        # We assume z is the surface normal direction.
        z_coords = atoms.positions[:, 2]
        # Or relative to max z? The user said "above z > cutoff".
        # Usually implies absolute if cutoff is explicit, or relative to slab bottom?
        # Let's assume absolute cutoff as per config (active_z_cutoff).
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
        # Fallback or empty
        # If customized mode, maybe user meant something else.
        # But if empty, return empty list (caller handles it)
        return []

    return active_indices.tolist()


def _run_single_search(
    initial_atoms: Atoms,
    potential_path: str,
    kmc_params: KMCParams,
    al_params: ALParams,
    seed: int
) -> Union[KMCResult, Tuple[float, Atoms]]:
    """Run a single dimer search in a separate process.

    Returns:
        KMCResult if UNCERTAIN or FAILED.
        Tuple[float, Atoms] (barrier, structure) if SUCCESS.
    """
    np.random.seed(seed)

    # Setup calculator locally
    atoms = initial_atoms.copy()
    _setup_calculator(atoms, potential_path)

    # 1. Select Target Atom
    active_indices = _select_active_atoms(atoms, kmc_params)
    if not active_indices:
        # Fallback to all atoms if active region is empty?
        # Or log warning and return nothing?
        # User said "fallback to selecting surface atoms or log a warning and skip KMC."
        # Here we just return None/empty result?
        # Let's try to fallback to top layer atoms if possible, or just fail this search.
        # Let's fail this search.
        return KMCResult(status=KMCStatus.NO_EVENT, structure=atoms, metadata={"reason": "No active atoms found"})

    # Bias Selection
    if kmc_params.selection_bias == "coordination":
        # Calculate CN
        cutoff = 3.5 # Standard cutoff for CN? Or use covalent radii.
        # Simple cutoff is faster.
        nl = NeighborList([cutoff/2]*len(atoms), self_interaction=False, bothways=True)
        nl.update(atoms)
        cns = np.array([len(nl.get_neighbors(i)[0]) for i in active_indices])

        # Weight P ~ (1/CN)^alpha. Avoid div by zero.
        # Use (1 / (CN + 0.1))^alpha
        weights = (1.0 / (cns + 0.1)) ** kmc_params.bias_strength
        probs = weights / np.sum(weights)
        target_idx = np.random.choice(active_indices, p=probs)
    else:
        target_idx = np.random.choice(active_indices)

    # 2. Perturb (Cluster or Single)
    search_atoms = atoms.copy()
    _setup_calculator(search_atoms, potential_path)

    if kmc_params.move_type == "cluster":
        # Find neighbors of target_idx within cluster_radius
        dists = search_atoms.get_distances(target_idx, range(len(search_atoms)), mic=True)
        cluster_indices = np.where(dists < kmc_params.cluster_radius)[0]

        # Apply displacement to all cluster atoms
        # "randomized displacement vector to the *entire cluster* (with some random noise per atom)"
        # Coherent shift + noise
        coherent_disp = np.random.normal(0, kmc_params.search_radius, 3)

        for idx in cluster_indices:
            noise = np.random.normal(0, kmc_params.search_radius * 0.2, 3) # 20% noise
            search_atoms.positions[idx] += coherent_disp + noise

    else:
        # Single atom move
        displacement = np.random.normal(0, kmc_params.search_radius, 3)
        search_atoms.positions[target_idx] += displacement

    # 3. Dimer Search
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

        # Check Gamma
        try:
            # MinModeAtoms.calc might not expose results directly if wrapped.
            # Use search_atoms which shares the calculator reference/logic.
            # Actually MinModeAtoms uses force calls on the calculator.
            # We need to inspect the underlying calculator's last results.
            # search_atoms.calc should have it.
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
            # Calculate Barrier
            # Need initial energy (from atoms, not search_atoms which was perturbed)
            # We assume initial_atoms was at minimum.
            # We need to calculate E_saddle and E_initial.
            # E_saddle is from dimer_atoms (MinModeAtoms).
            e_saddle = dimer_atoms.get_potential_energy()

            # E_initial: we calculate it fresh to be sure?
            # Or assume the caller passed a relaxed structure.
            # But we don't have it here. We should calc it or pass it.
            # Efficient: Calc it on 'atoms' (which is copy of initial).
            e_initial = atoms.get_potential_energy()

            barrier = e_saddle - e_initial

            if barrier > 0.01:
                # Return success tuple
                return (barrier, product_atoms)

    # Not converged or no barrier
    return KMCResult(status=KMCStatus.NO_EVENT, structure=initial_atoms)


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

        Strategy:
        1. Minimize initial structure (basin bottom).
        2. Perform N saddle point searches in PARALLEL.
        3. For each search: ... (delegated to worker)
        4. Select event using BKL / KMC rate equation.
        5. Return new structure and time increment.
        """

        # Working copy
        atoms = initial_atoms.copy()
        self._setup_calculator(atoms, potential_path)

        # 1. Minimize current basin (ensure we are at a minimum)
        try:
            opt = FIRE(atoms, logfile=None)
            opt.run(fmax=self.kmc_params.dimer_fmax, steps=200)
        except Exception as e:
            logger.warning(f"Initial minimization failed: {e}")

        # Prepare for parallel execution
        # Strip calculator from atoms before pickling
        atoms.calc = None

        found_events: List[Tuple[float, Atoms]] = []
        uncertain_results: List[KMCResult] = []

        # 2. Saddle Point Searches (Parallel)
        logger.info(f"Starting {self.kmc_params.n_searches} parallel KMC searches with {self.kmc_params.n_workers} workers.")

        with ProcessPoolExecutor(max_workers=self.kmc_params.n_workers) as executor:
            futures = []
            for i in range(self.kmc_params.n_searches):
                # Pass a unique seed to each worker
                seed = np.random.randint(0, 1000000)
                futures.append(
                    executor.submit(
                        _run_single_search,
                        atoms, # Clean atoms
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
                        # (barrier, product_atoms)
                        found_events.append(result)
                except Exception as e:
                    logger.error(f"KMC Search worker failed: {e}")

        # Prioritize Uncertainty
        if uncertain_results:
            # Return the first uncertain result (or the one with max gamma?)
            # Let's return the first one encountered.
            logger.info(f"Collected {len(uncertain_results)} uncertain searches. Triggering AL.")
            return uncertain_results[0]

        # 4. Event Selection (KMC)
        if not found_events:
            # Need to re-attach calc to return properly structured atoms?
            # initial_atoms has no calc (we stripped it from 'atoms' which was copy).
            # But initial_atoms passed in arg is untouched.
            return KMCResult(status=KMCStatus.NO_EVENT, structure=initial_atoms)

        # Rate R_i = v * exp(-E_a / k_B T)
        k_B = 8.617333262e-5
        T = self.kmc_params.temperature
        v = self.kmc_params.prefactor

        rates = []
        for barrier, _ in found_events:
            r = v * np.exp(-barrier / (k_B * T))
            rates.append(r)

        total_rate = sum(rates)
        if total_rate == 0:
             return KMCResult(status=KMCStatus.NO_EVENT, structure=initial_atoms)

        # Select event
        dt = -np.log(np.random.random()) / total_rate

        selected_idx = np.random.choice(len(rates), p=np.array(rates)/total_rate)
        selected_event = found_events[selected_idx]

        final_structure = selected_event[1]

        # Strip calc from final structure just in case (optional, but cleaner)
        final_structure.calc = None

        logger.info(f"KMC Event Selected: Barrier={selected_event[0]:.3f} eV, Time Step={dt:.3e} s")

        return KMCResult(
            status=KMCStatus.SUCCESS,
            structure=final_structure,
            time_step=dt,
            metadata={"barrier": selected_event[0], "rates": rates}
        )
