"""KMC Engine Implementation.

This module implements the Off-Lattice KMC Engine using ASE Dimer method
and Pacemaker potential with k-step uncertainty checking.
"""

import logging
import numpy as np
from ase import Atoms
from ase.mep import MinModeAtoms, DimerControl
from ase.optimize import FIRE
from typing import Optional, List, Tuple, Dict, Any

try:
    from pyace import PyACECalculator
except ImportError:
    # Fallback for environments where pyace is not installed (e.g. testing)
    PyACECalculator = None

from src.core.interfaces import KMCEngine, KMCResult
from src.core.enums import KMCStatus
from src.core.config import KMCParams, ALParams

logger = logging.getLogger(__name__)

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
        if PyACECalculator is None:
            # Check if we are mocking/testing
            if hasattr(atoms, "calc") and atoms.calc is not None:
                return
            raise ImportError("pyace module is required for KMC Engine.")

        # We assume potential_path is a .yace or .yaml file
        calc = PyACECalculator(potential_path)
        atoms.calc = calc

    def run_step(self, initial_atoms: Atoms, potential_path: str) -> KMCResult:
        """Run a single KMC step: Saddle point search and event selection.

        Strategy:
        1. Minimize initial structure (basin bottom).
        2. Perform N saddle point searches (Dimer method).
        3. For each search:
           - Displace random atom.
           - Run Dimer optimization with k-step check.
           - If uncertain, abort and return UNCERTAIN.
           - If converged, relax to product state.
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

        found_events: List[Tuple[float, Atoms]] = [] # List of (barrier, product_structure)

        # 2. Saddle Point Searches
        for i in range(self.kmc_params.n_searches):
            logger.info(f"KMC Search {i+1}/{self.kmc_params.n_searches}")

            # Perturb structure to start search
            search_atoms = atoms.copy()
            self._setup_calculator(search_atoms, potential_path)

            # Random displacement to break symmetry and find different saddles
            indices = range(len(search_atoms))
            target_idx = np.random.choice(indices)
            displacement = np.random.normal(0, self.kmc_params.search_radius, 3)
            search_atoms.positions[target_idx] += displacement

            # Setup Dimer
            # DimerControl manages the dimer parameters
            dimer_control = DimerControl(
                logfile=None,
                eigenmode_method='displacement',
                f_rot_min=0.1, # Default params usually fine, explicit for clarity
                f_rot_max=1.0
            )
            # MinModeAtoms wraps the atoms object to behave like a dimer
            dimer_atoms = MinModeAtoms(search_atoms, dimer_control)

            # Displace the dimer to set initial orientation
            # We use a random vector for the dimer orientation
            dimer_atoms.displace()

            opt = FIRE(dimer_atoms, logfile=None)

            # --- Custom k-step Optimization Loop ---
            converged = False
            uncertain = False
            max_steps = 1000 # Safety limit
            current_step = 0

            # We need to access the underlying calculator results.
            # dimer_atoms.calc delegates to search_atoms.calc usually,
            # but MinModeAtoms might mask it or wrap it.
            # Actually MinModeAtoms uses the atoms.calc to evaluate forces/energies.
            # So we can check search_atoms.calc or dimer_atoms.calc.

            while current_step < max_steps:
                # Run k steps
                opt.run(steps=self.kmc_params.check_interval)
                current_step += self.kmc_params.check_interval

                # Check 1: Convergence
                if opt.converged():
                    converged = True
                    break

                # Check 2: Uncertainty
                try:
                    # Access gamma from the underlying atoms object
                    # search_atoms is the one holding the calculator
                    gamma_vals = search_atoms.calc.results.get('gamma')
                    if gamma_vals is None and hasattr(search_atoms.calc, 'get_property'):
                         # Try getting it explicitly if supported
                         try:
                             gamma_vals = search_atoms.calc.get_property('gamma', search_atoms)
                         except:
                             pass

                    if gamma_vals is not None:
                        max_gamma = np.max(gamma_vals)
                        if max_gamma > self.al_params.gamma_threshold:
                            logger.info(f"High uncertainty detected (gamma={max_gamma:.4f}) during saddle search.")
                            uncertain = True
                            break
                except Exception as e:
                    # If gamma check fails, we assume it's fine or log warning
                    # logger.warning(f"Failed to check gamma: {e}")
                    pass

            if uncertain:
                return KMCResult(
                    status=KMCStatus.UNCERTAIN,
                    structure=search_atoms, # Return the structure where gamma is high
                    metadata={"reason": "High Gamma in Saddle Search", "gamma": max_gamma}
                )

            if converged:
                # Relax to product
                product_atoms = search_atoms.copy()
                self._setup_calculator(product_atoms, potential_path)

                prod_opt = FIRE(product_atoms, logfile=None)
                prod_converged = False
                prod_uncertain = False
                p_step = 0

                while p_step < max_steps:
                    prod_opt.run(steps=self.kmc_params.check_interval)
                    p_step += self.kmc_params.check_interval

                    if prod_opt.converged():
                        prod_converged = True
                        break

                    # Check Gamma
                    try:
                        g_vals = product_atoms.calc.results.get('gamma')
                        if g_vals is not None and np.max(g_vals) > self.al_params.gamma_threshold:
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
                    e_saddle = dimer_atoms.get_potential_energy()
                    e_initial = atoms.get_potential_energy()
                    barrier = e_saddle - e_initial

                    if barrier > 0.01: # Filter numerical noise
                        found_events.append((barrier, product_atoms))

        # 4. Event Selection (KMC)
        if not found_events:
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

        logger.info(f"KMC Event Selected: Barrier={selected_event[0]:.3f} eV, Time Step={dt:.3e} s")

        return KMCResult(
            status=KMCStatus.SUCCESS,
            structure=final_structure,
            time_step=dt,
            metadata={"barrier": selected_event[0], "rates": rates}
        )
