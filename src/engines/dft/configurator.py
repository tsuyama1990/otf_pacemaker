"""DFT Configurator.

This module builds the ASE Calculator object for DFT calculations, merging
static configuration from DFTParams with dynamic physical heuristics.
"""

import logging
from typing import Optional, List, Dict, Any
from ase import Atoms
from ase.calculators.espresso import Espresso, EspressoProfile
from pathlib import Path

from src.core.config import DFTParams, MetaConfig
from src.engines.dft.heuristics import PymatgenHeuristics
from src.utils.sssp_loader import (
    load_sssp_database,
    calculate_cutoffs,
    get_pseudopotentials_dict,
    validate_pseudopotentials
)

logger = logging.getLogger(__name__)

class DFTConfigurator:
    """Builder for DFT Calculators merging Config and Heuristics."""

    def __init__(self, params: DFTParams, meta: MetaConfig):
        """Initialize with DFT parameters.

        Args:
            params: The static DFT configuration.
            meta: The environment configuration (paths, commands).
        """
        self.params = params
        self.meta = meta

    def build(self, atoms: Atoms, elements: List[str], kpts: Optional[tuple] = None) -> tuple[Espresso, Dict[str, float]]:
        """Build a configured Espresso calculator.

        Args:
            atoms: The structure to calculate (used for heuristics).
            elements: List of elements involved (used for pseudopotentials).
            kpts: K-points tuple (e.g. (3,3,3)).

        Returns:
            Tuple[Espresso, Dict[str, float]]: The configured calculator and the magnetic moments map.
        """
        # 1. Load SSSP and Pseudopotentials (Environment Config)
        sssp_path = self.meta.sssp_json_path
        pseudo_dir_abs = str(self.meta.pseudo_dir.resolve())

        sssp_db = load_sssp_database(str(sssp_path))
        validate_pseudopotentials(pseudo_dir_abs, elements, sssp_db)

        pseudopotentials = get_pseudopotentials_dict(elements, sssp_db)
        ecutwfc, ecutrho = calculate_cutoffs(elements, sssp_db)

        # 2. Base Input Data
        input_data = {
            "control": {
                "pseudo_dir": pseudo_dir_abs,
                "calculation": "scf",
                "disk_io": "none",
                "tprnfor": True, # Ensure forces are printed
                "tstress": True, # Ensure stress is calculated
            },
            "system": {
                "ecutwfc": ecutwfc,
                "ecutrho": ecutrho,
            },
            "electrons": {
                "mixing_beta": 0.7, # Standard robust value
                "conv_thr": 1.0e-6
            }
        }

        # 3. Apply Heuristics (if enabled)
        magnetism_settings = {}
        if self.params.auto_physics:
            recommendations = PymatgenHeuristics.get_recommended_params(atoms)

            # Merge System Settings (Smearing)
            input_data["system"].update(recommendations.get("system", {}))

            # Handle Magnetism
            mag_rec = recommendations.get("magnetism", {})
            if mag_rec.get("nspin", 1) == 2:
                input_data["system"]["nspin"] = 2
                magnetism_settings = mag_rec.get("moments", {})
                logger.info("Auto-Physics: Enabled Spin Polarization (nspin=2)")

        # 4. Handle System Charge
        if 'charge' in atoms.info:
            charge = atoms.info['charge']
            input_data["system"]["tot_charge"] = charge
            logger.info(f"System charge {charge} propagated to QE tot_charge.")

        # 5. Handle Starting Magnetization
        # We purposely do NOT set 'starting_magnetization(i)' in input_data here.
        # ASE's Espresso calculator determines the species order for the input file dynamically.
        # Attempting to guess the index 'i' based on sorted(elements) here creates a race condition
        # (Time Bomb) where we might assign magnetic moments to the wrong species if ASE sorts differently.
        #
        # Instead, we return 'magnetism_settings' (element -> moment) and rely on the Labeler
        # to apply these directly to the Atoms object via atoms.set_initial_magnetic_moments().
        # ASE will then correctly generate the starting_magnetization tags matched to its internal ordering.

        # 5. Create Profile and Calculator
        profile = EspressoProfile(
            command=self.meta.dft_command,
            pseudo_dir=pseudo_dir_abs
        )

        if kpts is None:
             kpts = (3, 3, 3)

        calculator = Espresso(
            profile=profile,
            pseudopotentials=pseudopotentials,
            input_data=input_data,
            kpts=kpts,
            koffset=(1, 1, 1),
            pseudo_dir=pseudo_dir_abs
        )

        return calculator, magnetism_settings
