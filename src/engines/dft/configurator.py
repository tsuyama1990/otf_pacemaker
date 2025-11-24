"""DFT Configurator.

This module builds the ASE Calculator object for DFT calculations, merging
static configuration from DFTParams with dynamic physical heuristics.
"""

import logging
from typing import Optional, List, Dict, Any
from ase import Atoms
from ase.calculators.espresso import Espresso, EspressoProfile
from pathlib import Path

from src.core.config import DFTParams
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

    def __init__(self, params: DFTParams):
        """Initialize with DFT parameters.

        Args:
            params: The static DFT configuration.
        """
        self.params = params

    def build(self, atoms: Atoms, elements: List[str], kpts: Optional[tuple] = None) -> Espresso:
        """Build a configured Espresso calculator.

        Args:
            atoms: The structure to calculate (used for heuristics).
            elements: List of elements involved (used for pseudopotentials).
            kpts: K-points tuple (e.g. (3,3,3)).

        Returns:
            Espresso: The configured calculator.
        """
        # 1. Load SSSP and Pseudopotentials (Static Config)
        sssp_db = load_sssp_database(self.params.sssp_json_path)
        pseudo_dir_abs = str(Path(self.params.pseudo_dir).resolve())

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

        # 4. Handle Starting Magnetization
        # In ASE Espresso, we can pass 'magmoms' to the Atoms object, or set 'starting_magnetization' in input_data.
        # However, input_data requires mapping species index (1..ntyp) to values.
        # ASE handles `initial_magnetic_moments` on the Atoms object automatically if we don't specify it in input_data?
        # Actually, ASE's Espresso calculator is smart. If we set atoms.set_initial_magnetic_moments(), it writes starting_magnetization.
        # BUT, since we are returning a calculator that might be attached to different atoms (or re-attached),
        # it's safer to configure the calculator to handle it or apply it to the atoms.
        #
        # Better approach: We return the calculator. The caller attaches it to atoms.
        # But 'starting_magnetization' depends on species.
        # If we use `input_data['system']['starting_magnetization']`, we need to know the species order.
        # ASE's `write_espresso_in` determines species order.
        #
        # A robust way is to put the recommended magmoms into the atoms object itself before calculation,
        # but this method builds the *calculator*.
        #
        # Solution: We will pass `magnetism_settings` (dict of {Element: moment}) back to the caller
        # OR we set it in `input_data` using ASE's convention if possible?
        # ASE Espresso calculator does NOT automatically map `starting_magnetization` from a dict unless we use specific keys.
        #
        # Wait, the prompt says: "magnetic_map を展開し、starting_magnetization(i) を設定".
        # Since we don't know the species index 'i' until ASE writes the file, this is tricky *inside* the calculator dict
        # unless we know the species list order ASE will use.
        # ASE sorts unique species alphabetically usually.
        #
        # Let's rely on `atoms.set_initial_magnetic_moments`? No, the Labeler receives `atoms`.
        # The Configurator builds the Calculator.
        # If we want the Calculator to carry this info, it's hard.
        #
        # Alternative: The prompt says "starting_magnetization(i) を設定".
        # The `elements` list passed to this method `build` is likely the full list of elements in the system.
        # If we sort `elements`, we can map them.
        #
        # sorted_elements = sorted(list(set(elements)))
        # for i, el in enumerate(sorted_elements):
        #     if el in magnetism_settings:
        #         input_data["system"][f"starting_magnetization({i+1})"] = magnetism_settings[el]

        sorted_elements = sorted(list(set(elements)))
        for i, el in enumerate(sorted_elements):
            if el in magnetism_settings:
                val = magnetism_settings[el]
                # Only set if non-zero to keep input clean, or set anyway.
                input_data["system"][f"starting_magnetization({i+1})"] = val

        # 5. Create Profile and Calculator
        profile = EspressoProfile(
            command=self.params.command,
            pseudo_dir=pseudo_dir_abs
        )

        if kpts is None:
             kpts = (3, 3, 3)

        return Espresso(
            profile=profile,
            pseudopotentials=pseudopotentials,
            input_data=input_data,
            kpts=kpts,
            koffset=(1, 1, 1),
            pseudo_dir=pseudo_dir_abs
        )
