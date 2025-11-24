"""Component Factory for ACE Active Carver.

This module encapsulates the creation and wiring of system components,
adhering to the Dependency Injection pattern.
"""

import logging
from pathlib import Path
from ase.calculators.espresso import Espresso, EspressoProfile
from ase.calculators.lj import LennardJones

from src.core.config import Config
from src.core.interfaces import MDEngine, KMCEngine, Sampler, StructureGenerator, Labeler, Trainer
from src.sampling.strategies.max_gamma import MaxGammaSampler
from src.scenario_generation.strategies.small_cell import SmallCellGenerator
from src.labeling.strategies.delta_labeler import DeltaLabeler
from src.labeling.calculators.shifted_lj import ShiftedLennardJones
from src.engines.lammps.runner import LAMMPSRunner
from src.engines.lammps.input_generator import LAMMPSInputGenerator
from src.engines.kmc import OffLatticeKMCEngine
from src.training.strategies.pacemaker import PacemakerTrainer
from src.utils.sssp_loader import (
    load_sssp_database,
    calculate_cutoffs,
    get_pseudopotentials_dict,
    validate_pseudopotentials
)

logger = logging.getLogger(__name__)

class ComponentFactory:
    """Factory class for creating system components."""

    def __init__(self, config: Config):
        self.config = config

    def create_md_engine(self) -> MDEngine:
        """Creates the Molecular Dynamics Engine."""
        input_generator = LAMMPSInputGenerator(
            lj_params={
                "epsilon": self.config.lj_params.epsilon,
                "sigma": self.config.lj_params.sigma,
                "cutoff": self.config.lj_params.cutoff
            },
            md_params={
                "elements": self.config.md_params.elements,
                "timestep": self.config.md_params.timestep,
                "temperature": self.config.md_params.temperature,
                "pressure": self.config.md_params.pressure,
                "restart_freq": self.config.md_params.restart_freq,
                "dump_freq": self.config.md_params.dump_freq,
                "masses": self.config.md_params.masses
            }
        )

        return LAMMPSRunner(
            cmd=self.config.md_params.lammps_command,
            input_generator=input_generator
        )

    def create_kmc_engine(self) -> KMCEngine:
        """Creates the KMC Engine."""
        return OffLatticeKMCEngine(
            kmc_params=self.config.kmc_params,
            al_params=self.config.al_params
        )

    def create_sampler(self) -> Sampler:
        """Creates the Active Learning Sampler."""
        # Currently hardcoded to MaxGammaSampler as per previous main.py
        # Could be configurable via config.al_params.query_strategy
        return MaxGammaSampler()

    def create_generator(self) -> StructureGenerator:
        """Creates the Structure Generator for AL."""
        stoich_ratio = {el: 1.0 for el in self.config.md_params.elements}

        return SmallCellGenerator(
            r_core=self.config.al_params.r_core,
            box_size=self.config.al_params.box_size,
            stoichiometric_ratio=stoich_ratio,
            lammps_cmd=self.config.md_params.lammps_command,
            min_bond_distance=self.config.al_params.min_bond_distance,
            stoichiometry_tolerance=self.config.al_params.stoichiometry_tolerance,
            # Pass LJ params for potential internal pre-optimization/checks
            lj_params={
                "epsilon": self.config.lj_params.epsilon,
                "sigma": self.config.lj_params.sigma,
                "cutoff": self.config.lj_params.cutoff
            },
            elements=self.config.md_params.elements
        )

    def create_labeler(self) -> Labeler:
        """Creates the Labeler (Delta-Learning: DFT - LJ)."""
        # 1. Setup DFT (Espresso)
        logger.info(f"Loading SSSP database from {self.config.dft_params.sssp_json_path}")
        sssp_db = load_sssp_database(self.config.dft_params.sssp_json_path)

        elements = self.config.md_params.elements
        pseudo_dir_abs = str(Path(self.config.dft_params.pseudo_dir).resolve())

        validate_pseudopotentials(pseudo_dir_abs, elements, sssp_db)

        pseudopotentials = get_pseudopotentials_dict(elements, sssp_db)
        ecutwfc, ecutrho = calculate_cutoffs(elements, sssp_db)
        logger.info(f"Using SSSP cutoffs: ecutwfc={ecutwfc} Ry, ecutrho={ecutrho} Ry")

        default_kpts = (3, 3, 3) # Todo: Dynamic

        qe_input_data = {
            "control": {
                "pseudo_dir": pseudo_dir_abs,
                "calculation": "scf",
                "disk_io": "none",
            },
            "system": {
                "ecutwfc": ecutwfc,
                "ecutrho": ecutrho,
            },
            "electrons": {}
        }

        profile = EspressoProfile(
            command=self.config.dft_params.command,
            pseudo_dir=pseudo_dir_abs
        )

        qe_calculator = Espresso(
            profile=profile,
            pseudopotentials=pseudopotentials,
            input_data=qe_input_data,
            kpts=default_kpts,
            koffset=(1, 1, 1),
            pseudo_dir=pseudo_dir_abs
        )

        # 2. Setup Baseline (LJ)
        lj_kwargs = {
            'epsilon': self.config.lj_params.epsilon,
            'sigma': self.config.lj_params.sigma,
            'rc': self.config.lj_params.cutoff
        }
        lj_calculator = ShiftedLennardJones(**lj_kwargs)

        return DeltaLabeler(
            reference_calculator=qe_calculator,
            baseline_calculator=lj_calculator
        )

    def create_trainer(self) -> Trainer:
        """Creates the Trainer."""
        return PacemakerTrainer()
