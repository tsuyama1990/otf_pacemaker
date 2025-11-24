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
from src.sampling.strategies.composite import CompositeSampler
from src.scenario_generation.strategies.small_cell import SmallCellGenerator
from src.labeling.strategies.delta_labeler import DeltaLabeler
from src.labeling.calculators.shifted_lj import ShiftedLennardJones
from src.engines.lammps.runner import LAMMPSRunner
from src.engines.lammps.input_generator import LAMMPSInputGenerator
from src.engines.kmc import OffLatticeKMCEngine
from src.engines.dft.configurator import DFTConfigurator
from src.training.strategies.pacemaker import PacemakerTrainer
from src.utils.sssp_loader import (
    load_sssp_database,
    calculate_cutoffs,
    get_pseudopotentials_dict,
    validate_pseudopotentials
)
from src.utils.atomic_energies import AtomicEnergyManager

logger = logging.getLogger(__name__)

class ComponentFactory:
    """Factory class for creating system components."""

    def __init__(self, config: Config):
        self.config = config

    def create_md_engine(self) -> MDEngine:
        """Creates the Molecular Dynamics Engine."""
        # Use shift_energy from lj_params
        lj_params = {
            "epsilon": self.config.lj_params.epsilon,
            "sigma": self.config.lj_params.sigma,
            "cutoff": self.config.lj_params.cutoff,
            "shift_energy": self.config.lj_params.shift_energy
        }

        input_generator = LAMMPSInputGenerator(
            lj_params=lj_params,
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

    def _create_dft_calculator(self, kpts=None):
        """Helper to create DFT calculator with consistent settings using Configurator."""
        elements = self.config.md_params.elements

        # We need a dummy atoms object for heuristics if we are creating a generic calculator.
        # However, the configurator's build() method asks for atoms.
        # The Labeler needs a calculator *instance* passed to it.
        # But if the calculator settings depend on the atoms (e.g. Heuristics),
        # we ideally need to re-configure per atoms in the Labeler.
        # BUT, `Espresso` calculator in ASE is stateful.
        # If we just return a pre-configured calculator here, it might not have the correct magnetism
        # for a specific structure if the heuristics depend on that structure's composition (which is constant in a run usually).
        # Assuming composition is constant (defined in md_params.elements), Heuristics based on element types are safe.
        # Heuristics based on structure (e.g. geometry) are not used yet (only composition).

        # We create a dummy atoms object with all elements to let heuristics run once.
        from ase import Atoms
        dummy_atoms = Atoms(symbols=elements)

        configurator = DFTConfigurator(self.config.dft_params)
        # Returns (calculator, magnetism_settings)
        return configurator.build(dummy_atoms, elements, kpts)

    def _get_e0_dict(self):
        """Retrieve E0 using AtomicEnergyManager."""
        storage_path = Path("data/seed/potential.e0.yaml") # Convention or Config?
        if self.config.al_params.initial_potential:
             # If using existing potential, E0 might be near it?
             # But usually we define a standard place.
             # Or we look for .e0.yaml near potential.yaml_path
             pot_path = Path(self.config.al_params.potential_yaml_path)
             if pot_path.parent.exists():
                 storage_path = pot_path.parent / "potential.e0.yaml"

        manager = AtomicEnergyManager(storage_path)

        # Factory for isolated atom calculation
        def dft_factory():
             # Gamma point only for isolated atom
             return self._create_dft_calculator(kpts=(1,1,1))

        return manager.get_e0(self.config.md_params.elements, dft_factory)

    def create_kmc_engine(self) -> KMCEngine:
        """Creates the KMC Engine."""
        # We need E0 for KMC SumCalculator
        e0_dict = self._get_e0_dict()

        return OffLatticeKMCEngine(
            kmc_params=self.config.kmc_params,
            al_params=self.config.al_params,
            lj_params=self.config.lj_params,
            e0_dict=e0_dict
        )

    def create_sampler(self) -> Sampler:
        """Creates the Active Learning Sampler."""
        if self.config.al_params.sampling_strategy == "composite":
            return CompositeSampler()
        elif self.config.al_params.sampling_strategy == "max_gamma":
            return MaxGammaSampler()
        else:
             # Default fallback
             return CompositeSampler()

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
        # 1. Setup DFT (Espresso) - Now returns tuple
        qe_calculator, magnetism_settings = self._create_dft_calculator()

        # 2. Setup Baseline (LJ)
        lj_kwargs = {
            'epsilon': self.config.lj_params.epsilon,
            'sigma': self.config.lj_params.sigma,
            'rc': self.config.lj_params.cutoff,
            'shift_energy': self.config.lj_params.shift_energy
        }
        lj_calculator = ShiftedLennardJones(**lj_kwargs)

        # 3. Get E0
        e0_dict = self._get_e0_dict()

        return DeltaLabeler(
            reference_calculator=qe_calculator,
            baseline_calculator=lj_calculator,
            e0_dict=e0_dict,
            outlier_energy_max=self.config.al_params.outlier_energy_max,
            magnetism_settings=magnetism_settings
        )

    def create_trainer(self) -> Trainer:
        """Creates the Trainer."""
        return PacemakerTrainer(training_params=self.config.training_params)
