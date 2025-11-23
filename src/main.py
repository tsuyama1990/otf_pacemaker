"""Main entry point for the ACE Active Carver application.

This module sets up the configuration and dependencies, then orchestrates the active learning loop.
"""

import logging
import sys
from pathlib import Path
from ase.calculators.espresso import Espresso, EspressoProfile

from src.sampling.strategies.max_gamma import MaxGammaSampler
from src.scenario_generation.strategies.small_cell import SmallCellGenerator
from src.core.config import Config
from src.labeling.strategies.delta_labeler import DeltaLabeler
from src.labeling.calculators.shifted_lj import ShiftedLennardJones
from src.engines.lammps.runner import LAMMPSRunner
from src.engines.lammps.input_generator import LAMMPSInputGenerator
from src.engines.kmc import OffLatticeKMCEngine
from src.training.strategies.pacemaker import PacemakerTrainer
from src.workflows.orchestrator import ActiveLearningOrchestrator
from src.workflows.seed_generation import SeedGenerator
from src.utils.logger import CSVLogger

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main():
    """Main execution entry point."""
    # 1. Load Configuration
    config_path = Path("config.yaml")
    if not config_path.exists():
        logger.error("config.yaml not found.")
        sys.exit(1)

    config = Config.from_yaml(config_path)

    # 2. Check Phase 1 Requirement
    initial_pot_path = Path(config.al_params.initial_potential)
    if not initial_pot_path.exists():
        logger.info(f"Initial potential not found at {initial_pot_path}. Starting Phase 1: Seed Generation.")
        try:
            seed_gen = SeedGenerator(config)
            seed_gen.run()

            # After Seed Gen, we assume the potential is created at config.al_params.initial_potential
            # Note: SeedGenerator writes to 'data/seed/seed_potential.yace'.
            # We need to make sure config points to that or we move it?
            # Or we just assume the user configured initial_potential to point to where seed gen puts it.

            # If the configured path is different from where SeedGenerator put it, we might have an issue.
            # SeedGenerator hardcodes output to 'data/seed/seed_potential.yace'.
            generated_pot = Path("data/seed/seed_potential.yace")

            if not generated_pot.exists():
                logger.error("Seed generation finished but potential file is missing.")
                sys.exit(1)

            if generated_pot.resolve() != initial_pot_path.resolve():
                logger.info(f"Copying generated potential to configured path: {initial_pot_path}")
                import shutil
                shutil.copy(generated_pot, initial_pot_path)

        except Exception as e:
            logger.exception(f"Phase 1 failed: {e}")
            sys.exit(1)
    else:
        logger.info("Initial potential found. Skipping Phase 1.")

    # 3. Initialize Components for Phase 2

    # MD Engine
    input_generator = LAMMPSInputGenerator(
        lj_params={
            "epsilon": config.lj_params.epsilon,
            "sigma": config.lj_params.sigma,
            "cutoff": config.lj_params.cutoff
        },
        md_params={
            "elements": config.md_params.elements,
            "timestep": config.md_params.timestep,
            "temperature": config.md_params.temperature,
            "pressure": config.md_params.pressure,
            "restart_freq": config.md_params.restart_freq,
            "dump_freq": config.md_params.dump_freq,
            "masses": config.md_params.masses
        }
    )

    runner = LAMMPSRunner(
        cmd="lmp_serial",  # In production, this might come from env or config
        input_generator=input_generator
    )

    # KMC Engine
    kmc_engine = OffLatticeKMCEngine(
        kmc_params=config.kmc_params,
        al_params=config.al_params
    )

    # Sampler
    sampler = MaxGammaSampler()

    # Generator
    stoich_ratio = {el: 1.0 for el in config.md_params.elements}

    generator = SmallCellGenerator(
        r_core=config.al_params.r_core,
        box_size=config.al_params.box_size,
        stoichiometric_ratio=stoich_ratio,
        lammps_cmd="lmp_serial",
        min_bond_distance=config.al_params.min_bond_distance,
        stoichiometry_tolerance=config.al_params.stoichiometry_tolerance
    )

    # Labeler
    # Setup QE Calculator (Reference)
    # Ensure pseudo_dir is absolute
    pseudo_dir_abs = str(Path(config.dft_params.pseudo_dir).resolve())

    qe_input_data = {
        "control": {
            "pseudo_dir": pseudo_dir_abs,
            "calculation": "scf",
            "disk_io": "none", # Optimization for speed/parallelism
        },
        "system": {
            "ecutwfc": config.dft_params.ecutwfc,
        },
        "electrons": {
            "k_points": config.dft_params.kpts
        }
    }

    # Correctly instantiate EspressoProfile for ASE 3.26.0
    profile = EspressoProfile(
        command=config.dft_params.command,
        pseudo_dir=pseudo_dir_abs
    )

    qe_calculator = Espresso(
        profile=profile,
        input_data=qe_input_data,
        kpts=config.dft_params.kpts,
        pseudo_dir=pseudo_dir_abs # Kept for compatibility if needed
    )

    # Setup LJ Calculator (Baseline)
    lj_kwargs = {
        'epsilon': config.lj_params.epsilon,
        'sigma': config.lj_params.sigma,
        'rc': config.lj_params.cutoff
    }
    lj_calculator = ShiftedLennardJones(**lj_kwargs)

    labeler = DeltaLabeler(
        reference_calculator=qe_calculator,
        baseline_calculator=lj_calculator
    )

    # Trainer
    trainer = PacemakerTrainer()

    # Logger
    csv_logger = CSVLogger()

    # 4. Initialize Orchestrator
    orchestrator = ActiveLearningOrchestrator(
        config=config,
        md_engine=runner,
        kmc_engine=kmc_engine,
        sampler=sampler,
        generator=generator,
        labeler=labeler,
        trainer=trainer,
        csv_logger=csv_logger
    )

    # 5. Run Phase 2
    logger.info("Starting Phase 2: Active Learning Loop.")
    orchestrator.run()


if __name__ == "__main__":
    main()
