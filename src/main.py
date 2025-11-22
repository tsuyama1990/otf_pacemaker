"""Main entry point for the ACE Active Carver application.

This module sets up the configuration and dependencies, then orchestrates the active learning loop.
"""

import logging
import sys
from pathlib import Path
from ase.calculators.espresso import Espresso

from src.active_learning import MaxGammaSampler, SmallCellGenerator
from src.config import Config
from src.labeler import DeltaLabeler, ShiftedLennardJones
from src.md_engine import LAMMPSRunner, LAMMPSInputGenerator
from src.trainer import PacemakerTrainer
from src.orchestrator import ActiveLearningOrchestrator

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

    # 2. Initialize Components

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
    qe_input_data = {
        "control": {
            "pseudo_dir": config.dft_params.pseudo_dir,
            "calculation": "scf",
        },
        "system": {
            "ecutwfc": config.dft_params.ecutwfc,
        },
        "electrons": {
            "k_points": config.dft_params.kpts
        }
    }

    qe_calculator = Espresso(
        command=config.dft_params.command,
        input_data=qe_input_data,
        kpts=config.dft_params.kpts,
        pseudo_dir=config.dft_params.pseudo_dir
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

    # 3. Initialize Orchestrator
    orchestrator = ActiveLearningOrchestrator(
        config=config,
        md_engine=runner,
        sampler=sampler,
        generator=generator,
        labeler=labeler,
        trainer=trainer
    )

    # 4. Run
    orchestrator.run()


if __name__ == "__main__":
    main()
