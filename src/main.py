"""Main entry point for the ACE Active Carver application.

This module sets up the configuration and dependencies, then launches the workflow.
"""

import logging
import sys
from pathlib import Path

from ase.calculators.espresso import Espresso

from src.active_learning import MaxGammaSampler, SmallCellGenerator
from src.config import Config
from src.labeler import DeltaLabeler
from src.md_engine import LAMMPSRunner
from src.trainer import PacemakerTrainer
from src.workflow import ActiveLearningWorkflow

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
    runner = LAMMPSRunner(
        cmd="lmp_serial", # In production, this might come from env or config
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

    # Sampler
    sampler = MaxGammaSampler()

    # Generator
    generator = SmallCellGenerator(
        box_size=config.al_params.box_size,
        r_core=config.al_params.r_core,
        lammps_cmd="lmp_serial",
        stoichiometry_tolerance=config.al_params.stoichiometry_tolerance,
        elements=config.md_params.elements
    )

    # Labeler
    # Setup QE Calculator
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

    labeler = DeltaLabeler(
        qe_calculator=qe_calculator,
        lj_params={
            "epsilon": config.lj_params.epsilon,
            "sigma": config.lj_params.sigma,
            "cutoff": config.lj_params.cutoff
        }
    )

    # Trainer
    trainer = PacemakerTrainer()

    # 3. Create and Run Workflow
    workflow = ActiveLearningWorkflow(
        config=config,
        md_engine=runner,
        sampler=sampler,
        generator=generator,
        labeler=labeler,
        trainer=trainer
    )

    workflow.run()


if __name__ == "__main__":
    main()
