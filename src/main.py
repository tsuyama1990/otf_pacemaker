"""Main entry point for the ACE Active Carver application.

This module sets up the configuration and dependencies using a Factory pattern,
then orchestrates the active learning loop.
"""

import logging
import sys
import argparse
import shutil
import random
import numpy as np
from pathlib import Path
from dataclasses import dataclass

from src.core.config import Config
from src.core.factory import ComponentFactory
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


@dataclass
class AppContext:
    """Holds the fully initialized application components."""
    orchestrator: ActiveLearningOrchestrator


class AppBuilder:
    """Builder for constructing the Application Context."""

    def __init__(self, config: Config):
        self.config = config

    def build(self) -> AppContext:
        """Builds and wires all dependencies."""

        # Initialize Component Factory
        factory = ComponentFactory(self.config)

        try:
            md_engine = factory.create_md_engine()
            kmc_engine = factory.create_kmc_engine()
            sampler = factory.create_sampler()
            generator = factory.create_generator()
            labeler = factory.create_labeler()
            trainer = factory.create_trainer()
        except Exception as e:
            logger.exception(f"Failed to initialize components: {e}")
            sys.exit(1)

        csv_logger = CSVLogger()

        orchestrator = ActiveLearningOrchestrator(
            config=self.config,
            md_engine=md_engine,
            kmc_engine=kmc_engine,
            sampler=sampler,
            generator=generator,
            labeler=labeler,
            trainer=trainer,
            csv_logger=csv_logger
        )

        return AppContext(orchestrator=orchestrator)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="ACE Active Carver")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to the main configuration file (default: config.yaml)"
    )
    parser.add_argument(
        "--meta",
        type=Path,
        default=Path("meta_config.yaml"),
        help="Path to the meta configuration file (default: meta_config.yaml)"
    )
    return parser.parse_args()


def main():
    """Main execution entry point."""
    args = parse_arguments()
    config_path = args.config
    meta_path = args.meta

    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    if not meta_path.exists():
        logger.error(f"Meta config file not found: {meta_path}")
        sys.exit(1)

    # 1. Load Configuration
    try:
        # Load Meta first (L0)
        meta_config = Config.load_meta(meta_path)
        # Load Experiment/System (L1) and inject Meta
        config = Config.load_experiment(config_path, meta_config)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)

    # 2. Experiment Setup (Directory & Backup)
    output_dir = config.experiment.output_dir
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        # Backup configuration files for reproducibility
        shutil.copy(config_path, output_dir / "config.yaml.backup")
        shutil.copy(meta_path, output_dir / "meta_config.yaml.backup")
        logger.info(f"Initialized experiment '{config.experiment.name}' at {output_dir}")
        logger.info("Configuration files backed up.")
    except Exception as e:
        logger.error(f"Failed to setup experiment directory {output_dir}: {e}")
        sys.exit(1)

    # 3. Reproducibility: Set Global Random Seed
    seed = getattr(config, "seed", 42)
    logger.info(f"Setting global random seed to {seed}")
    random.seed(seed)
    np.random.seed(seed)

    # 4. Check Phase 1 Requirement (Seed Generation)
    initial_pot_path = Path(config.al_params.initial_potential)
    if not initial_pot_path.exists():
        logger.info(f"Initial potential not found at {initial_pot_path}. Starting Phase 1: Seed Generation.")
        try:
            seed_gen = SeedGenerator(config)
            seed_gen.run()

            generated_pot = Path("data/seed/seed_potential.yace")
            if not generated_pot.exists():
                logger.error("Seed generation finished but potential file is missing.")
                sys.exit(1)

            if generated_pot.resolve() != initial_pot_path.resolve():
                logger.info(f"Copying generated potential to configured path: {initial_pot_path}")
                shutil.copy(generated_pot, initial_pot_path)

        except Exception as e:
            logger.exception(f"Phase 1 failed: {e}")
            sys.exit(1)
    else:
        logger.info("Initial potential found. Skipping Phase 1.")

    # 5. Build and Run Phase 2 Application
    logger.info("Initializing Phase 2 Application...")
    app_context = AppBuilder(config).build()
    
    logger.info("Starting Phase 2: Active Learning Loop.")
    app_context.orchestrator.run()


if __name__ == "__main__":
    main()
