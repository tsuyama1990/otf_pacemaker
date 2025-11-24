"""Main entry point for the ACE Active Carver application.

This module sets up the configuration and dependencies using a Factory pattern,
then orchestrates the active learning loop.
"""

import logging
import sys
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


def main():
    """Main execution entry point."""
    # 1. Load Configuration
    config_path = Path("config.yaml")
    if not config_path.exists():
        logger.error("config.yaml not found.")
        sys.exit(1)

    try:
        config = Config.from_yaml(config_path)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)

    # 2. Reproducibility: Set Global Random Seed
    # Use default 42 if not specified, though config should probably have it.
    seed = getattr(config, "seed", 42)
    logger.info(f"Setting global random seed to {seed}")
    random.seed(seed)
    np.random.seed(seed)

    # 3. Check Phase 1 Requirement (Seed Generation)
    # This logic remains slightly procedural as it decides WHICH app to run.
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
                import shutil
                shutil.copy(generated_pot, initial_pot_path)

        except Exception as e:
            logger.exception(f"Phase 1 failed: {e}")
            sys.exit(1)
    else:
        logger.info("Initial potential found. Skipping Phase 1.")

    # 4. Build and Run Phase 2 Application
    logger.info("Initializing Phase 2 Application...")
    app_context = AppBuilder(config).build()
    
    logger.info("Starting Phase 2: Active Learning Loop.")
    app_context.orchestrator.run()


if __name__ == "__main__":
    main()
