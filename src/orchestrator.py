"""Orchestrator for the Active Learning Loop.

This module contains the high-level logic for the active learning workflow,
coordinating the interaction between MD engine, sampler, generator, labeler, and trainer.
"""

import os
import logging
from pathlib import Path
from typing import Optional, List
from ase.io import read
from ase import Atoms

from src.interfaces import MDEngine, Sampler, StructureGenerator, Labeler, Trainer
from src.enums import SimulationState
from src.config import Config

logger = logging.getLogger(__name__)


class ActiveLearningOrchestrator:
    """Manages the active learning loop."""

    def __init__(
        self,
        config: Config,
        md_engine: MDEngine,
        sampler: Sampler,
        generator: StructureGenerator,
        labeler: Labeler,
        trainer: Trainer,
    ):
        """Initialize the orchestrator.

        Args:
            config: Configuration object.
            md_engine: MD Engine instance.
            sampler: Sampler instance.
            generator: Structure Generator instance.
            labeler: Labeler instance.
            trainer: Trainer instance.
        """
        self.config = config
        self.md_engine = md_engine
        self.sampler = sampler
        self.generator = generator
        self.labeler = labeler
        self.trainer = trainer

    def run(self):
        """Executes the active learning loop."""
        current_potential = self.config.al_params.initial_potential
        current_structure = self.config.md_params.initial_structure
        is_restart = False
        iteration = 0
        original_cwd = Path.cwd()

        while True:
            iteration += 1
            work_dir = Path(f"data/iteration_{iteration}")
            work_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"--- Starting Iteration {iteration} ---")

            try:
                os.chdir(work_dir)

                # Resolve paths relative to original CWD
                abs_potential_path = self._resolve_path(current_potential, original_cwd)

                if not abs_potential_path.exists():
                    logger.error(f"Potential file not found: {abs_potential_path}")
                    break

                # Prepare Structure Path
                input_structure_arg = self._prepare_structure_path(
                    is_restart, iteration, current_structure, original_cwd
                )
                if not input_structure_arg:
                    break

                # 1. Run MD
                logger.info("Running MD...")
                state = self.md_engine.run(
                    potential_path=str(abs_potential_path),
                    steps=self.config.md_params.n_steps,
                    gamma_threshold=self.config.al_params.gamma_threshold,
                    input_structure=input_structure_arg,
                    is_restart=is_restart
                )

                logger.info(f"MD Finished with state: {state}")

                if state == SimulationState.COMPLETED:
                    logger.info("Simulation completed successfully.")
                    break
                elif state == SimulationState.FAILED:
                    logger.error("Simulation failed.")
                    break
                elif state == SimulationState.UNCERTAIN:
                    # 2. Handle Uncertainty
                    logger.info("Uncertainty detected. Starting Active Learning cycle.")

                    dump_file = Path("dump.lammpstrj")
                    if not dump_file.exists():
                        raise FileNotFoundError("Dump file not found.")

                    # Read last frame
                    atoms = read(dump_file, index=-1, format="lammps-dump-text")

                    # Fix symbols if needed
                    self._ensure_chemical_symbols(atoms)

                    # Sample
                    center_ids = self.sampler.sample(atoms, self.config.al_params.n_clusters)

                    labeled_clusters = []
                    logger.info(f"Generating and labeling {len(center_ids)} small cells...")

                    for cid in center_ids:
                        try:
                            # Generate Small Cell
                            cell = self.generator.generate_cell(atoms, cid, str(abs_potential_path))

                            # Label
                            labeled_cluster = self.labeler.label(cell)
                            if labeled_cluster is None:
                                logger.warning(f"Labeling failed for cluster {cid}. Skipping.")
                                continue

                            labeled_clusters.append(labeled_cluster)
                        except Exception as e:
                            logger.warning(f"Processing failed for cluster {cid}: {e}. Skipping.")
                            continue

                    if not labeled_clusters:
                        logger.error("No clusters labeled successfully. Aborting active learning loop.")
                        break

                    # Train
                    logger.info("Training new potential...")
                    dataset_path = self.trainer.prepare_dataset(labeled_clusters)
                    new_potential = self.trainer.train(dataset_path, str(abs_potential_path))

                    current_potential = str(Path(new_potential).resolve())
                    is_restart = True
                    logger.info(f"New potential trained: {current_potential}")

            except Exception as e:
                logger.exception(f"An error occurred in iteration {iteration}: {e}")
                break
            finally:
                os.chdir(original_cwd)

    def _resolve_path(self, path_str: str, base_cwd: Path) -> Path:
        """Resolve a path to absolute, handling relative paths correctly."""
        p = Path(path_str)
        if p.is_absolute():
            return p
        return (base_cwd / p).resolve()

    def _prepare_structure_path(
        self, is_restart: bool, iteration: int, initial_structure: str, base_cwd: Path
    ) -> Optional[str]:
        """Determine the correct input structure path."""
        if is_restart:
            prev_dir = base_cwd / f"data/iteration_{iteration-1}"
            restart_file = prev_dir / "restart.chk"
            if not restart_file.exists():
                logger.error(f"Restart file missing for resume: {restart_file}")
                return None
            return str(restart_file)
        else:
            return str(self._resolve_path(initial_structure, base_cwd))

    def _ensure_chemical_symbols(self, atoms: Atoms):
        """Maps numeric types to chemical symbols if missing."""
        if 'type' in atoms.arrays:
             types = atoms.get_array('type')
             elements = self.config.md_params.elements
             symbols = []
             for t in types:
                 if 1 <= t <= len(elements):
                     symbols.append(elements[t-1])
                 else:
                     symbols.append("X")
             atoms.set_chemical_symbols(symbols)
