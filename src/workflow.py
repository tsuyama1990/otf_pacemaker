"""Workflow orchestration for ACE Active Carver.

This module defines the ActiveLearningWorkflow class which orchestrates
the interaction between the MD engine, sampler, structure generator, labeler, and trainer.
"""

import logging
import os
import shutil
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
from ase.io import read

from src.config import Config
from src.interfaces import MDEngine, Sampler, StructureGenerator, Labeler, Trainer
from src.enums import SimulationState

logger = logging.getLogger(__name__)


class ActiveLearningWorkflow:
    """Orchestrates the active learning loop."""

    def __init__(
        self,
        config: Config,
        md_engine: MDEngine,
        sampler: Sampler,
        generator: StructureGenerator,
        labeler: Labeler,
        trainer: Trainer,
    ):
        """Initialize the workflow.

        Args:
            config: Configuration object.
            md_engine: Component for running MD.
            sampler: Component for selecting uncertain atoms.
            generator: Component for generating small cells.
            labeler: Component for calculating targets.
            trainer: Component for training potentials.
        """
        self.config = config
        self.md_engine = md_engine
        self.sampler = sampler
        self.generator = generator
        self.labeler = labeler
        self.trainer = trainer

    def run(self):
        """Execute the active learning loop."""
        current_potential = self.config.al_params.initial_potential
        current_structure = self.config.md_params.initial_structure
        is_restart = False
        iteration = 0

        while True:
            iteration += 1
            work_dir = Path(f"data/iteration_{iteration}")
            work_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"--- Starting Iteration {iteration} ---")

            original_cwd = Path.cwd()
            try:
                os.chdir(work_dir)

                # Resolve paths relative to original CWD if they are not absolute
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
                    new_potential = self._handle_uncertainty(abs_potential_path)
                    current_potential = str(Path(new_potential).resolve())
                    is_restart = True

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
                logger.error("Restart file missing for resume.")
                return None
            return str(restart_file)
        else:
            return str(self._resolve_path(initial_structure, base_cwd))

    def _handle_uncertainty(self, current_potential_path: Path) -> str:
        """Handle the uncertainty state: Sample, Label, Train.

        Returns:
             str: Path to the newly trained potential.
        """
        logger.info("Uncertainty detected. Starting Active Learning cycle.")

        dump_file = Path("dump.lammpstrj")
        if not dump_file.exists():
            raise FileNotFoundError("Dump file not found.")

        # Read last frame
        try:
            atoms = read(dump_file, index=-1, format="lammps-dump-text")
            # Basic species mapping if needed, assuming generic handling or correct dump
            # If atoms have type, map to elements
            if self.config.md_params.elements:
                types = atoms.get_array('type')
                elements = self.config.md_params.elements
                symbols = []
                for t in types:
                    if t <= len(elements):
                        symbols.append(elements[t-1])
                    else:
                        symbols.append("X")
                atoms.set_chemical_symbols(symbols)
        except Exception as e:
            raise RuntimeError(f"Failed to read dump file: {e}")

        # Sample
        center_ids = self.sampler.sample(atoms, self.config.al_params.n_clusters)

        labeled_clusters = []
        logger.info(f"Generating and labeling {len(center_ids)} small cells...")

        for cid in center_ids:
            try:
                cell = self.generator.generate(atoms, cid, str(current_potential_path))
                labeled_cluster = self.labeler.label(cell)
                labeled_clusters.append(labeled_cluster)
            except Exception as e:
                logger.warning(f"Processing failed for cluster {cid}: {e}. Skipping.")
                continue

        if not labeled_clusters:
            raise RuntimeError("No clusters labeled successfully.")

        # Train
        logger.info("Training new potential...")
        dataset_path = self.trainer.prepare_dataset(labeled_clusters)
        new_potential = self.trainer.train(dataset_path, str(current_potential_path))
        logger.info(f"New potential trained: {new_potential}")
        return new_potential
