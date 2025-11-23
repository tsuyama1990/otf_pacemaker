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
        current_asi = self.config.al_params.initial_active_set_path

        is_restart = False
        iteration = 0
        original_cwd = Path.cwd()

        # Resolve paths that don't change
        potential_yaml_path = self._resolve_path(self.config.al_params.potential_yaml_path, original_cwd)
        initial_dataset_path = None
        if self.config.al_params.initial_dataset_path:
             initial_dataset_path = self._resolve_path(self.config.al_params.initial_dataset_path, original_cwd)

        # 0. Initialize Active Set if needed
        # If current_asi is None, we need to generate it from initial dataset
        if not current_asi and initial_dataset_path:
             logger.info("No initial Active Set provided. Generating from initial dataset...")
             try:
                 # We can use the Trainer to update the active set (generate initial one)
                 # We need to temporarily be in a writable dir or handle absolute paths carefully.
                 # Let's do it in 'data/seed' if it exists or just in root?
                 # Trainer.update_active_set outputs to 'potential.asi' in current dir.
                 # Let's generate it in 'data/seed' if we can or just assume one exists.
                 # Actually, let's assume Phase 1 is done and we can check 'data/seed/potential.asi'
                 # But if config didn't specify it, we generate it.

                 # Let's create a temporary working dir for initialization or just do it in the first iteration dir?
                 # Better to do it once.
                 init_dir = original_cwd / "data" / "seed"
                 init_dir.mkdir(parents=True, exist_ok=True)
                 os.chdir(init_dir)

                 current_asi = self.trainer.update_active_set(str(initial_dataset_path), str(potential_yaml_path))
                 logger.info(f"Initial Active Set generated: {current_asi}")

                 os.chdir(original_cwd)
             except Exception as e:
                 logger.error(f"Failed to generate initial active set: {e}")
                 os.chdir(original_cwd)
                 return

        # If still no active set, MaxVolSampler might fail.
        if not current_asi:
             logger.warning("No Active Set available. Sampling might be suboptimal or fail.")

        while True:
            iteration += 1
            work_dir = Path(f"data/iteration_{iteration}")
            work_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"--- Starting Iteration {iteration} ---")

            try:
                os.chdir(work_dir)

                # Resolve paths relative to original CWD
                abs_potential_path = self._resolve_path(current_potential, original_cwd)
                abs_asi_path = self._resolve_path(current_asi, original_cwd) if current_asi else None

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

                    # Sample
                    # We pass all necessary info via kwargs to support MaxVolSampler
                    # Note: We need to pass 'atoms' for MaxGammaSampler (legacy)
                    # and 'dump_file'/'potential_yaml'/'asi_path' for MaxVolSampler.

                    # For compatibility, we can read the last frame for MaxGammaSampler
                    # But if we use MaxVolSampler, we prefer it to handle the file.
                    # Since we don't know which sampler we have (polymorphism), we prepare everything.

                    last_frame = read(dump_file, index=-1, format="lammps-dump-text")
                    self._ensure_chemical_symbols(last_frame)

                    sample_kwargs = {
                        "atoms": last_frame,
                        "n_clusters": self.config.al_params.n_clusters,
                        "dump_file": str(dump_file),
                        "potential_yaml_path": str(potential_yaml_path),
                        "asi_path": str(abs_asi_path) if abs_asi_path else None,
                        "elements": self.config.md_params.elements
                    }

                    # Sampler returns List[Tuple[Atoms, int]]
                    selected_samples = self.sampler.sample(**sample_kwargs)

                    labeled_clusters = []
                    logger.info(f"Generating and labeling {len(selected_samples)} small cells...")

                    for atoms, center_id in selected_samples:
                        try:
                            # Generate Small Cell
                            cell = self.generator.generate_cell(atoms, center_id, str(abs_potential_path))

                            # Label
                            labeled_cluster = self.labeler.label(cell)
                            if labeled_cluster is None:
                                logger.warning(f"Labeling failed for cluster {center_id}. Skipping.")
                                continue

                            labeled_clusters.append(labeled_cluster)
                        except Exception as e:
                            logger.warning(f"Processing failed for cluster {center_id}: {e}. Skipping.")
                            continue

                    if not labeled_clusters:
                        logger.error("No clusters labeled successfully. Aborting active learning loop.")
                        break

                    # Train
                    logger.info("Training new potential...")
                    dataset_path = self.trainer.prepare_dataset(labeled_clusters)

                    # Train and update Active Set
                    # Note: we pass potential_yaml_path so trainer can update active set
                    new_potential = self.trainer.train(
                        dataset_path=dataset_path,
                        initial_potential=str(abs_potential_path),
                        potential_yaml_path=str(potential_yaml_path),
                        asi_path=str(abs_asi_path) if abs_asi_path else None
                    )

                    current_potential = str(Path(new_potential).resolve())

                    # If Trainer updated the active set, we should ideally get the new path.
                    # Currently trainer.train returns only potential path.
                    # However, Trainer stores the active set in 'potential.asi' in the working dir.
                    # So we should update 'current_asi' to point to this new file.
                    new_asi_path = work_dir / "potential.asi"
                    if new_asi_path.exists():
                        current_asi = str(new_asi_path.resolve())
                        logger.info(f"Updating current active set path to: {current_asi}")

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
