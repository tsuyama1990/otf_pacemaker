"""Orchestrator for the Active Learning Loop.

This module contains the high-level logic for the active learning workflow,
coordinating the interaction between MD engine, sampler, generator, labeler, and trainer.
"""

import os
import logging
import tempfile
import shutil
from pathlib import Path
from typing import Optional, List
from concurrent.futures import ProcessPoolExecutor, as_completed
from ase.io import read
from ase import Atoms

from src.core.interfaces import MDEngine, Sampler, StructureGenerator, Labeler, Trainer
from src.core.enums import SimulationState
from src.core.config import Config
from src.utils.logger import CSVLogger

logger = logging.getLogger(__name__)


def _run_labeling_task(labeler: Labeler, structure: Atoms) -> Optional[Atoms]:
    """Helper function to run labeling in a temporary directory to avoid conflicts.

    This function is intended to be run in a separate process.
    """
    # Create a temporary directory for this task
    # We use mkdtemp to ensure we have a unique directory
    tmpdir = tempfile.mkdtemp(prefix="label_task_")
    original_cwd = os.getcwd()

    try:
        os.chdir(tmpdir)
        # Run labeling
        # Note: If labeler uses relative paths for resources (like pseudopotentials),
        # they must have been resolved to absolute paths before passing here.
        return labeler.label(structure)
    except Exception as e:
        logger.error(f"Labeling task failed in {tmpdir}: {e}")
        return None
    finally:
        os.chdir(original_cwd)
        # Clean up temporary directory
        try:
            shutil.rmtree(tmpdir)
        except Exception as e:
            logger.warning(f"Failed to cleanup temp dir {tmpdir}: {e}")


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
        csv_logger: Optional[CSVLogger] = None
    ):
        """Initialize the orchestrator.

        Args:
            config: Configuration object.
            md_engine: MD Engine instance.
            sampler: Sampler instance.
            generator: Structure Generator instance.
            labeler: Labeler instance.
            trainer: Trainer instance.
            csv_logger: CSV Logger instance for metrics.
        """
        self.config = config
        self.md_engine = md_engine
        self.sampler = sampler
        self.generator = generator
        self.labeler = labeler
        self.trainer = trainer
        self.csv_logger = csv_logger or CSVLogger()
        self.max_workers = config.al_params.num_parallel_labeling

    def _label_clusters_parallel(self, clusters: List[Atoms]) -> List[Atoms]:
        """Label clusters in parallel using ProcessPoolExecutor.

        Args:
            clusters: List of atomic structures to label.

        Returns:
            List[Atoms]: List of successfully labeled structures.
        """
        labeled_results = []

        # We need to make sure 'self.labeler' is safe to pickle.
        # The Labeler (DeltaLabeler) contains calculator objects.
        # ASE calculators should be picklable, but if they have open files it might fail.
        # Espresso calculator is generally picklable.

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_cluster = {
                executor.submit(_run_labeling_task, self.labeler, c): c
                for c in clusters
            }

            for future in as_completed(future_to_cluster):
                cluster_in = future_to_cluster[future]
                try:
                    result = future.result()
                    if result is not None:
                        labeled_results.append(result)
                except Exception as e:
                    logger.error(f"Parallel labeling execution failed for a cluster: {e}")

        return labeled_results

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

            # Metrics for this iteration
            iter_max_gamma = 0.0
            iter_n_added = 0
            iter_active_set_size = 0
            iter_rmse_train = None # Not easily available from here yet without parsing logs

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

                # Check for max gamma if available in logs or dump (simplified)
                # MD engine might not return max gamma directly, but we can try to extract it from dump later
                # For now, we get it during sampling.

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
                    last_frame = read(dump_file, index=-1, format="lammps-dump-text")
                    self._ensure_chemical_symbols(last_frame)

                    # Try to estimate max gamma from the last frame for logging
                    if 'f_f_gamma' in last_frame.arrays:
                         import numpy as np
                         iter_max_gamma = np.max(last_frame.arrays['f_f_gamma'])

                    sample_kwargs = {
                        "atoms": last_frame,
                        "n_clusters": self.config.al_params.n_clusters,
                        "dump_file": str(dump_file),
                        "potential_yaml_path": str(potential_yaml_path),
                        "asi_path": str(abs_asi_path) if abs_asi_path else None,
                        "elements": self.config.md_params.elements
                    }

                    selected_samples = self.sampler.sample(**sample_kwargs)

                    logger.info(f"Generating {len(selected_samples)} small cells...")

                    clusters_to_label = []
                    for atoms, center_id in selected_samples:
                         try:
                             cell = self.generator.generate_cell(atoms, center_id, str(abs_potential_path))
                             clusters_to_label.append(cell)
                         except Exception as e:
                             logger.warning(f"Generation failed for cluster {center_id}: {e}")

                    logger.info(f"Labeling {len(clusters_to_label)} clusters in parallel...")
                    labeled_clusters = self._label_clusters_parallel(clusters_to_label)

                    iter_n_added = len(labeled_clusters)

                    if not labeled_clusters:
                        logger.error("No clusters labeled successfully. Aborting active learning loop.")
                        break

                    # Train
                    logger.info("Training new potential...")
                    dataset_path = self.trainer.prepare_dataset(labeled_clusters)

                    # Train and update Active Set
                    new_potential = self.trainer.train(
                        dataset_path=dataset_path,
                        initial_potential=str(abs_potential_path),
                        potential_yaml_path=str(potential_yaml_path),
                        asi_path=str(abs_asi_path) if abs_asi_path else None
                    )

                    current_potential = str(Path(new_potential).resolve())

                    # Update current ASI path
                    new_asi_path = work_dir / "potential.asi"
                    if new_asi_path.exists():
                        current_asi = str(new_asi_path.resolve())
                        # Estimate active set size (number of lines in ASI usually)
                        with open(new_asi_path, 'r') as f:
                             iter_active_set_size = sum(1 for line in f)

                        logger.info(f"Updating current active set path to: {current_asi}")

                    is_restart = True
                    logger.info(f"New potential trained: {current_potential}")

                    # Log metrics
                    self.csv_logger.log_metrics(
                        iteration=iteration,
                        max_gamma=iter_max_gamma,
                        n_added=iter_n_added,
                        active_set_size=iter_active_set_size,
                        rmse_energy=None, # RMSE not captured from Trainer yet
                        rmse_forces=None
                    )

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
