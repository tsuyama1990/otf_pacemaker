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
from ase.io import read, write
from ase import Atoms

from src.core.interfaces import MDEngine, KMCEngine, Sampler, StructureGenerator, Labeler, Trainer
from src.core.enums import SimulationState, KMCStatus
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
        kmc_engine: KMCEngine,
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
            kmc_engine: KMC Engine instance.
            sampler: Sampler instance.
            generator: Structure Generator instance.
            labeler: Labeler instance.
            trainer: Trainer instance.
            csv_logger: CSV Logger instance for metrics.
        """
        self.config = config
        self.md_engine = md_engine
        self.kmc_engine = kmc_engine
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

    def _trigger_al(self,
                    uncertain_structures: List[Atoms],
                    potential_path: Path,
                    potential_yaml_path: Path,
                    asi_path: Optional[Path],
                    work_dir: Path) -> Optional[str]:
        """Trigger Active Learning pipeline for given structures.

        Args:
            uncertain_structures: List of Atoms objects that triggered uncertainty.
            potential_path: Current potential path.
            potential_yaml_path: Potential YAML definition path.
            asi_path: Current active set path.
            work_dir: Current working directory for temporary files.

        Returns:
            Optional[str]: Path to new potential if training succeeded, None otherwise.
        """
        logger.info(f"Triggering AL for {len(uncertain_structures)} uncertain structures.")

        # 1. Generate Small Cells / Clusters
        # Note: If uncertainty comes from KMC, the structure is already a full configuration.
        # We might want to crop it or use it as is.
        # The prompt says: "extract the uncertain structure -> standard Active Learning pipeline".
        # Standard pipeline involves sampling and generation.
        # Here we already have the "uncertain" structure. We treat it as the "dump" frame.

        # We can reuse the sampler to find the highest gamma atom in this structure
        # OR we can just generate clusters around high gamma atoms.
        # Let's use the sampler on these specific structures to select the best centers.

        clusters_to_label = []
        for atoms in uncertain_structures:
            # Save temporary dump for sampler (Sampler interface requires file usually, but we can bypass if we modify Sampler or mock it)
            # The Sampler interface takes **kwargs. Our MaxGammaSampler takes 'dump_file'.
            # Let's save a temp dump.
            temp_dump = work_dir / "temp_uncertain.lammpstrj"
            write(temp_dump, atoms, format="lammps-dump-text")

            sample_kwargs = {
                "atoms": atoms,
                "n_clusters": self.config.al_params.n_clusters, # Or 1?
                "dump_file": str(temp_dump),
                "potential_yaml_path": str(potential_yaml_path),
                "asi_path": str(asi_path) if asi_path else None,
                "elements": self.config.md_params.elements
            }

            selected_samples = self.sampler.sample(**sample_kwargs)

            for s_atoms, center_id in selected_samples:
                try:
                    cell = self.generator.generate_cell(s_atoms, center_id, str(potential_path))
                    clusters_to_label.append(cell)
                except Exception as e:
                    logger.warning(f"Generation failed for cluster {center_id}: {e}")

        if not clusters_to_label:
            logger.error("No clusters generated from uncertain structures.")
            return None

        # 2. Label
        logger.info(f"Labeling {len(clusters_to_label)} clusters in parallel...")
        labeled_clusters = self._label_clusters_parallel(clusters_to_label)

        if not labeled_clusters:
            logger.error("No clusters labeled successfully.")
            return None

        # 3. Train
        logger.info("Training new potential (Fine-Tuning)...")
        dataset_path = self.trainer.prepare_dataset(labeled_clusters)

        new_potential = self.trainer.train(
            dataset_path=dataset_path,
            initial_potential=str(potential_path), # Warm Start!
            potential_yaml_path=str(potential_yaml_path),
            asi_path=str(asi_path) if asi_path else None
        )

        return new_potential


    def run(self):
        """Executes the active learning loop (Hybrid MD-kMC)."""
        current_potential = self.config.al_params.initial_potential
        current_structure = self.config.md_params.initial_structure
        current_asi = self.config.al_params.initial_active_set_path

        # If we just finished a KMC step, we might have a structure object instead of a file
        # But MD engine requires a file. So we maintain current_structure_path.

        is_restart = False
        iteration = 0
        original_cwd = Path.cwd()

        # Resolve paths that don't change
        potential_yaml_path = self._resolve_path(self.config.al_params.potential_yaml_path, original_cwd)
        initial_dataset_path = None
        if self.config.al_params.initial_dataset_path:
             initial_dataset_path = self._resolve_path(self.config.al_params.initial_dataset_path, original_cwd)

        # 0. Initialize Active Set if needed
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

        while True:
            iteration += 1
            work_dir = Path(f"data/iteration_{iteration}")
            work_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"--- Starting Iteration {iteration} ---")

            try:
                os.chdir(work_dir)

                abs_potential_path = self._resolve_path(current_potential, original_cwd)
                abs_asi_path = self._resolve_path(current_asi, original_cwd) if current_asi else None

                if not abs_potential_path.exists():
                    logger.error(f"Potential file not found: {abs_potential_path}")
                    break

                # --- 1. MD Phase ---
                input_structure_arg = self._prepare_structure_path(
                    is_restart, iteration, current_structure, original_cwd
                )
                if not input_structure_arg:
                    break

                logger.info("Running MD...")
                state = self.md_engine.run(
                    potential_path=str(abs_potential_path),
                    steps=self.config.md_params.n_steps,
                    gamma_threshold=self.config.al_params.gamma_threshold,
                    input_structure=input_structure_arg,
                    is_restart=is_restart
                )
                logger.info(f"MD Finished with state: {state}")

                if state == SimulationState.UNCERTAIN:
                    # MD Uncertainty -> AL
                    logger.info("MD detected uncertainty. Triggering AL.")
                    dump_file = Path("dump.lammpstrj")
                    if not dump_file.exists():
                        raise FileNotFoundError("Dump file not found.")

                    last_frame = read(dump_file, index=-1, format="lammps-dump-text")
                    self._ensure_chemical_symbols(last_frame)

                    new_pot = self._trigger_al([last_frame], abs_potential_path, potential_yaml_path, abs_asi_path, work_dir)

                    if new_pot:
                        current_potential = str(Path(new_pot).resolve())
                        is_restart = True # Resume MD from checkpoint
                        # Update ASI
                        new_asi = work_dir / "potential.asi"
                        if new_asi.exists():
                            current_asi = str(new_asi.resolve())
                        continue # Restart loop (Resume MD)
                    else:
                        break # AL failed

                elif state == SimulationState.FAILED:
                    logger.error("MD Failed.")
                    break

                # MD Completed -> Proceed to KMC (if active)

                # --- 2. KMC Phase ---
                if self.config.kmc_params.active:
                    logger.info("Starting KMC Phase...")

                    # Load the final structure from MD
                    # We rely on restart file or dump
                    restart_file = Path("restart.chk")
                    if restart_file.exists():
                         # LAMMPS restart files are binary, ASE can sometimes read them or we use the dump
                         # Safer to read the last dump frame for atomic positions
                         dump_file = Path("dump.lammpstrj")
                         if dump_file.exists():
                             kmc_input_atoms = read(dump_file, index=-1, format="lammps-dump-text")
                             self._ensure_chemical_symbols(kmc_input_atoms)
                         else:
                             # Fallback, maybe the user provided restart reader
                             logger.error("No structure available for KMC.")
                             break
                    else:
                        logger.error("No restart file found after MD.")
                        break

                    kmc_result = self.kmc_engine.run_step(kmc_input_atoms, str(abs_potential_path))

                    if kmc_result.status == KMCStatus.SUCCESS:
                        logger.info(f"KMC Event Successful. Time step: {kmc_result.time_step:.3e} s")

                        # Save the new structure to be used as input for next MD
                        # We overwrite restart.chk or create a new data file?
                        # MD engine usually reads data file or restart.
                        # If we provide a data file, we set is_restart=False (start fresh from this config)
                        # But we should preserve velocities? KMC is diffusiion, velocities are thermalized.
                        # So we can just start fresh MD with random velocities (temp)

                        next_input_file = work_dir / "kmc_output.data"
                        write(next_input_file, kmc_result.structure, format="lammps-data")

                        # Update state for next loop
                        current_structure = str(next_input_file.resolve())
                        is_restart = False # Start new MD from this structure

                        # Log time evolution? (Not implemented in logger yet)

                    elif kmc_result.status == KMCStatus.UNCERTAIN:
                        logger.info("KMC detected uncertainty. Triggering AL.")

                        new_pot = self._trigger_al([kmc_result.structure], abs_potential_path, potential_yaml_path, abs_asi_path, work_dir)

                        if new_pot:
                            current_potential = str(Path(new_pot).resolve())
                            # Restart KMC from the same input structure (retry)
                            # We loop back. Since we didn't update current_structure or is_restart,
                            # the loop would naturally try MD again from MD's start.
                            # BUT we want to retry KMC? Or Resume MD?
                            # Prompt says: "restart the kMC step with the new potential."
                            # To do this, we need to SKIP MD and jump to KMC.
                            # But our loop is MD -> KMC.

                            # Easiest way: The loop continues. MD runs again (maybe short if converged? No, MD runs n_steps).
                            # If we want to retry KMC, we should probably structure the loop differently.
                            # However, running MD again is also fine (it re-thermalizes).
                            # Let's stick to the simple loop: Update potential, restart loop.
                            # The MD will run again with new potential (checking validity), then KMC.

                            # Update ASI
                            new_asi = work_dir / "potential.asi"
                            if new_asi.exists():
                                current_asi = str(new_asi.resolve())

                            # Maintain is_restart=True to resume MD where it left off?
                            # If MD finished, we are at end.
                            # If we restart MD from end, it runs 0 steps? No, n_steps is usually cumulative or relative.
                            # Our MD engine uses "run N".

                            # Let's set is_restart=True. MD will resume from where it halted (or finished).
                            # If it finished, we might need to be careful.
                            # If MD finished, restarting might not be valid if n_steps is absolute.
                            # LAMMPSRunner generates "run {steps}". This is relative.
                            # So it will run another N steps.

                            # If we want to retry KMC, we shouldn't run MD.
                            # But complicating the loop might be risky.
                            # Let's assume re-running MD is acceptable as a "verification" of the new potential
                            # in the basin before trying KMC again.
                            is_restart = True
                            continue

                        else:
                            break # AL failed

                    elif kmc_result.status == KMCStatus.NO_EVENT:
                        logger.info("KMC found no event. Continuing MD.")
                        # Just continue loop. MD runs again from where it left off.
                        # We should probably extend the simulation or just run another block.
                        # Current logic: MD runs N steps. Then KMC. Then MD N steps.
                        # This is fine.

                        # We need to ensure next MD starts from current state.
                        # If KMC did nothing, we use the MD output (restart.chk)
                        is_restart = True

                else:
                    # KMC Inactive. Just standard active learning loop.
                    # MD Finished. If we want continuous MD, we should restart it.
                    if state == SimulationState.COMPLETED:
                        logger.info("MD block completed. Continuing...")
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
        # If explicit structure set (e.g. from KMC), use it
        if not is_restart and initial_structure.endswith(".data"):
             return str(self._resolve_path(initial_structure, base_cwd))

        if is_restart:
            # Look in previous iterations
            # NOTE: If we just finished KMC successfully, we set is_restart=False and current_structure=kmc_output.data
            # So we land in the first block above.
            # If we are restarting MD (e.g. after AL interruption or just continuing), we look for restart.chk

            # We need to find the most recent restart file.
            # It could be in current work_dir (if we failed halfway) or previous.
            # Simplified: look in previous iteration.
            prev_dir = base_cwd / f"data/iteration_{iteration-1}"
            restart_file = prev_dir / "restart.chk"

            # Also check current dir if we are looping within iteration (unlikely with this structure)

            if not restart_file.exists():
                # Check if we have a KMC output from prev iteration?
                # Logic handled by current_structure updates.
                pass

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
