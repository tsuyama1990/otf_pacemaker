"""Orchestrator for the Active Learning Loop.

This module contains the high-level logic for the active learning workflow,
coordinating the interaction between MD engine, sampler, generator, labeler, and trainer.
"""

import os
import logging
import tempfile
import shutil
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
from ase.io import read, write
from ase import Atoms
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from src.core.interfaces import MDEngine, KMCEngine, Sampler, StructureGenerator, Labeler, Trainer
from src.core.enums import SimulationState, KMCStatus
from src.core.config import Config
from src.utils.logger import CSVLogger

logger = logging.getLogger(__name__)


def _run_labeling_task(labeler: Labeler, structure: Atoms) -> Optional[Atoms]:
    """Helper function to run labeling in a temporary directory to avoid conflicts.

    This function is intended to be run in a separate process.
    """
    tmpdir = tempfile.mkdtemp(prefix="label_task_")
    original_cwd = os.getcwd()

    try:
        os.chdir(tmpdir)
        return labeler.label(structure)
    except Exception as e:
        logger.error(f"Labeling task failed in {tmpdir}: {e}")
        return None
    finally:
        os.chdir(original_cwd)
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
        self.config = config
        self.md_engine = md_engine
        self.kmc_engine = kmc_engine
        self.sampler = sampler
        self.generator = generator
        self.labeler = labeler
        self.trainer = trainer
        self.csv_logger = csv_logger or CSVLogger()
        self.max_workers = config.al_params.num_parallel_labeling
        self.state_file = Path("orchestrator_state.json")

    def _save_state(self, state: Dict[str, Any]):
        """Save the current orchestrator state to a JSON file."""
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=4)

    def _load_state(self) -> Dict[str, Any]:
        """Load the orchestrator state from a JSON file."""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return {}

    def _label_clusters_parallel(self, clusters: List[Atoms]) -> List[Atoms]:
        """Label clusters in parallel using ProcessPoolExecutor."""
        labeled_results = []

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_cluster = {
                executor.submit(_run_labeling_task, self.labeler, c): c
                for c in clusters
            }

            for future in as_completed(future_to_cluster):
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
        """Trigger Active Learning pipeline for given structures."""
        logger.info(f"Triggering AL for {len(uncertain_structures)} uncertain structures.")

        clusters_to_label = []
        for atoms in uncertain_structures:
            temp_dump = work_dir / "temp_uncertain.lammpstrj"
            write(temp_dump, atoms, format="lammps-dump-text")

            sample_kwargs = {
                "atoms": atoms,
                "n_clusters": self.config.al_params.n_clusters,
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

        logger.info(f"Labeling {len(clusters_to_label)} clusters in parallel...")
        labeled_clusters = self._label_clusters_parallel(clusters_to_label)

        if not labeled_clusters:
            logger.error("No clusters labeled successfully.")
            return None

        logger.info("Training new potential (Fine-Tuning)...")
        dataset_path = self.trainer.prepare_dataset(labeled_clusters)

        new_potential = self.trainer.train(
            dataset_path=dataset_path,
            initial_potential=str(potential_path),
            potential_yaml_path=str(potential_yaml_path),
            asi_path=str(asi_path) if asi_path else None
        )

        return new_potential

    def run(self):
        """Executes the active learning loop (Hybrid MD-kMC) with state persistence."""

        # --- State Initialization ---
        saved_state = self._load_state()

        if saved_state:
            logger.info("Resuming from saved state.")
            iteration = saved_state.get("iteration", 1)
            current_potential = saved_state.get("current_potential")
            current_structure = saved_state.get("current_structure")
            current_asi = saved_state.get("current_asi")
            is_restart = saved_state.get("is_restart", True)
        else:
            logger.info("Starting new simulation.")
            iteration = 1 # Start from 1
            current_potential = self.config.al_params.initial_potential
            current_structure = self.config.md_params.initial_structure
            current_asi = self.config.al_params.initial_active_set_path
            is_restart = False

        original_cwd = Path.cwd()

        # Resolve paths that don't change
        potential_yaml_path = self._resolve_path(self.config.al_params.potential_yaml_path, original_cwd)
        initial_dataset_path = None
        if self.config.al_params.initial_dataset_path:
             initial_dataset_path = self._resolve_path(self.config.al_params.initial_dataset_path, original_cwd)

        # 0. Initialize Active Set if needed
        if not current_asi and initial_dataset_path and iteration == 1:
             logger.info("No initial Active Set provided. Generating from initial dataset...")
             try:
                 init_dir = original_cwd / "data" / "seed"
                 init_dir.mkdir(parents=True, exist_ok=True)
                 os.chdir(init_dir)

                 current_asi = self.trainer.update_active_set(str(initial_dataset_path), str(potential_yaml_path))
                 logger.info(f"Initial Active Set generated: {current_asi}")

                 # Save initial state
                 self._save_state({
                     "iteration": iteration,
                     "current_potential": current_potential,
                     "current_structure": current_structure,
                     "current_asi": current_asi,
                     "is_restart": is_restart
                 })

                 os.chdir(original_cwd)
             except Exception as e:
                 logger.error(f"Failed to generate initial active set: {e}")
                 os.chdir(original_cwd)
                 return

        while True:
            # Check for max_al_retries loop (Prevent Infinite Loop)
            # Implemented per-iteration retry counter? No, "Same iteration max retries"
            # Since iteration increments on loop, we track retries within the loop or strictly increment iteration for every MD run.
            # Let's track retries in state if we want persistence, or just increment iteration.
            # The prompt says: "max_al_retries counter ... if retraining > 3 times in same iteration -> Error"
            # This implies we loop INSIDE the iteration for retries or we don't increment iteration.
            # My current loop increments iteration at start.

            # Let's assume standard behavior:
            # 1 Iteration = 1 MD Block (+ potential AL).
            # If AL happens, we might stay in the same iteration conceptually or move to next.
            # Usually we move to next. But if we fail immediately again, we are in a loop.
            # Let's count consecutive AL triggers.

            al_consecutive_counter = saved_state.get("al_consecutive_counter", 0)

            work_dir = Path(f"data/iteration_{iteration}")
            work_dir.mkdir(parents=True, exist_ok=True)

            # Update State on disk
            self._save_state({
                "iteration": iteration,
                "current_potential": current_potential,
                "current_structure": current_structure,
                "current_asi": current_asi,
                "is_restart": is_restart,
                "al_consecutive_counter": al_consecutive_counter
            })

            logger.info(f"--- Starting Iteration {iteration} (AL Retries: {al_consecutive_counter}) ---")

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
                    # Check safety valve
                    if al_consecutive_counter >= 3:
                        raise RuntimeError("Max AL retries reached. Infinite loop detected.")

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

                        # Increment counters
                        iteration += 1
                        al_consecutive_counter += 1
                        continue # Restart loop
                    else:
                        break # AL failed

                elif state == SimulationState.FAILED:
                    logger.error("MD Failed.")
                    break

                # MD Success -> Reset AL counter
                al_consecutive_counter = 0

                # --- 2. KMC Phase ---
                if self.config.kmc_params.active:
                    logger.info("Starting KMC Phase...")

                    restart_file = Path("restart.chk")
                    if restart_file.exists():
                         dump_file = Path("dump.lammpstrj")
                         if dump_file.exists():
                             kmc_input_atoms = read(dump_file, index=-1, format="lammps-dump-text")
                             self._ensure_chemical_symbols(kmc_input_atoms)
                         else:
                             logger.error("No structure available for KMC.")
                             break
                    else:
                        logger.error("No restart file found after MD.")
                        break

                    kmc_result = self.kmc_engine.run_step(kmc_input_atoms, str(abs_potential_path))

                    if kmc_result.status == KMCStatus.SUCCESS:
                        logger.info(f"KMC Event Successful. Time step: {kmc_result.time_step:.3e} s")

                        MaxwellBoltzmannDistribution(
                            kmc_result.structure,
                            temperature_K=self.config.md_params.temperature
                        )

                        next_input_file = work_dir / "kmc_output.data"
                        write(next_input_file, kmc_result.structure, format="lammps-data", velocities=True)

                        current_structure = str(next_input_file.resolve())
                        is_restart = False
                        iteration += 1

                    elif kmc_result.status == KMCStatus.UNCERTAIN:
                        logger.info("KMC detected uncertainty. Triggering AL.")

                        if al_consecutive_counter >= 3:
                            raise RuntimeError("Max AL retries reached in KMC. Infinite loop detected.")

                        new_pot = self._trigger_al([kmc_result.structure], abs_potential_path, potential_yaml_path, abs_asi_path, work_dir)

                        if new_pot:
                            current_potential = str(Path(new_pot).resolve())

                            new_asi = work_dir / "potential.asi"
                            if new_asi.exists():
                                current_asi = str(new_asi.resolve())

                            is_restart = True # Resume from where we were
                            iteration += 1
                            al_consecutive_counter += 1
                            continue
                        else:
                            break

                    elif kmc_result.status == KMCStatus.NO_EVENT:
                        logger.info("KMC found no event. Continuing MD.")
                        is_restart = True
                        iteration += 1

                else:
                    if state == SimulationState.COMPLETED:
                        logger.info("MD block completed. Continuing...")
                        is_restart = True
                        iteration += 1


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
        # Check current_structure from state first (if explicitly set to a .data file by KMC)
        if not is_restart and initial_structure and initial_structure.endswith(".data"):
             return str(self._resolve_path(initial_structure, base_cwd))

        if is_restart:
            # Look in previous iterations for restart.chk
            # Since we increment iteration at the end of the loop,
            # "iteration" variable is the CURRENT new iteration.
            # We want to restart from the PREVIOUS one.

            # Simple heuristic: scan backwards from iteration-1
            search_start = iteration - 1
            if search_start < 1:
                # No previous iteration, maybe initial structure?
                # But is_restart=True usually implies we ran at least once.
                # Or we are resuming a crashed run in iter 1.
                # If we are in iter 1 and restart is true, we look in iter 1?
                pass

            prev_dir = base_cwd / f"data/iteration_{search_start}"
            restart_file = prev_dir / "restart.chk"

            if not restart_file.exists():
                # Check current dir (if we crashed mid-way and are resuming same iter)
                curr_dir = base_cwd / f"data/iteration_{iteration}"
                curr_restart = curr_dir / "restart.chk"
                if curr_restart.exists():
                    return str(curr_restart)

            if not restart_file.exists():
                 # Maybe we have KMC output from previous?
                 # But that would be caught by the first if block (current_structure updated).
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
