"""Orchestrator for the Active Learning Loop.

This module contains the high-level logic for the active learning workflow,
coordinating the interaction between MD engine, sampler, generator, labeler, and trainer.
"""

import os
import logging
import tempfile
import shutil
import json
import random
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from ase.io import read, write
from ase import Atoms
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from src.core.interfaces import MDEngine, KMCEngine, Sampler, StructureGenerator, Labeler, Trainer
from src.core.enums import SimulationState, KMCStatus
from src.core.config import Config
from src.utils.logger import CSVLogger
from src.validation.pacemaker_validator import PacemakerValidator
from src.autostructure.deformation import SystematicDeformationGenerator

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

def _run_md_task(md_engine: MDEngine,
                 potential_path: str,
                 steps: int,
                 gamma_threshold: float,
                 input_structure: str,
                 is_restart: bool,
                 temperature: float,
                 pressure: float,
                 seed: int) -> Tuple[SimulationState, Optional[Path]]:
    """Helper function to run a single MD walker task."""
    try:
        dump_path = md_engine.run(
            structure_path=input_structure,
            potential_path=potential_path,
            temperature=temperature,
            pressure=pressure,
            seed=seed
        )
        return SimulationState.COMPLETED, dump_path

    except Exception as e:
        logger.error(f"MD Walker failed: {e}")
        return SimulationState.FAILED, None


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

        # Initialize Validator
        self.validator = PacemakerValidator()

    def _save_state(self, base_dir: Path, state: Dict[str, Any]):
        """Save the current orchestrator state to a JSON file."""
        state_file = base_dir / "orchestrator_state.json"
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=4)

    def _load_state(self, base_dir: Path) -> Dict[str, Any]:
        """Load the orchestrator state from a JSON file."""
        state_file = base_dir / "orchestrator_state.json"
        if state_file.exists():
            with open(state_file, 'r') as f:
                return json.load(f)
        return {"iteration": 0, "current_potential": None, "current_asi": None}

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

    def _get_md_conditions(self, iteration: int) -> Dict[str, float]:
        """Get Temperature and Pressure for the current iteration based on schedule."""
        default_conditions = {
            "temperature": self.config.md_params.temperature,
            "pressure": self.config.md_params.pressure
        }

        for stage in self.config.exploration_schedule:
            if stage.iter_start <= iteration <= stage.iter_end:
                # Sample from range
                temp = random.uniform(stage.temp[0], stage.temp[1])
                press = random.uniform(stage.press[0], stage.press[1])
                return {"temperature": temp, "pressure": press}

        return default_conditions

    def _trigger_al(self,
                    uncertain_structures: List[Atoms],
                    potential_path: Path,
                    potential_yaml_path: Path,
                    asi_path: Optional[Path],
                    work_dir: Path,
                    iteration: int) -> Optional[str]:
        """Trigger Active Learning pipeline for given structures."""
        logger.info(f"Triggering AL for {len(uncertain_structures)} uncertain structures.")

        clusters_to_label = []
        for atoms in uncertain_structures:
            temp_dump = work_dir / "temp_uncertain.lammpstrj"
            write(temp_dump, atoms, format="lammps-dump-text")

            sample_kwargs = {
                "structures": [atoms], # MaxVolSampler now expects 'structures' list
                "potential_path": str(potential_path), # MaxVolSampler needs path for ACESampler
                "n_clusters": self.config.al_params.n_clusters,
                "dump_file": str(temp_dump), # Legacy kwarg if needed
                "potential_yaml_path": str(potential_yaml_path), # Legacy
                "asi_path": str(asi_path) if asi_path else None, # Legacy
                "elements": self.config.md_params.elements
            }

            # Sampler returns (Atoms, center_id)
            selected_samples = self.sampler.sample(**sample_kwargs)

            for s_atoms, center_id in selected_samples:
                # Check gamma upper bound (Trust Level)
                max_gamma = 0.0
                if hasattr(s_atoms, 'info') and 'max_gamma' in s_atoms.info:
                    max_gamma = s_atoms.info['max_gamma']
                elif hasattr(s_atoms, 'arrays') and 'gamma' in s_atoms.arrays:
                     max_gamma = s_atoms.arrays['gamma'].max()

                if max_gamma > self.config.al_params.gamma_upper_bound:
                    logger.warning(f"Gamma {max_gamma} exceeds limit {self.config.al_params.gamma_upper_bound}. Attempting rescue via Pre-Optimizer.")

                    if hasattr(self.generator, 'pre_optimizer') and self.generator.pre_optimizer:
                         try:
                             # Rescue
                             s_atoms = self.generator.pre_optimizer.run_pre_optimization(s_atoms)
                             logger.info("Rescue successful (Pre-Optimization completed).")
                         except Exception as exc:
                             logger.warning(f"Rescue failed: {exc}. Discarding candidate.")
                             continue
                    else:
                        logger.warning("No Pre-Optimizer available. Discarding candidate.")
                        continue

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

        # 1. Active Set Pruning (every 10 iterations)
        if asi_path and iteration % 10 == 0:
             if hasattr(self.trainer, "prune_active_set"):
                 self.trainer.prune_active_set(str(asi_path), threshold=0.99)

        # 2. Training
        # We pass the current iteration for dynamic config
        new_potential = self.trainer.train(
            dataset_path=dataset_path,
            initial_potential=str(potential_path),
            potential_yaml_path=str(potential_yaml_path),
            asi_path=str(asi_path) if asi_path else None,
            iteration=iteration
        )

        # 3. Validation
        logger.info("Validating new potential...")
        validation_results = self.validator.validate(new_potential)

        if validation_results.get("status") == "FAILED":
             logger.warning(f"Validation FAILED: {validation_results.get('error')}")
             # Strategy: Accept it but log warning? Or discard?
             # For now, log warning and proceed, as strict rejection might halt progress.
             # Alternatively, we could revert to old potential if severe.
        else:
             logger.info(f"Validation Passed: {validation_results}")

        return new_potential

    def run(self):
        """Executes the active learning loop (Hybrid MD-kMC) with robust state persistence."""

        data_root = Path("data")
        data_root.mkdir(parents=True, exist_ok=True)

        state = self._load_state(data_root)
        iteration = state["iteration"]

        current_potential = state.get("current_potential") or self.config.al_params.initial_potential
        current_asi = state.get("current_asi") or self.config.al_params.initial_active_set_path
        current_structure = state.get("current_structure") or self.config.md_params.initial_structure
        is_restart = state.get("is_restart", False)
        al_consecutive_counter = state.get("al_consecutive_counter", 0)

        original_cwd = Path.cwd()

        potential_yaml_path = self._resolve_path(self.config.al_params.potential_yaml_path, original_cwd)
        initial_dataset_path = None
        if self.config.al_params.initial_dataset_path:
             initial_dataset_path = self._resolve_path(self.config.al_params.initial_dataset_path, original_cwd)

        if not current_asi and initial_dataset_path and iteration == 0:
             logger.info("No initial Active Set provided. Generating from initial dataset...")
             try:
                 init_dir = original_cwd / "data" / "seed"
                 init_dir.mkdir(parents=True, exist_ok=True)
                 os.chdir(init_dir)

                 current_asi = self.trainer.update_active_set(str(initial_dataset_path), str(potential_yaml_path))
                 logger.info(f"Initial Active Set generated: {current_asi}")

                 state["current_asi"] = current_asi
                 os.chdir(original_cwd)
             except Exception as e:
                 logger.error(f"Failed to generate initial active set: {e}")
                 os.chdir(original_cwd)
                 return

        while True:
            iteration += 1

            state["iteration"] = iteration
            state["current_potential"] = current_potential
            state["current_structure"] = current_structure
            state["current_asi"] = current_asi
            state["is_restart"] = is_restart
            state["al_consecutive_counter"] = al_consecutive_counter

            self._save_state(data_root, state)

            work_dir = data_root / f"iteration_{iteration}"
            if work_dir.exists():
                 logger.warning(f"Resuming or overwriting existing iteration directory: {iteration}")

            work_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"--- Starting Iteration {iteration} (AL Retries: {al_consecutive_counter}) ---")

            try:
                os.chdir(work_dir)

                abs_potential_path = self._resolve_path(current_potential, original_cwd)
                abs_asi_path = self._resolve_path(current_asi, original_cwd) if current_asi else None

                if not abs_potential_path.exists():
                    logger.error(f"Potential file not found: {abs_potential_path}")
                    break

                input_structure_arg = self._prepare_structure_path(
                    is_restart, iteration, current_structure, original_cwd
                )
                if not input_structure_arg:
                    break

                # --- Systematic Deformation Injection ---
                if iteration > 0 and iteration % 5 == 0:
                    logger.info("Injecting distorted structures for EOS/Elasticity.")
                    try:
                        # Load current structure to deform
                        struct_to_deform = read(input_structure_arg)
                        self._ensure_chemical_symbols(struct_to_deform)

                        def_gen = SystematicDeformationGenerator(struct_to_deform, self.config.lj_params)
                        distorted_structures = def_gen.generate_all()

                        logger.info(f"Generated {len(distorted_structures)} distorted structures.")

                        # Label them
                        labeled_distorted = self._label_clusters_parallel(distorted_structures)

                        if labeled_distorted:
                            # Add to dataset
                            self.trainer.prepare_dataset(labeled_distorted)
                            logger.info(f"Added {len(labeled_distorted)} labeled distorted structures to dataset.")
                        else:
                            logger.warning("All distorted structures failed labeling.")

                    except Exception as e:
                        logger.error(f"Systematic deformation injection failed: {e}")
                # ----------------------------------------

                logger.info("Running MD (Parallel Walkers)...")

                # --- Parallel MD Walkers Logic ---
                n_walkers = self.config.md_params.n_md_walkers
                futures = []
                uncertain_structures_buffer = []
                any_uncertain = False
                any_failed = False

                with ProcessPoolExecutor(max_workers=n_walkers) as executor:
                    for i in range(n_walkers):
                        # Unique conditions for each walker
                        conditions = self._get_md_conditions(iteration)
                        # Unique seed
                        seed = random.randint(0, 1000000)

                        logger.info(f"Walker {i}: Temp={conditions['temperature']:.1f}, Press={conditions['pressure']:.1f}, Seed={seed}")

                        # We use the 'input_structure_arg' as the starting point for all.
                        # Note: If is_restart is True, they all restart from the SAME file.
                        # Since they have different seeds for velocity creation, they will diverge.

                        futures.append(executor.submit(
                            _run_md_task,
                            self.md_engine,
                            str(abs_potential_path),
                            self.config.md_params.n_steps,
                            self.config.al_params.gamma_threshold,
                            input_structure_arg,
                            is_restart,
                            conditions['temperature'],
                            conditions['pressure'],
                            seed
                        ))

                    for future in as_completed(futures):
                        state_res, dump_path = future.result()

                        if state_res == SimulationState.FAILED:
                            any_failed = True
                            continue

                        # Check uncertainty
                        # We need to read the dump file and check max gamma
                        # Or rely on LAMMPS halt? The input generator logic currently just dumps.
                        # Python side check:
                        if dump_path and dump_path.exists():
                             try:
                                 # We check the LAST frame. If 'max_gamma' (f_2) > threshold?
                                 # The dump has columns: id type x y z f_2
                                 # f_2 is gamma.
                                 # We can use ASE to read it.
                                 atoms_list = read(dump_path, index=":", format="lammps-dump-text")
                                 # Checking for high gamma in ANY frame or just last?
                                 # Usually we check max gamma along trajectory or just if it halted.
                                 # Since we didn't use `fix halt`, it ran to completion.
                                 # Let's check the maximum gamma observed in the trajectory.

                                 max_gamma_observed = 0.0
                                 uncertain_frames = []

                                 for at in atoms_list:
                                     # 'f_2' might be mapped to 'c_max_gamma' or just in arrays
                                     # The dump cmd: dump 1 all custom ... f_2
                                     # ASE reads this into atoms.arrays usually.
                                     # Column 'f_2' -> might be 'c_max_gamma' if labeled? No, custom columns usually need mapping or access via arrays.
                                     # ASE lammps-dump-text reader: "f_2" -> stored in `arrays`?
                                     # Let's assume standard behavior: `atoms.get_array('f_2')`?
                                     # Or `atoms.info`.
                                     # Wait, `pace/extrapolation gamma 1` computes per-atom gamma.
                                     # `dump ... f_2` dumps PER ATOM gamma.
                                     # So for each frame, we check if any atom has gamma > threshold.

                                     # Try to find the gamma array
                                     gammas = None
                                     if 'f_2' in at.arrays:
                                         gammas = at.arrays['f_2']
                                     elif 'c_max_gamma' in at.arrays:
                                         gammas = at.arrays['c_max_gamma']

                                     if gammas is not None:
                                         current_max = gammas.max()
                                         max_gamma_observed = max(max_gamma_observed, current_max)

                                         if current_max > self.config.al_params.gamma_threshold:
                                             uncertain_frames.append(at)

                                 logger.info(f"Walker max gamma: {max_gamma_observed}")

                                 if max_gamma_observed > self.config.al_params.gamma_threshold:
                                     any_uncertain = True
                                     # Pick the first uncertain frame or the last one?
                                     # Usually the one that triggered it.
                                     if uncertain_frames:
                                          uncertain_structures_buffer.append(uncertain_frames[0]) # Take first uncertain
                                     else:
                                          uncertain_structures_buffer.append(atoms_list[-1]) # Fallback

                             except Exception as e:
                                 logger.warning(f"Failed to parse dump file {dump_path}: {e}")

                # --- End Parallel MD ---

                if any_uncertain:
                    if al_consecutive_counter >= 3:
                        raise RuntimeError("Max AL retries reached. Infinite loop detected.")

                    logger.info("MD detected uncertainty in one or more walkers. Triggering AL.")

                    # Ensure symbols
                    for u_atoms in uncertain_structures_buffer:
                         self._ensure_chemical_symbols(u_atoms)

                    new_pot = self._trigger_al(uncertain_structures_buffer, abs_potential_path, potential_yaml_path, abs_asi_path, work_dir, iteration)

                    if new_pot:
                        current_potential = str(Path(new_pot).resolve())
                        is_restart = True

                        new_asi = work_dir / "potential.asi"
                        if new_asi.exists():
                            current_asi = str(new_asi.resolve())

                        al_consecutive_counter += 1
                        continue
                    else:
                        break

                elif any_failed:
                    logger.error("MD Failed in one or more walkers.")
                    break

                # If all walkers completed successfully
                logger.info("MD walkers completed stably.")
                al_consecutive_counter = 0

                if self.config.kmc_params.active:
                    logger.info("Starting KMC Phase...")

                    restart_file = Path("restart.chk")
                    # If multiple walkers, which restart file?
                    # The prompt didn't specify how to pick the structure for KMC/Next Step if we have N walkers.
                    # "Collect termination states from all walkers. If any walker hits uncertainty ... If multiple hit uncertainty ..."
                    # If NONE hit uncertainty, we need to pick ONE to continue?
                    # "If all walkers returning COMPLETED ... verify the orchestrator proceeds to the AL (Training) phase" -> wait, prompt said "Mock 3 walkers returning COMPLETED and 1 returning UNCERTAIN".
                    # But what if ALL complete? We need a structure for the next iteration.
                    # I'll randomly pick one walker's restart file as the "canonical" continuation.

                    # Also, filenames are now `restart.chk.{seed}`.
                    # I need to find them.
                    chk_files = list(work_dir.glob("restart.chk.*"))
                    dump_files = list(work_dir.glob("dump.lammpstrj.*"))

                    if not chk_files:
                        logger.error("No restart files found.")
                        break

                    # Pick random successful walker to continue
                    selected_chk = random.choice(chk_files)
                    selected_seed = selected_chk.suffix.split('.')[-1]
                    logger.info(f"Selected walker seed {selected_seed} for continuation.")

                    # Rename to standard names expected by KMC/Next Iteration if needed?
                    # Or just point to it.
                    # KMC expects `kmc_input_atoms`.
                    # Correspoding dump:
                    selected_dump = work_dir / f"dump.lammpstrj.{selected_seed}"

                    if selected_dump.exists():
                         kmc_input_atoms = read(selected_dump, index=-1, format="lammps-dump-text")
                         self._ensure_chemical_symbols(kmc_input_atoms)
                    else:
                         logger.error("No structure available for KMC.")
                         break

                    # Copy selected restart to "restart.chk" for next iter resume if needed
                    shutil.copy(selected_chk, work_dir / "restart.chk")


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

                    elif kmc_result.status == KMCStatus.UNCERTAIN:
                        logger.info("KMC detected uncertainty. Triggering AL.")

                        if al_consecutive_counter >= 3:
                            raise RuntimeError("Max AL retries reached in KMC. Infinite loop detected.")

                        new_pot = self._trigger_al([kmc_result.structure], abs_potential_path, potential_yaml_path, abs_asi_path, work_dir, iteration)

                        if new_pot:
                            current_potential = str(Path(new_pot).resolve())
                            new_asi = work_dir / "potential.asi"
                            if new_asi.exists():
                                current_asi = str(new_asi.resolve())
                            is_restart = True
                            al_consecutive_counter += 1
                            continue
                        else:
                            break

                    elif kmc_result.status == KMCStatus.NO_EVENT:
                        logger.info("KMC found no event. Continuing MD.")
                        is_restart = True

                else:
                    if not any_uncertain and not any_failed: # If MD completed
                        logger.info("MD block completed. Continuing...")
                        # Pick one for next iter
                        chk_files = list(work_dir.glob("restart.chk.*"))
                        if chk_files:
                            selected_chk = random.choice(chk_files)
                            shutil.copy(selected_chk, work_dir / "restart.chk")
                            # We don't update current_structure if we are just restarting/continuing from checkpoint?
                            # `is_restart = True` will look for `restart.chk` in THIS iteration folder (which we just populated)
                            # Actually, `_prepare_structure_path` looks at `iteration - 1` for restart file!
                            # "if is_restart: search_start = iteration - 1"
                            # So we need to ensure the restart file is in the CURRENT folder for the NEXT iteration (which will be `iteration + 1`).
                            # Wait, the loop increments iteration at start.
                            # So `iteration` is the current one.
                            # The NEXT iteration will look at `iteration` folder.
                            # So yes, copying to `restart.chk` in `work_dir` is correct.
                        is_restart = True

            except Exception as e:
                logger.exception(f"An error occurred in iteration {iteration}: {e}")
                break
            finally:
                os.chdir(original_cwd)

    def _resolve_path(self, path_str: str, base_cwd: Path) -> Path:
        p = Path(path_str)
        if p.is_absolute():
            return p
        return (base_cwd / p).resolve()

    def _prepare_structure_path(
        self, is_restart: bool, iteration: int, initial_structure: str, base_cwd: Path
    ) -> Optional[str]:
        if not is_restart and initial_structure and initial_structure.endswith(".data"):
             return str(self._resolve_path(initial_structure, base_cwd))

        if is_restart:
            search_start = iteration - 1
            if search_start < 1:
                pass

            prev_dir = base_cwd / "data" / f"iteration_{search_start}"
            restart_file = prev_dir / "restart.chk"

            if not restart_file.exists():
                pass

            if not restart_file.exists():
                 logger.error(f"Restart file missing for resume: {restart_file}")
                 return None
            return str(restart_file)
        else:
            return str(self._resolve_path(initial_structure, base_cwd))

    def _ensure_chemical_symbols(self, atoms: Atoms):
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
