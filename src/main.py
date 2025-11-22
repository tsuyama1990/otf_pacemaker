"""Main entry point for the ACE Active Carver application.

This module sets up the configuration and dependencies, then orchestrates the active learning loop.
"""

import logging
import sys
import os
import shutil
import numpy as np
from typing import Optional
from pathlib import Path
from ase.io import read
from ase.calculators.espresso import Espresso

from src.active_learning import MaxGammaSampler, SmallCellGenerator
from src.config import Config
from src.enums import SimulationState
from src.labeler import DeltaLabeler
from src.md_engine import LAMMPSRunner
from src.trainer import PacemakerTrainer

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
        cmd="lmp_serial",  # In production, this might come from env or config
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
    # Note: config.al_params.stoichiometry_tolerance is used in the original code,
    # but here we use it to fulfill the "stoichiometric_ratio" argument or strict checking if needed.
    # Assuming config.al_params has access to expected stoichiometry or we pass a dummy/derived dict.
    # For now, we pass a derived dict from elements assuming equal ratio or just empty if not strictly enforced.
    # A better approach is to rely on the initial structure, but SmallCellGenerator wants it in init.
    # We'll assume equal weights for now or pass empty dict if check isn't critical inside generator.
    stoich_ratio = {el: 1.0 for el in config.md_params.elements}

    generator = SmallCellGenerator(
        r_core=config.al_params.r_core,
        box_size=config.al_params.box_size,
        stoichiometric_ratio=stoich_ratio,
        lammps_cmd="lmp_serial"
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

    # 3. Active Learning Loop
    run_active_learning_loop(config, runner, sampler, generator, labeler, trainer)


def run_active_learning_loop(config, md_engine, sampler, generator, labeler, trainer):
    """Executes the active learning loop explicitly."""
    current_potential = config.al_params.initial_potential
    current_structure = config.md_params.initial_structure
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
            abs_potential_path = _resolve_path(current_potential, original_cwd)

            if not abs_potential_path.exists():
                logger.error(f"Potential file not found: {abs_potential_path}")
                break

            # Prepare Structure Path
            input_structure_arg = _prepare_structure_path(
                is_restart, iteration, current_structure, original_cwd
            )
            if not input_structure_arg:
                break

            # 1. Run MD
            logger.info("Running MD...")
            state = md_engine.run(
                potential_path=str(abs_potential_path),
                steps=config.md_params.n_steps,
                gamma_threshold=config.al_params.gamma_threshold,
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
                # ase read handles lammpstrj
                atoms = read(dump_file, index=-1, format="lammps-dump-text")

                # Ensure species mapping
                if config.md_params.elements:
                     # We might need to map types to symbols if dump doesn't have them
                     # Check if 'type' is present and symbols are 'X'
                     if 'type' in atoms.arrays:
                         types = atoms.get_array('type')
                         elements = config.md_params.elements
                         symbols = []
                         for t in types:
                             # type is 1-based
                             if 1 <= t <= len(elements):
                                 symbols.append(elements[t-1])
                             else:
                                 symbols.append("X")
                         atoms.set_chemical_symbols(symbols)

                # Sample
                center_ids = sampler.sample(atoms, config.al_params.n_clusters)

                labeled_clusters = []
                logger.info(f"Generating and labeling {len(center_ids)} small cells...")

                for cid in center_ids:
                    try:
                        # Generate Small Cell
                        cell = generator.generate_cell(atoms, cid, str(abs_potential_path))

                        # Label
                        labeled_cluster = labeler.label(cell)
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
                dataset_path = trainer.prepare_dataset(labeled_clusters)
                new_potential = trainer.train(dataset_path, str(abs_potential_path))

                current_potential = str(Path(new_potential).resolve())
                is_restart = True
                logger.info(f"New potential trained: {current_potential}")

        except Exception as e:
            logger.exception(f"An error occurred in iteration {iteration}: {e}")
            break
        finally:
            os.chdir(original_cwd)


def _resolve_path(path_str: str, base_cwd: Path) -> Path:
    """Resolve a path to absolute, handling relative paths correctly."""
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (base_cwd / p).resolve()


def _prepare_structure_path(
    is_restart: bool, iteration: int, initial_structure: str, base_cwd: Path
) -> Optional[str]:
    """Determine the correct input structure path."""
    if is_restart:
        # Resume from the restart file of the PREVIOUS iteration (where it halted)
        # Actually, if we are in iteration N, we just ran MD in iteration N.
        # It halted. We trained. Now we want to resume.
        # The loop starts iteration N+1.
        # We want to read the restart file from iteration N.
        prev_dir = base_cwd / f"data/iteration_{iteration-1}"
        restart_file = prev_dir / "restart.chk"
        if not restart_file.exists():
            # Fallback or error?
            # If restart file is missing but we have dump, we could theoretically use dump,
            # but LAMMPS restart is binary and preserves velocity.
            # If missing, we might need to fail.
            logger.error(f"Restart file missing for resume: {restart_file}")
            return None
        return str(restart_file)
    else:
        return str(_resolve_path(initial_structure, base_cwd))


if __name__ == "__main__":
    main()
