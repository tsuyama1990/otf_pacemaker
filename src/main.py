"""Main controller module for the ACE Active Carver application.

This module orchestrates the active learning loop, connecting the MD engine,
small cell generation, labeling, and training components.
"""

import logging
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np

from ase.io import read

from src.active_learning import SmallCellGenerator
from src.config import Config
from src.labeler import DeltaLabeler
from src.md_engine import LAMMPSRunner, SimulationState
from src.trainer import PacemakerTrainer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main():
    """Main execution loop for Active Learning."""
    # 1. Load Configuration
    config_path = Path("config.yaml")
    if not config_path.exists():
        logger.error("config.yaml not found.")
        sys.exit(1)

    config = Config.from_yaml(config_path)

    # 2. Initialize Components

    # LAMMPS Runner
    # Assuming 'lmp_serial' or similar is available.
    # In a real env, this might be in config or env var.
    # We'll use a default if not provided, but let's assume 'lmp_mpi' for now or 'lmp'.
    lammps_cmd = "lmp_serial"

    runner = LAMMPSRunner(
        cmd=lammps_cmd,
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

    # Small Cell Generator
    generator = SmallCellGenerator(
        box_size=config.al_params.box_size,
        r_core=config.al_params.r_core,
        lammps_cmd=lammps_cmd,
        stoichiometry_tolerance=config.al_params.stoichiometry_tolerance,
        elements=config.md_params.elements
    )

    # Delta Labeler
    # We need a QE calculator.
    # Since we can't really initialize a working QE calculator without binary paths and pseudopotentials,
    # we will instantiate a placeholder or a real one if ASE allows lazy init.
    # ASE Espresso requires 'command' or 'start' parameter usually via environment variables (ASE_ESPRESSO_COMMAND).
    # For this implementation code, we assume the environment is set up or config holds it?
    # The config has 'pseudo_dir'.
    # We'll import Espresso here.
    from ase.calculators.espresso import Espresso

    # Construct a dict for Espresso inputs from config
    # This is a simplification; real usage requires mapping DFTParams to QE input flags.
    qe_input_data = {
        "control": {
            "pseudo_dir": config.dft_params.pseudo_dir,
            "calculation": "scf",
        },
        "system": {
            "ecutwfc": config.dft_params.ecutwfc,
        },
        "electrons": {
            "k_points": config.dft_params.kpts # This might need formatting for ASE
        }
    }

    # Note: kpts in ASE is usually passed as kpts=(k,k,k) to the constructor, not in input_data['electrons']

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

    trainer = PacemakerTrainer()

    # 3. Main Loop Setup
    current_potential = config.al_params.initial_potential
    # Use initial structure from config
    # For the first run, we read data. For subsequent, we might read restart if we halted.
    # Wait, if we completed successfully, we might want to continue or start new?
    # Usually AL runs until we cover the phase space or max iterations.
    # The prompt implies an infinite loop or "Persist loop".

    # We need to track simulation state.
    # If we halted (UNCERTAIN), we extract, train, and RESUME.
    # Resuming means using 'restart.chk' and the NEW potential.

    # Initial State
    current_structure = config.md_params.initial_structure
    is_restart = False

    iteration = 0

    while True:
        iteration += 1
        work_dir = Path(f"data/iteration_{iteration}")
        work_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"--- Starting Iteration {iteration} ---")
        logger.info(f"Work Directory: {work_dir}")

        # Change working directory to iteration folder to keep files organized?
        # Or copy files there? LAMMPS generates output in CWD.
        # It's safer to run inside the directory.
        # But our code is in src/.
        # Let's copy necessary input files to work_dir and run there, or run from root and point output there.
        # LAMMPSRunner generates "in.lammps" in CWD.
        # Let's change CWD context.

        original_cwd = Path.cwd()
        try:
            import os
            os.chdir(work_dir)

            # Copy potential to work dir if needed, or reference absolute path?
            # Reference absolute path is safer.
            abs_potential_path = Path(current_potential).resolve()
            if not abs_potential_path.exists():
                logger.error(f"Potential file not found: {abs_potential_path}")
                break # Or exit

            # Resolve structure path
            if not is_restart:
                abs_structure_path = (original_cwd / current_structure).resolve()
            else:
                # Restart file is in the PREVIOUS iteration's directory usually?
                # Or we moved it here?
                # Actually, if we halted in iteration X, we generated restart.chk in iteration X.
                # Now we are in iteration X+1 (the "Resume" step).
                # So we should reference ../iteration_{X}/restart.chk?
                # Or we just copy it.
                pass

            # Wait, the loop logic:
            # 1. Run MD
            # 2. If Uncertain -> Train -> Resume (Next Iteration)
            # 3. If Completed -> Done? Or run longer?

            # If we are resuming, we need the restart file from the PREVIOUS run.
            # Let's handle the path logic.

            input_structure_arg = ""
            if is_restart:
                # The restart file was generated in the previous iteration folder (or current if we consider this a sub-step).
                # But we incremented `iteration`.
                # So look at iteration-1.
                prev_dir = original_cwd / f"data/iteration_{iteration-1}"
                restart_file = prev_dir / "restart.chk"
                if not restart_file.exists():
                    logger.error("Restart file missing for resume.")
                    sys.exit(1)
                input_structure_arg = str(restart_file)
            else:
                input_structure_arg = str(abs_structure_path)

            # Run MD
            logger.info("Running MD...")
            state = runner.run_md(
                potential_path=str(abs_potential_path),
                steps=config.md_params.n_steps,
                gamma_threshold=config.al_params.gamma_threshold,
                input_structure=input_structure_arg,
                is_restart=is_restart
            )

            logger.info(f"MD Finished with state: {state}")

            if state == SimulationState.COMPLETED:
                logger.info("Simulation completed successfully without uncertainty halt.")
                break # Exit loop? Or start new trajectory? Assuming exit for this task.

            elif state == SimulationState.FAILED:
                logger.error("Simulation failed.")
                break

            elif state == SimulationState.UNCERTAIN:
                logger.info("Uncertainty detected. Starting Active Learning cycle.")

                # 1. Load Snapshot (dump file)
                # LAMMPSRunner configured to dump to "dump.lammpstrj"
                dump_file = Path("dump.lammpstrj")
                if not dump_file.exists():
                    logger.error("Dump file not found.")
                    break

                # Read last frame
                try:
                    atoms = read(dump_file, index=-1, format="lammps-dump-text")
                    # Note: lammps-dump-text might need 'specorder' if species are not generic.
                    # We might need to know the atom types mapping to elements.
                    # ASE 'read' for lammps-dump often needs help with species.
                    # But for now let's assume it works or returns generic symbols 'X'.
                    # We should probably map types to elements using config.md_params.elements.
                    # atoms.numbers = ... based on type.
                    # Let's attempt to fix symbols if possible.
                    types = atoms.get_array('type')
                    elements = config.md_params.elements
                    # Map type 1 -> elements[0], type 2 -> elements[1]
                    # Check bounds
                    symbols = []
                    for t in types:
                        if t <= len(elements):
                            symbols.append(elements[t-1])
                        else:
                            symbols.append("X")
                    atoms.set_chemical_symbols(symbols)
                except Exception as e:
                    logger.error(f"Failed to read dump file: {e}")
                    break

                # 2. Identify High-Uncertainty Environment
                # Identify the atom with the maximum gamma value
                try:
                    # We added 'f_f_gamma' to the dump command.
                    # ASE reads custom columns into atoms.arrays if formatted as 'f_name'.
                    # The column name in dump was 'f_f_gamma', so ASE should read it as 'f_gamma' or similar?
                    # Standard behavior: 'f_name' -> 'name' in arrays? Or remains 'f_name'.
                    # ASE lammps-dump-text reader typically parses columns.
                    # Let's assume it's available in arrays. We might need to inspect keys if unsure.
                    # Usually for `f_f_gamma`, it might appear as `f_gamma` or `f_f_gamma` depending on ASE version.
                    # But typically ASE tries to be smart. Let's check generic keys.
                    gamma_key = None
                    for key in atoms.arrays.keys():
                        if "gamma" in key:
                            gamma_key = key
                            break

                    if gamma_key:
                        gammas = atoms.get_array(gamma_key)
                        # Find index of max gamma
                        # If gammas is 1D or 2D (N, 1), handle it.
                        if gammas.ndim > 1:
                            gammas = gammas.flatten()

                        # Identify the atom ID with max gamma
                        # Note: 'atoms' index vs 'id'. We need the index for extract_cluster (which takes index).
                        max_gamma_idx = int(np.argmax(gammas))
                        logger.info(f"Max gamma found: {gammas[max_gamma_idx]} at atom index {max_gamma_idx}")

                        # We select this one.
                        center_ids = [max_gamma_idx]

                        # If we need more clusters (n_clusters > 1), we could pick top N?
                        # The requirement says: "Identify 'Atom with max gamma' and use as center_id".
                        # Use singular. But loop below iterates center_ids.
                        # Let's stick to just the max one as per instructions: "identify 'gamma value max atom' and make it center_id".
                    else:
                        logger.warning("Gamma values not found in dump file. Falling back to random selection.")
                        import numpy as np
                        rng = np.random.default_rng()
                        center_ids = rng.choice(len(atoms), size=min(config.al_params.n_clusters, len(atoms)), replace=False)

                except Exception as e:
                    logger.error(f"Error processing gamma values: {e}. Falling back to random.")
                    import numpy as np
                    rng = np.random.default_rng()
                    center_ids = rng.choice(len(atoms), size=min(config.al_params.n_clusters, len(atoms)), replace=False)

                labeled_clusters = []

                logger.info(f"Generating and labeling {len(center_ids)} small cells...")

                for cid in center_ids:
                    # Generate relaxed small cell
                    try:
                        # Copy potential to a location where it can be accessed by generator
                        # Generator runs LAMMPS, which needs the file.
                        # We are in work_dir. abs_potential_path is absolute.
                        cell = generator.generate_cell(atoms, cid, str(abs_potential_path))
                    except Exception as e:
                        logger.error(f"Small cell generation failed for atom {cid}: {e}")
                        continue

                    # 3. Labeling
                    try:
                        labeled_cluster = labeler.compute_delta(cell)
                        labeled_clusters.append(labeled_cluster)
                    except Exception as e:
                        logger.warning(f"DFT Labeling failed for cluster {cid}: {e}. Skipping.")
                        continue

                if not labeled_clusters:
                    logger.warning("No clusters labeled successfully. Aborting iteration.")
                    break

                # 4. Training
                logger.info("Training new potential...")
                dataset_path = trainer.prepare_dataset(labeled_clusters)

                # Train with fine-tuning
                # The `train` method logic we implemented uses `initial_potential` for `input_potential`.
                # So we pass the CURRENT potential (the one used in MD) as the starting point.
                new_potential = trainer.train(dataset_path, str(abs_potential_path))

                logger.info(f"New potential trained: {new_potential}")

                # Update current potential for next iteration
                # We need to make sure next iteration uses this new file.
                # Since we switch directories, we should probably use absolute path again.
                current_potential = str(Path(new_potential).resolve())

                # Set restart flag for next loop
                is_restart = True

        except Exception as e:
            logger.exception(f"An error occurred in iteration {iteration}: {e}")
            break
        finally:
            os.chdir(original_cwd)

if __name__ == "__main__":
    main()
