"""Seed Generation Script.

This script orchestrates the Phase 1: Seed Generation pipeline.
It generates random structures, filters them using a foundation model,
samples diverse structures using ACE descriptors, labels them with DFT,
and trains an initial seed potential.
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path
from ase.calculators.espresso import Espresso

# Add src to path if running as script
sys.path.append(str(Path(__file__).parent.parent))

from src.config import Config
from src.labeler import DeltaLabeler, ShiftedLennardJones
from src.trainer import PacemakerTrainer
from src.generation.candidate import RandomStructureGenerator
from src.generation.filter import MACEFilter
from src.generation.sampler import DirectSampler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Main execution function for seed generation."""
    parser = argparse.ArgumentParser(description="ACE Active Carver - Seed Generation")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--output-dir", type=str, default="data/seed", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load Config
    try:
        config = Config.from_yaml(args.config)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)

    # 1. Candidate Generation
    logger.info("Phase 1.1: Candidate Generation (PyXtal)")
    generator = RandomStructureGenerator(
        elements=config.md_params.elements,
        max_atoms=8  # As per requirements
    )

    # Target roughly 1000-10000 candidates. Let's go for 2000 to be safe for filtering.
    n_candidates = 2000
    candidates = generator.generate(n_structures=n_candidates)
    logger.info(f"Generated {len(candidates)} candidate structures.")

    if not candidates:
        logger.error("No candidates generated. Exiting.")
        sys.exit(1)

    # 2. Foundation Model Filtering
    logger.info("Phase 1.2: Foundation Model Filtering (MACE)")
    # Using default medium model, force cutoff e.g. 10.0 eV/A to be generous but safe
    mace_filter = MACEFilter(model_size="medium", force_cutoff=20.0)
    filtered_structures = mace_filter.filter(candidates)

    if not filtered_structures:
        logger.error("All candidates filtered out. Exiting.")
        sys.exit(1)

    # 3. DIRECT Sampling
    logger.info("Phase 1.3: DIRECT Sampling (ACE + BIRCH)")
    # Target 200 structures for seed
    n_seed = 200
    sampler = DirectSampler(n_clusters=n_seed)
    selected_structures = sampler.sample(filtered_structures)

    # Save selected structures for reference
    from ase.io import write
    write(output_dir / "seed_structures.xyz", selected_structures)

    # 4. Labeling
    logger.info("Phase 1.4: DFT Labeling")

    # Setup Calculators
    # DFT
    dft_params = config.dft_params
    # We need to construct the command properly.
    # Espresso calculator expects 'command' or 'profile'.
    # We will use the command string.

    # NOTE: We assume dft_params.command is full command like "mpirun ... pw.x ..."
    # Espresso calculator usually takes 'command' env var or argument.
    # But ASE Espresso calculator arguments are for input file mostly.
    # The command to run is usually set via `command=...` in constructor or ASE_ESPRESSO_COMMAND.

    # We must check how DeltaLabeler expects 'reference_calculator'.
    # It expects an instantiated Calculator.

    # Configuring Espresso
    # We map dft_params to Espresso args.
    # Note: We need pseudo_dir and pseudopotentials.
    # config.dft_params has pseudo_dir. We need to infer pseudopotentials from elements.
    # This assumes standard naming like "Al.pbe-n-kjpaw_psl.1.0.0.UPF".
    # Since we don't know exact filenames, we might need a mapping or rely on user env.
    # For now, we pass what we can.

    pseudopotentials = {el: f"{el}.UPF" for el in config.md_params.elements}

    # Need to handle kpts. config has list/tuple
    kpts = config.dft_params.kpts

    dft_calc = Espresso(
        command=dft_params.command,
        pseudopotentials=pseudopotentials,
        pseudo_dir=dft_params.pseudo_dir,
        tstress=True,
        tprnfor=True,
        kpts=kpts,
        ecutwfc=dft_params.ecutwfc,
        input_data={
            'control': {
                'calculation': 'scf',
                'disk_io': 'none', # Avoid large files
            },
            'system': {
                'ecutwfc': dft_params.ecutwfc,
                'occupations': 'smearing',
                'smearing': 'mv',
                'degauss': 0.02,
            },
            'electrons': {
                'mixing_beta': 0.7,
            }
        }
    )

    # LJ Baseline
    lj_params = config.lj_params
    lj_calc = ShiftedLennardJones(
        epsilon=lj_params.epsilon,
        sigma=lj_params.sigma,
        rc=lj_params.cutoff
    )

    labeler = DeltaLabeler(reference_calculator=dft_calc, baseline_calculator=lj_calc)

    labeled_structures = []
    for i, atoms in enumerate(selected_structures):
        logger.info(f"Labeling structure {i+1}/{len(selected_structures)}")
        labeled = labeler.label(atoms)
        if labeled is not None:
            labeled_structures.append(labeled)

    logger.info(f"Successfully labeled {len(labeled_structures)} structures.")
    if not labeled_structures:
        logger.error("No structures successfully labeled. Exiting.")
        sys.exit(1)

    # 5. Initial Training
    logger.info("Phase 1.5: Initial Training")
    trainer = PacemakerTrainer()

    # Prepare dataset
    dataset_path = trainer.prepare_dataset(labeled_structures)
    # Move dataset to output dir
    final_dataset_path = output_dir / "seed_data.pckl.gzip"
    shutil.move(dataset_path, final_dataset_path)

    # Train
    # We run training in the output dir to keep files clean?
    # Or just copy result. Pacemaker creates files in current dir.
    # It's safer to run in a context where we can move files.

    try:
        potential_path = trainer.train(str(final_dataset_path), initial_potential=None)
        final_potential_path = output_dir / "seed_potential.yace"
        shutil.move(potential_path, final_potential_path)
        logger.info(f"Seed potential generated at: {final_potential_path}")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)

    logger.info("Seed generation completed successfully.")

if __name__ == "__main__":
    main()
