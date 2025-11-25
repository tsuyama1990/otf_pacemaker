"""Seed Generation Module.

This module orchestrates the Phase 1: Seed Generation pipeline.
It generates random structures, filters them using a foundation model,
samples diverse structures using ACE descriptors, labels them with DFT,
and trains an initial seed potential.

Phase 4 Update:
Integrates Scenario-Driven Generation and MACE Pre-optimization.
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path
from typing import List
from ase import Atoms
from ase.calculators.espresso import Espresso, EspressoProfile

from src.core.config import Config
from src.labeling.strategies.delta_labeler import DeltaLabeler
from src.labeling.calculators.shifted_lj import ShiftedLennardJones
from src.training.strategies.pacemaker import PacemakerTrainer
from src.scenario_generation.candidate import RandomStructureGenerator
from src.scenario_generation.filter import MACEFilter
from src.scenario_generation.sampler import DirectSampler
from src.scenario_generation.scenarios import ScenarioFactory
from src.scenario_generation.optimizer import FoundationOptimizer
from src.utils.sssp_loader import (
    load_sssp_database,
    calculate_cutoffs,
    get_pseudopotentials_dict,
    calculate_kpoints,
    validate_pseudopotentials
)
from src.utils.atomic_energies import AtomicEnergyManager
from src.engines.dft.heuristics import PymatgenHeuristics

# Configure logging
logger = logging.getLogger(__name__)


class SeedGenerator:
    """Manages Phase 1: Seed Generation."""

    def __init__(self, config: Config):
        """Initialize the SeedGenerator.

        Args:
            config: Configuration object.
        """
        self.config = config
        self.output_dir = Path("data/seed")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _get_e0_dict(self):
        """Retrieve or calculate isolated atomic energies."""
        storage_path = self.output_dir / "potential.e0.yaml"
        manager = AtomicEnergyManager(storage_path)

        # Prepare DFT factory for single atoms
        sssp_db = load_sssp_database(self.config.dft_params.sssp_json_path)

        def dft_factory(element: str):
             # Re-use DFT logic but for Gamma point
             elements = [element] # Single element for this calc
             pseudo_dir_abs = str(Path(self.config.dft_params.pseudo_dir).resolve())
             pseudopotentials = get_pseudopotentials_dict(elements, sssp_db)
             ecutwfc, ecutrho = calculate_cutoffs(elements, sssp_db)

             profile = EspressoProfile(
                command=self.config.dft_params.command,
                pseudo_dir=pseudo_dir_abs
             )

             qe_input_data = {
                "control": {
                    "pseudo_dir": pseudo_dir_abs,
                    "calculation": "scf",
                    "disk_io": "none",
                },
                "system": {
                    "ecutwfc": ecutwfc,
                    "ecutrho": ecutrho,
                    "occupations": 'smearing',
                    'smearing': 'mv',
                    'degauss': 0.02,
                },
                "electrons": {
                    "mixing_beta": 0.7,
                }
             }

             # Check for magnetism
             atom_for_check = Atoms(element)
             heuristics = PymatgenHeuristics.get_recommended_params(atom_for_check)
             if heuristics["magnetism"]["nspin"] == 2:
                logger.info(f"Element {element} is magnetic. Enabling spin polarization for E0.")
                qe_input_data["system"]["nspin"] = 2
                moment = heuristics["magnetism"]["moments"].get(element, 0.0)
                qe_input_data["system"]["starting_magnetization"] = {element: moment}


             return Espresso(
                profile=profile,
                pseudopotentials=pseudopotentials,
                input_data=qe_input_data,
                kpts=(1, 1, 1),
                koffset=(1, 1, 1),
                pseudo_dir=pseudo_dir_abs
             )

        return manager.get_e0(self.config.md_params.elements, dft_factory)

    def run(self):
        """Execute the seed generation pipeline."""
        logger.info("Starting Phase 1: Seed Generation")

        # 0. Calculate E0 first
        logger.info("Phase 1.0: E0 Calculation")
        try:
            e0_dict = self._get_e0_dict()
            logger.info(f"Using E0: {e0_dict}")
        except Exception as e:
            logger.error(f"Failed to calculate E0: {e}")
            raise e

        # 1. Candidate Generation
        logger.info("Phase 1.1: Candidate Generation")

        # 1a. Random Generation
        logger.info("Phase 1.1a: Random Generation (PyXtal)")
        generator = RandomStructureGenerator(
            elements=self.config.md_params.elements,
            max_atoms=8
        )
        n_candidates = 2000
        random_candidates = generator.generate(n_structures=n_candidates)
        logger.info(f"Generated {len(random_candidates)} random candidate structures.")

        # 1b. Scenario-Driven Generation
        logger.info("Phase 1.1b: Scenario-Driven Generation")
        scenario_candidates: List[Atoms] = []
        gen_params = self.config.generation_params

        for scenario_conf in gen_params.scenarios:
            try:
                sc_gen = ScenarioFactory.create(scenario_conf)
                structures = sc_gen.generate()
                scenario_candidates.extend(structures)
                logger.info(f"Generated {len(structures)} structures from scenario: {scenario_conf.get('type')}")
            except Exception as e:
                logger.error(f"Failed to generate scenario {scenario_conf.get('type')}: {e}")

        # 2. Pre-optimization (MACE)
        pre_opt_params = gen_params.pre_optimization
        if pre_opt_params.enabled and scenario_candidates:
            logger.info("Phase 1.2a: MACE Pre-optimization for Scenario Structures")
            try:
                optimizer = FoundationOptimizer(
                    model=pre_opt_params.model,
                    device=pre_opt_params.device,
                    fmax=pre_opt_params.fmax,
                    steps=pre_opt_params.steps
                )
                relaxed_scenarios = optimizer.relax(scenario_candidates)
                logger.info(f"Relaxed {len(relaxed_scenarios)} scenario structures.")
                scenario_candidates = relaxed_scenarios
            except ImportError:
                logger.warning("MACE Pre-optimization skipped (mace-torch not found).")
            except Exception as e:
                logger.error(f"Pre-optimization failed: {e}")

        # Combine candidates
        logger.info("Phase 1.2b: Foundation Model Filtering (MACE)")
        all_candidates = random_candidates + scenario_candidates

        mace_filter = MACEFilter(
            model_size="medium", 
            force_cutoff=20.0,
            device=gen_params.device
        )
        filtered_structures = mace_filter.filter(all_candidates)

        if not filtered_structures:
            raise RuntimeError("All candidates filtered out.")

        logger.info(f"Total structures after filtering: {len(filtered_structures)}")

        # 3. DIRECT Sampling
        logger.info("Phase 1.3: DIRECT Sampling (ACE + BIRCH)")
        n_seed = 200
        if len(filtered_structures) < n_seed:
            logger.warning(f"Number of filtered structures ({len(filtered_structures)}) is less than n_seed ({n_seed}). Using all.")
            selected_structures = filtered_structures
        else:
            sampler = DirectSampler(n_clusters=n_seed)
            selected_structures = sampler.sample(filtered_structures)

        from ase.io import write
        write(self.output_dir / "seed_structures.xyz", selected_structures)

        # 4. Labeling
        logger.info("Phase 1.4: DFT Labeling")

        # Load SSSP database
        logger.info(f"Loading SSSP database from {self.config.dft_params.sssp_json_path}")
        sssp_db = load_sssp_database(self.config.dft_params.sssp_json_path)
        
        elements = self.config.md_params.elements
        pseudo_dir = Path(self.config.dft_params.pseudo_dir).resolve()
        validate_pseudopotentials(str(pseudo_dir), elements, sssp_db)
        pseudopotentials = get_pseudopotentials_dict(elements, sssp_db)
        ecutwfc, ecutrho = calculate_cutoffs(elements, sssp_db)
        logger.info(f"Using SSSP cutoffs: ecutwfc={ecutwfc} Ry, ecutrho={ecutrho} Ry")
        
        profile = EspressoProfile(
            command=self.config.dft_params.command,
            pseudo_dir=str(pseudo_dir)
        )

        dft_calc_template = {
            'profile': profile,
            'pseudopotentials': pseudopotentials,
            'pseudo_dir': str(pseudo_dir),
            'tstress': True,
            'tprnfor': True,
            'ecutwfc': ecutwfc,
            'ecutrho': ecutrho,
            'input_data': {
                'control': {
                    'calculation': 'scf',
                    'disk_io': 'none',
                },
                'system': {
                    'ecutwfc': ecutwfc,
                    'ecutrho': ecutrho,
                    'occupations': 'smearing',
                    'smearing': 'mv',
                    'degauss': 0.02,
                },
                'electrons': {
                    'mixing_beta': 0.7,
                }
            }
        }

        lj_calc = ShiftedLennardJones(
            epsilon=self.config.lj_params.epsilon,
            sigma=self.config.lj_params.sigma,
            rc=self.config.lj_params.cutoff,
            shift_energy=self.config.lj_params.shift_energy
        )

        initial_kpts = (3, 3, 3)
        dft_calc = Espresso(
            kpts=initial_kpts,
            koffset=(1, 1, 1),
            **dft_calc_template
        )

        labeler = DeltaLabeler(
            reference_calculator=dft_calc,
            baseline_calculator=lj_calc,
            e0_dict=e0_dict
        )

        labeled_structures = []
        for i, atoms in enumerate(selected_structures):
            logger.info(f"Labeling structure {i+1}/{len(selected_structures)}")
            
            kpts, kpts_shift = calculate_kpoints(
                atoms.cell,
                kpoint_density=self.config.dft_params.kpoint_density
            )
            logger.info(f"Structure {i+1}: k-points={kpts}, shift={kpts_shift}")
            
            dft_calc = Espresso(
                kpts=kpts,
                koffset=kpts_shift,
                **dft_calc_template
            )
            
            labeler.reference_calculator = dft_calc
            
            labeled = labeler.label(atoms)
            if labeled is not None:
                labeled_structures.append(labeled)

        logger.info(f"Successfully labeled {len(labeled_structures)} structures.")
        if not labeled_structures:
            raise RuntimeError("No structures successfully labeled.")

        # 5. Initial Training
        logger.info("Phase 1.5: Initial Training")
        trainer = PacemakerTrainer()

        dataset_path = trainer.prepare_dataset(labeled_structures)
        final_dataset_path = self.output_dir / "seed_data.pckl.gzip"
        shutil.move(dataset_path, final_dataset_path)

        try:
            potential_path = trainer.train(str(final_dataset_path), initial_potential=None)
            final_potential_path = self.output_dir / "seed_potential.yace"
            shutil.move(potential_path, final_potential_path)
            logger.info(f"Seed potential generated at: {final_potential_path}")
        except Exception as e:
            raise RuntimeError(f"Training failed: {e}")

        logger.info("Seed generation completed successfully.")


def main():
    """Main execution function when running as script."""
    parser = argparse.ArgumentParser(description="ACE Active Carver - Seed Generation")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    # Load Config
    try:
        config = Config.from_yaml(args.config)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)

    try:
        generator = SeedGenerator(config)
        generator.run()
    except Exception as e:
        logger.error(f"Seed generation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
