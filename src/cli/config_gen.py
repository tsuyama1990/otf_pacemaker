"""Interactive Configuration Generator.

This CLI tool generates `config.yaml` and `meta_config.yaml` based on minimal user input,
enforcing best practices and enabling reproduceable experiments.
"""

import argparse
import sys
import yaml
import shutil
import logging
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any

from ase.data import atomic_numbers, covalent_radii
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def get_user_input(prompt: str, default: Any = None) -> str:
    """Helper to get user input with a default value."""
    if default:
        full_prompt = f"{prompt} [{default}]: "
    else:
        full_prompt = f"{prompt}: "

    val = input(full_prompt).strip()
    if not val and default is not None:
        return str(default)
    return val

def generate_default_lj_params(elements: List[str]) -> Dict[str, float]:
    """Generates robust default Lennard-Jones parameters (copied logic to avoid circular deps before refactor)."""
    if not elements:
        return {"epsilon": 1.0, "sigma": 2.0, "cutoff": 5.0}

    try:
        radii = []
        for el in elements:
            z = atomic_numbers.get(el)
            if z is None:
                raise ValueError(f"Unknown element symbol: {el}")
            radii.append(covalent_radii[z])

        avg_radius = np.mean(radii)
        # r_min = 2^(1/6) * sigma => sigma = r_min / 1.122
        # approximate r_min as 2 * avg_radius (bond length)
        sigma = (2.0 * avg_radius) * 0.8909

        return {
            "epsilon": 1.0,
            "sigma": float(round(sigma, 3)),
            "cutoff": float(round(2.5 * sigma, 3)),
            "shift_energy": True
        }
    except Exception as e:
        logger.warning(f"Could not auto-generate LJ params ({e}). Using safe defaults.")
        return {"epsilon": 1.0, "sigma": 2.0, "cutoff": 5.0, "shift_energy": True}

def generate_meta_config_content() -> Dict[str, Any]:
    """Generates the content for meta_config.yaml."""
    return {
        "dft": {
            "command": "pw.x -nk 4",
            "pseudo_dir": "./pseudos",
            "sssp_json_path": "./sssp.json"
        },
        "lammps": {
            "command": "lmp_serial"
        }
    }

def generate_experiment_config_content(elements: List[str], cutoff: float) -> Dict[str, Any]:
    """Generates the content for config.yaml."""

    lj_params = generate_default_lj_params(elements)

    return {
        "experiment": {
            "name": "default_experiment",
            "output_dir": "output/test_run_01"
        },
        "md_params": {
            "timestep": 0.001,
            "temperature": 1000.0,
            "pressure": 0.0,
            "n_steps": 5000,
            "elements": elements,
            "initial_structure": "data/initial.xyz",
            "masses": {el: 1.0 for el in elements}, # User should update this
            "restart_freq": 1000,
            "dump_freq": 1000
        },
        "al_params": {
            "gamma_threshold": 0.2,
            "n_clusters": 5,
            "r_core": 4.5,
            "box_size": 15.0,
            "initial_potential": "data/initial_potential.yace",
            "potential_yaml_path": "data/potential.yaml",
            "initial_dataset_path": "data/initial_dataset.pckl",
            "stoichiometry_tolerance": 0.1,
            "min_bond_distance": 1.8,
            "num_parallel_labeling": 4,
            "query_strategy": "uncertainty",
            "sampling_strategy": "composite",
            "outlier_energy_max": 100.0
        },
        "kmc_params": {
            "active": True,
            "temperature": 800.0,
            "n_searches": 20,
            "active_region_mode": "surface_and_species",
            "active_species": elements
        },
        "ace_model": {
            "pacemaker_config": {
                "cutoff": cutoff,
                "b_basis": {
                    "max_deg": 3,
                    "r_cut": cutoff,
                    "d_cut": 4.0
                },
                "fit": {
                    "loss": {
                         "kappa": 0.3,
                         "L1_coeffs": 1e-8,
                         "L2_coeffs": 1e-8
                    },
                    "optimizer": "BFGS"
                },
                "backend": {
                    "evaluator": "tensorpot"
                }
            },
            "initial_potentials": []
        },
        "dft_params": {
             "kpoint_density": 40.0,
             "auto_physics": True
        },
        "lj_params": lj_params,
        "training_params": {
            "replay_ratio": 1.0,
            "global_dataset_path": "data/global_dataset.pckl"
        },
        "generation": {
             "pre_optimization": {
                 "enabled": False
             },
             "scenarios": []
        }
    }

def main():
    parser = argparse.ArgumentParser(description="Interactive Config Generator")
    parser.add_argument("--init", action="store_true", help="Initialize configuration files.")
    parser.add_argument("--force", action="store_true", help="Force overwrite existing meta_config.yaml.")

    args = parser.parse_args()

    if not args.init:
        parser.print_help()
        sys.exit(0)

    print("=== ACE Active Carver Configuration Generator ===")

    # 1. User Input
    elements_str = get_user_input("Target Elements (comma-separated)", "Al, Mg")
    elements = [e.strip() for e in elements_str.split(",")]

    cutoff_str = get_user_input("ACE Cutoff Radius (Angstrom)", "5.0")
    try:
        cutoff = float(cutoff_str)
    except ValueError:
        logger.error("Invalid cutoff value. Using default 5.0")
        cutoff = 5.0

    # 2. Meta Config Generation
    meta_path = Path("meta_config.yaml")
    if meta_path.exists() and not args.force:
        logger.warning(f"'{meta_path}' already exists. Skipping to prevent overwrite. Use --force to overwrite.")
    else:
        meta_content = generate_meta_config_content()
        with open(meta_path, "w") as f:
            yaml.dump(meta_content, f, sort_keys=False)
        logger.info(f"Generated {meta_path}")

    # 3. Experiment Config Generation
    config_path = Path("config.yaml")
    if config_path.exists():
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S") if 'pd' in locals() else "backup"
        backup_path = config_path.with_suffix(f".yaml.{timestamp}")
        shutil.move(config_path, backup_path)
        logger.info(f"Existing '{config_path}' backed up to '{backup_path}'")

    config_content = generate_experiment_config_content(elements, cutoff)
    with open(config_path, "w") as f:
        yaml.dump(config_content, f, sort_keys=False)
    logger.info(f"Generated {config_path}")

    logger.info("\nDone! Please review generated files and update paths/masses as needed.")

if __name__ == "__main__":
    main()
