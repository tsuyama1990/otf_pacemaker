"""Pacemaker Training Strategy."""

import pandas as pd
import subprocess
import yaml
import logging
from pathlib import Path
from typing import List, Optional
from ase import Atoms

from src.core.interfaces import Trainer

logger = logging.getLogger(__name__)

class PacemakerTrainer(Trainer):
    """Manages the training process for ACE potentials."""

    def prepare_dataset(self, structures: List[Atoms]) -> str:
        """Convert a list of labeled Atoms objects into a training dataset.

        Args:
            structures: A list of labeled ASE Atoms objects.

        Returns:
            str: The file path to the created dataset.
        """
        df = pd.DataFrame({"ase_atoms": structures})
        output_path = "training_data.pckl.gzip"
        df.to_pickle(output_path, compression="gzip")
        return output_path

    def update_active_set(self, dataset_path: str, potential_yaml_path: str) -> str:
        """Update the active set using pace_activeset.

        Args:
            dataset_path: Path to the full training dataset.
            potential_yaml_path: Path to the potential basis set definition.

        Returns:
            str: The path to the updated active set (.asi) file.
        """
        output_asi = "potential.asi"

        # pace_activeset -d dataset_path -f potential_yaml_path -o output_asi
        # The prompt specifies that dataset_path should include the full training data.
        # It is the caller's responsibility to ensure dataset_path points to the correct file.

        cmd = [
            "pace_activeset",
            "-d", dataset_path,
            "-f", potential_yaml_path,
            "-o", output_asi
        ]

        logger.info(f"Updating Active Set: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"pace_activeset failed: {e.stderr}")
            raise RuntimeError(f"pace_activeset failed: {e.stderr}")

        if not Path(output_asi).exists():
             raise FileNotFoundError("pace_activeset did not generate an output file.")

        return str(Path(output_asi).resolve())

    def train(self, dataset_path: str, initial_potential: Optional[str] = None,
              potential_yaml_path: Optional[str] = None, asi_path: Optional[str] = None) -> str:
        """Train the potential using Pacemaker.

        Args:
            dataset_path: Path to the training dataset file.
            initial_potential: Path to the initial potential file to start from.
                               If None, trains from scratch.
            potential_yaml_path: Path to potential.yaml (required for updating active set).
            asi_path: Path to the current Active Set Index file.

        Returns:
            str: The path to the newly trained potential file.
        """

        # 1. Update Active Set if possible
        # We run pace_activeset before training to ensure the active set reflects the latest data.
        current_asi = asi_path
        if potential_yaml_path:
             try:
                 current_asi = self.update_active_set(dataset_path, potential_yaml_path)
                 logger.info(f"Active Set updated: {current_asi}")
             except Exception as e:
                 logger.error(f"Failed to update active set: {e}. Proceeding with existing ASI if available.")

        # Determine elements from dataset
        elements = []
        if initial_potential is None:
            try:
                df = pd.read_pickle(dataset_path)
                # Collect unique elements
                all_symbols = set()
                for atoms in df['ase_atoms']:
                    all_symbols.update(atoms.get_chemical_symbols())
                elements = sorted(list(all_symbols))
            except Exception as e:
                logger.warning(f"Could not read elements from dataset: {e}")

        config = {
            "cutoff": 7.0,
            "data": {
                "filename": dataset_path,
                "test_size": 0.1,
            },
            "potential": {
                "delta": True,
            },
            "fitting": {
                "fit_cycles": 1,
                "max_time": 3600,
            },
            "backend": {
                "evaluator": "tensorpot",
            }
        }

        if initial_potential:
            config["fitting"]["input_potential"] = initial_potential
        else:
            config["potential"]["elements"] = elements
            config["potential"]["bonds"] = {
                "N": 3,
                "max_deg": 10,
                "r0": 1.5,
                "rad_base": "Chebyshev",
                "rad_parameters": [7.0], # r_cut
            }
            config["potential"]["embeddings"] = {
                 el: {
                     "npot": "FinnisSinclair",
                     "fs_parameters": [1, 1, 1, 0.5],
                     "ndensity": 1,
                 } for el in elements
            }

        input_yaml_path = "input.yaml"
        with open(input_yaml_path, "w") as f:
            yaml.dump(config, f)

        try:
            subprocess.run(["pacemaker", input_yaml_path], check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Pacemaker training failed: {e}")
            raise e

        output_potential = "output_potential.yace"

        if not Path(output_potential).exists():
            potentials = list(Path(".").glob("*.yace"))
            candidates = []
            for p in potentials:
                if initial_potential and str(p) == str(Path(initial_potential).name):
                    continue
                candidates.append(str(p))

            if candidates:
                candidates.sort(key=lambda x: Path(x).stat().st_mtime, reverse=True)
                return candidates[0]

            raise FileNotFoundError("No output potential found after training.")

        return output_potential
