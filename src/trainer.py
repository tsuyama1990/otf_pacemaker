"""Training module for Pacemaker potentials.

This module handles the preparation of datasets and training of ACE potentials.
"""

import pandas as pd
import subprocess
import yaml
from pathlib import Path
from typing import List
from ase import Atoms

from src.interfaces import Trainer


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

    def train(self, dataset_path: str, initial_potential: str) -> str:
        """Train the potential using Pacemaker.

        Args:
            dataset_path: Path to the training dataset file.
            initial_potential: Path to the initial potential file to start from.

        Returns:
            str: The path to the newly trained potential file.
        """
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
                "input_potential": initial_potential,
                "fit_cycles": 1,
                "max_time": 3600,
            },
            "backend": {
                "evaluator": "tensorpot",
            }
        }

        input_yaml_path = "input.yaml"
        with open(input_yaml_path, "w") as f:
            yaml.dump(config, f)

        try:
            subprocess.run(["pacemaker", input_yaml_path], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Pacemaker training failed: {e}")
            raise e

        output_potential = "output_potential.yace"

        if not Path(output_potential).exists():
            potentials = list(Path(".").glob("*.yace"))
            for p in potentials:
                if str(p) != initial_potential:
                    return str(p)
            raise FileNotFoundError("No output potential found after training.")

        return output_potential
