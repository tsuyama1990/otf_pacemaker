"""Training module for Pacemaker potentials.

This module handles the preparation of datasets and training of ACE potentials
using the Pacemaker library.
"""

import pandas as pd
import subprocess
import yaml
from pathlib import Path
from ase import Atoms


class PacemakerTrainer:
    """Manages the training process for ACE potentials."""

    def prepare_dataset(self, atoms_list: list[Atoms]) -> str:
        """Convert a list of labeled Atoms objects into a training dataset.

        The method should serialize the data into a format suitable for Pacemaker
        (e.g., a pandas DataFrame saved as a pickle file).

        Args:
            atoms_list: A list of labeled ASE Atoms objects (containing delta energy/forces).

        Returns:
            str: The file path to the created dataset (e.g., 'data.pckl.gzip').
        """
        df = pd.DataFrame({"ase_atoms": atoms_list})
        output_path = "training_data.pckl.gzip"
        df.to_pickle(output_path, compression="gzip")
        return output_path

    def train(self, dataset_path: str, initial_potential: str) -> str:
        """Train the potential using Pacemaker.

        This method executes the training process, potentially using `subprocess`
        to call Pacemaker CLI tools.

        Args:
            dataset_path: Path to the training dataset file.
            initial_potential: Path to the initial potential file (or configuration) to start from.

        Returns:
            str: The path to the newly trained potential file.
        """
        # Define input.yaml content structure
        # Based on typical Pacemaker usage.
        # We need to reference the dataset and the initial potential for fine-tuning.

        config = {
            "cutoff": 7.0, # Standard fallback, should ideally match potential but fine-tuning handles it.
            "data": {
                "filename": dataset_path,
                "test_size": 0.1, # Reserve some for testing
            },
            "potential": {
                "delta": True, # We are training a delta model
            },
            "fitting": {
                "input_potential": initial_potential,
                "fit_cycles": 1, # Simplification for the loop
                "max_time": 3600, # 1 hour max
            },
            "backend": {
                "evaluator": "tensorpot", # or specific backend if known
            }
        }

        # Write input.yaml
        input_yaml_path = "input.yaml"
        with open(input_yaml_path, "w") as f:
            yaml.dump(config, f)

        # Run Pacemaker
        # Command: pacemaker input.yaml
        # We check=True to raise exception on failure
        try:
            subprocess.run(["pacemaker", input_yaml_path], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Pacemaker training failed: {e}")
            raise e

        # After training, Pacemaker usually produces a potential file.
        # The name depends on the input potential or config.
        # Typically "output_potential.yace" or similar.
        # For now, let's assume a standard output name or find the newest .yace file.
        # However, with "input_potential", it might overwrite or create "new_potential.yace".
        # Let's assume it creates "output_potential.yace" based on default behavior if not specified.
        # Or we can check the directory for the result.

        output_potential = "output_potential.yace"
        # If pacemaker creates a timestamped folder or file, we might need to be more robust.
        # But for this exercise, we assume standard output.

        if not Path(output_potential).exists():
            # Fallback: look for any .yace created recently?
            # Or maybe the user didn't specify the output name behavior.
            # Let's try to find *any* yace file that isn't the input one.
            potentials = list(Path(".").glob("*.yace"))
            for p in potentials:
                if str(p) != initial_potential:
                    return str(p)
            raise FileNotFoundError("No output potential found after training.")

        return output_potential
