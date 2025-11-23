"""Training module for Pacemaker potentials.

This module handles the preparation of datasets and training of ACE potentials.
"""

import pandas as pd
import subprocess
import yaml
from pathlib import Path
from typing import List, Optional
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

    def train(self, dataset_path: str, initial_potential: Optional[str] = None) -> str:
        """Train the potential using Pacemaker.

        Args:
            dataset_path: Path to the training dataset file.
            initial_potential: Path to the initial potential file to start from.
                               If None, trains from scratch.

        Returns:
            str: The path to the newly trained potential file.
        """

        # Determine elements from dataset
        # We need to read the dataset to know elements if we are training from scratch
        # OR we assume the caller knows?
        # Actually, for generating a fresh potential config, we need the element list.
        # Pacemaker can infer it, but if we generate the config manually we need it.
        # However, we can just omit specific element details if pacemaker handles it,
        # OR we can peek at the dataset.
        # Let's peek at the dataset to be safe if we are creating a fresh config.

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
                print(f"Warning: Could not read elements from dataset: {e}")
                # Fallback or let pacemaker fail

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
            # Training from scratch: need to define potential structure
            # We inject a default configuration similar to what DirectSampler uses or what is standard
            # or just minimal and let pacemaker fill defaults.
            # Pacemaker usually needs 'embeddings' and 'bonds' for a new potential.

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
            print(f"Pacemaker training failed: {e}")
            raise e

        output_potential = "output_potential.yace"

        if not Path(output_potential).exists():
            # If we started from scratch, output might be named differently?
            # usually 'output_potential.yace' is the default output name in pacemaker
            # UNLESS specified otherwise.
            # Let's check for any .yace that isn't initial_potential
            potentials = list(Path(".").glob("*.yace"))
            candidates = []
            for p in potentials:
                if initial_potential and str(p) == initial_potential:
                    continue
                candidates.append(str(p))

            if candidates:
                # Return the most recent one?
                candidates.sort(key=lambda x: Path(x).stat().st_mtime, reverse=True)
                return candidates[0]

            raise FileNotFoundError("No output potential found after training.")

        return output_potential
