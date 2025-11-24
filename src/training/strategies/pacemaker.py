"""Pacemaker Training Strategy."""

import pandas as pd
import subprocess
import yaml
import logging
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Any
from ase import Atoms

from src.core.interfaces import Trainer
from src.core.config import TrainingParams

logger = logging.getLogger(__name__)

class PacemakerTrainer(Trainer):
    """Manages the training process for ACE potentials with Experience Replay."""

    def __init__(self, training_params: TrainingParams):
        """Initialize PacemakerTrainer.

        Args:
            training_params: Configuration parameters for training.
        """
        self.params = training_params

    def prepare_dataset(self, structures: List[Atoms]) -> str:
        """Convert a list of labeled Atoms objects into a training dataset.

        Args:
            structures: A list of labeled ASE Atoms objects.

        Returns:
            str: The file path to the created dataset (temp or specific for this iter).
        """
        df = pd.DataFrame({"ase_atoms": structures})
        output_path = "training_data_current_iter.pckl.gzip"
        df.to_pickle(output_path, compression="gzip")
        return output_path

    def _update_and_sample_dataset(self, new_data_path: str) -> str:
        """Update global dataset and create a mixed training set (Experience Replay).

        Args:
            new_data_path: Path to the new data pickle file.

        Returns:
            str: Path to the mixed training dataset file.
        """
        global_path = Path(self.params.global_dataset_path)

        # 1. Load New Data
        new_df = pd.read_pickle(new_data_path)
        if hasattr(new_df, "ase_atoms"):
             # Standardize column name if needed, though prepare_dataset sets it to ase_atoms
             pass

        # 2. Update Global Dataset
        if global_path.exists():
            try:
                global_df = pd.read_pickle(global_path)
                # Append
                combined_global = pd.concat([global_df, new_df], ignore_index=True)
            except Exception as e:
                logger.error(f"Failed to load global dataset: {e}. Starting fresh.")
                combined_global = new_df
        else:
            # Ensure parent directory exists
            global_path.parent.mkdir(parents=True, exist_ok=True)
            combined_global = new_df

        # Save Updated Global Dataset
        combined_global.to_pickle(global_path, compression="gzip")
        logger.info(f"Global dataset updated: {len(combined_global)} structures stored at {global_path}")

        # 3. Experience Replay Sampling
        # Strategy: Use 100% of new data + (ratio * len(new)) of OLD data.

        # Identify old data (Global - New)
        # Since we just concatenated, the last len(new_df) are new.
        # But to be robust against duplicates or re-runs, let's treat 'combined_global' as the source
        # and we explicitly want to include 'new_df'.

        n_new = len(new_df)
        n_global = len(combined_global)
        n_old = n_global - n_new

        ratio = self.params.replay_ratio

        if n_old > 0 and ratio > 0:
            n_replay = int(n_new * ratio)
            n_replay = min(n_replay, n_old) # Can't sample more than we have

            if n_replay > 0:
                # Old data is approximately the first n_old indices if we just appended
                # This is a safe assumption for simple appending.
                old_df = combined_global.iloc[:n_old]
                sampled_old = old_df.sample(n=n_replay)

                final_training_df = pd.concat([new_df, sampled_old], ignore_index=True)
                logger.info(f"Experience Replay: Mixed {n_new} new + {n_replay} old structures.")
            else:
                final_training_df = new_df
        else:
            final_training_df = new_df
            logger.info("Experience Replay: Disabled or insufficient history. Using only new data.")

        output_mixed = "training_data_mixed.pckl.gzip"
        final_training_df.to_pickle(output_mixed, compression="gzip")
        return output_mixed

    def update_active_set(self, dataset_path: str, potential_yaml_path: str) -> str:
        """Update the active set using pace_activeset.

        Args:
            dataset_path: Path to the full training dataset.
            potential_yaml_path: Path to the potential basis set definition.

        Returns:
            str: The path to the updated active set (.asi) file.
        """
        output_asi = "potential.asi"

        # Ensure we are using the robust mixed dataset for active set selection if possible,
        # but the interface receives 'dataset_path'.
        # Usually, Orchestrator calls prepare_dataset -> returns path -> calls update_active_set -> calls train.
        # We will assume dataset_path is the one to use.

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
            dataset_path: Path to the new data (or current iteration data).
            initial_potential: Path to the initial potential file to start from.
            potential_yaml_path: Path to potential.yaml.
            asi_path: Path to the current Active Set Index file.

        Returns:
            str: The path to the newly trained potential file.
        """

        # 1. Experience Replay Mixing
        # The input 'dataset_path' typically comes from 'prepare_dataset' which contains only new structures.
        # We now mix it with global history.
        mixed_dataset_path = self._update_and_sample_dataset(dataset_path)

        # 2. Update Active Set
        current_asi = asi_path
        if potential_yaml_path:
             try:
                 # We use the mixed dataset to select the active set to ensure coverage
                 current_asi = self.update_active_set(mixed_dataset_path, potential_yaml_path)
                 logger.info(f"Active Set updated: {current_asi}")
             except Exception as e:
                 logger.error(f"Failed to update active set: {e}. Proceeding with existing ASI if available.")

        # Determine elements
        elements = []
        if initial_potential is None:
            try:
                df = pd.read_pickle(mixed_dataset_path)
                all_symbols = set()
                for atoms in df['ase_atoms']:
                    all_symbols.update(atoms.get_chemical_symbols())
                elements = sorted(list(all_symbols))
            except Exception as e:
                logger.warning(f"Could not read elements from dataset: {e}")

        # 3. Dynamic Config Generation
        config = {
            "cutoff": self.params.ace_cutoff,
            "data": {
                "filename": mixed_dataset_path,
                "test_size": self.params.test_size,
            },
            "potential": {
                "delta": True,
            },
            "fitting": {
                "fit_cycles": 1,
                "max_time": self.params.max_training_time,
                "weighting": {
                     "type": "EnergyForce",
                     "energy": self.params.energy_weight,
                     "force": self.params.force_weight
                }
            },
            "backend": {
                "evaluator": "tensorpot",
            }
        }

        if self.params.ladder_step:
             config["fitting"]["ladder_step"] = self.params.ladder_step

        if initial_potential:
            config["fitting"]["input_potential"] = initial_potential
        else:
            config["potential"]["elements"] = elements
            config["potential"]["bonds"] = {
                "N": 3,
                "max_deg": 10,
                "r0": 1.5,
                "rad_base": "Chebyshev",
                "rad_parameters": [self.params.ace_cutoff],
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

        # 4. Run Training
        logger.info(f"Starting Pacemaker training with config: {config}")
        try:
            subprocess.run(["pacemaker", input_yaml_path], check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Pacemaker training failed: {e}")
            raise e

        output_potential = "output_potential.yace"

        if not Path(output_potential).exists():
            # Fallback search strategy
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
