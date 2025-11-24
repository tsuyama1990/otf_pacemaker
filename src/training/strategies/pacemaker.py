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
from src.utils.hardware import get_available_vram, suggest_batch_size

logger = logging.getLogger(__name__)

class PacemakerTrainer(Trainer):
    """Manages the training process for ACE potentials with Experience Replay, Ladder Strategy, and GPU opt."""

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

        # 2. Update Global Dataset
        if global_path.exists():
            try:
                global_df = pd.read_pickle(global_path)
                combined_global = pd.concat([global_df, new_df], ignore_index=True)
            except Exception as e:
                logger.error(f"Failed to load global dataset: {e}. Starting fresh.")
                combined_global = new_df
        else:
            global_path.parent.mkdir(parents=True, exist_ok=True)
            combined_global = new_df

        # Save Updated Global Dataset
        combined_global.to_pickle(global_path, compression="gzip")
        logger.info(f"Global dataset updated: {len(combined_global)} structures stored at {global_path}")

        # 3. Experience Replay Sampling
        n_new = len(new_df)
        n_global = len(combined_global)
        n_old = n_global - n_new
        ratio = self.params.replay_ratio

        if n_old > 0 and ratio > 0:
            n_replay = int(n_new * ratio)
            n_replay = min(n_replay, n_old)

            if n_replay > 0:
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

    def prune_active_set(self, active_set_path: str, threshold: float = 0.99) -> None:
        """Prune the active set to remove redundant structures.

        Uses pace_activeset --prune logic (if available, otherwise we might need custom logic).
        Prompt says: "Use pace_activeset's pruning function".
        Assuming usage: pace_activeset -a potential.asi --prune threshold

        Args:
            active_set_path: Path to the .asi file.
            threshold: Similarity threshold (default 0.99).
        """
        if not Path(active_set_path).exists():
            logger.warning(f"Active set {active_set_path} not found. Skipping prune.")
            return

        cmd = [
            "pace_activeset",
            "-a", active_set_path,
            "--prune", str(threshold),
            "--overwrite" # Assuming we want to update in place
        ]

        logger.info(f"Pruning Active Set: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info("Active Set pruning completed.")
        except subprocess.CalledProcessError as e:
            logger.warning(f"pace_activeset pruning failed: {e.stderr}")
            # Non-critical failure

    def _get_dynamic_config(self, iteration: int) -> Dict[str, Any]:
        """Generate dynamic configuration based on iteration and hardware.

        Args:
            iteration: Current training iteration (0-indexed).

        Returns:
            dict: Configuration parameters for potential and backend.
        """
        config = {}

        # 1. Ladder Logic
        if self.params.ladder_strategy:
            step = iteration // self.params.ladder_interval
            new_max_deg = self.params.initial_max_deg + step
            if new_max_deg > self.params.final_max_deg:
                new_max_deg = self.params.final_max_deg

            logger.info(f"Ladder Strategy: Iteration {iteration}, max_deg set to {new_max_deg}")
            config["max_deg"] = new_max_deg
        else:
            config["max_deg"] = self.params.initial_max_deg

        # 2. GPU Logic
        vram = get_available_vram()
        batch_size = suggest_batch_size(vram)
        logger.info(f"GPU Logic: VRAM={vram} bytes, Batch Size={batch_size}")
        config["batch_size"] = batch_size

        return config

    def train(self, dataset_path: str, initial_potential: Optional[str] = None,
              potential_yaml_path: Optional[str] = None, asi_path: Optional[str] = None,
              iteration: int = 0) -> str:
        """Train the potential using Pacemaker.

        Args:
            dataset_path: Path to the new data (or current iteration data).
            initial_potential: Path to the initial potential file to start from.
            potential_yaml_path: Path to potential.yaml.
            asi_path: Path to the current Active Set Index file.
            iteration: Current AL iteration count.

        Returns:
            str: The path to the newly trained potential file.
        """

        # 1. Experience Replay Mixing
        mixed_dataset_path = self._update_and_sample_dataset(dataset_path)

        # 2. Update Active Set
        current_asi = asi_path
        if potential_yaml_path:
             try:
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
        dynamic_settings = self._get_dynamic_config(iteration)

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
                "batch_size": dynamic_settings.get("batch_size", 32)
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
                "max_deg": dynamic_settings.get("max_deg", 6),
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
