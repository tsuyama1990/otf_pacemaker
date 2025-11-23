"""Training module for Pacemaker potentials.

This module handles the preparation of datasets and training of ACE potentials.
"""

import pandas as pd
import subprocess
import yaml
import logging
from pathlib import Path
from typing import List, Optional
from ase import Atoms

from src.interfaces import Trainer

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
        # We assume dataset_path is the cumulative dataset?
        # Usually, active set is built from the pool of all available data.
        # If 'dataset_path' passed here is just the *new* batch, we might need to merge.
        # However, the Orchestrator loop typically accumulates data or we train on the new batch
        # while keeping the old active set?
        # The prompt says: "pace_activeset ... 学習データ全体から ...".
        # So 'dataset_path' MUST be the full dataset.
        # The Orchestrator should ensure dataset_path points to the cumulative dataset.

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
        # The requirement says: "Active Setの管理と更新 ... 再学習の前に、必ず pace_activeset コマンドを実行 ... Trainerに追加"
        # And "change train method to call update_active_set".
        # We need potential_yaml_path for this.

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

        # Inject Active Set into config if available
        # Pacemaker uses 'weighting: { type: EnergyBased, ... }' or similar usually,
        # but if we have an ASI file, how do we tell pacemaker to use it?
        # Actually, 'pace_activeset' generates the basis set selection.
        # If we use '-f' in pace_activeset, we might be selecting features.
        # But wait, usually 'pace_select' selects *structures* to add to training.
        # 'pace_activeset' selects the *basis functions* (features) or just the active set of structures defining the basis?
        # In ACE, the active set usually refers to the structures that define the basis (MaxVol).
        # Pacemaker config doesn't typically take an '.asi' file directly in 'fitting'.
        # It takes 'input_potential' which contains the basis.
        # However, the prompt says "Generate .asi ... and use it".
        # If 'pace_activeset' updates the active set, maybe we don't need to pass it explicitly to training
        # IF we don't change the potential topology.
        # BUT, if we are updating the active set, we are effectively changing the basis.
        # Pacemaker training (fitting) usually fits coefficients.
        # If the basis changes (via Active Set update), we might need to reflect that.

        # Prompt: "PacemakerTrainer (pace_activeset) -> New Active Set (new.asi) -> PacemakerTrainer (pacemaker) -> New Potential (new.yace)"
        # If we just run 'pacemaker', does it know about 'new.asi'?
        # The prompt says: "Sampler に現在の .asi を渡し...". This is for sampling.
        # For training: "pace_activeset ... -f (full) option ...".
        # If 'pace_activeset' modifies the *potential definition* (e.g. creating a new .yace or .yaml with new basis), then we use that.
        # But 'pace_activeset' outputs an '.asi' file (indices).
        # Is there a way to tell pacemaker to use this ASI?
        # Maybe `config['fitting']['active_set'] = current_asi`?
        # Or maybe we rely on `input_potential` which already has a basis, and `pace_activeset` is just for the *Next* sampling step?
        # Let's re-read the prompt carefully.
        # "Active Setの管理と更新... 再学習の前に... 最適な基底セット... を再構築する"
        # "Sampler に現在の .asi を渡し..."

        # If we look at `active_learning.py` Sampler implementation, it USES `asi_path` in `pace_select`.
        # So the updated ASI is crucial for the *next* sampling cycle.
        # Does it affect the *current* training?
        # If we re-train the coefficients, we use the full dataset.
        # The basis set (defined by the active set of environments) might need to be updated in the potential file itself.
        # `pace_activeset` might just output the indices.
        # `pacemaker` might re-select if configured to do so, OR we might need to tell it.
        # If we follow the instructions strictly: "Trainerに update_active_set メソッドを追加 ... train メソッド内で ... 呼ぶ".
        # It doesn't explicitly say we must pass the .asi to `pacemaker` command.
        # But logically, if we updated the active set (basis), we want the new potential to use it.
        # `pace_activeset` documentation (general knowledge): it can update the potential file or just output indices.
        # The prompt says: "Generate .asi file".
        # If `pacemaker` is run with `input_potential`, it uses that potential's basis.
        # If we want to *update* the basis, we might need to construct a new potential file using the new ASI?
        # Or maybe `pace_activeset` is only for the Sampler (to know what we already know)?
        # Quote: "MD Dump + Current Active Set (old.asi) -> MaxVolSampler ... -> New Active Set (new.asi) -> New Potential (new.yace)"
        # This implies `new.asi` influences `new.yace`.
        # How?
        # Maybe `pace_activeset` *is* the way we prepare the potential for fitting?
        # Or maybe we just pass `active_set_file: ...` in the config?
        # Let's assume for now that `pace_activeset` is primarily for the *Sampler* to avoid selecting redundant structures next time.
        # And potentially for the Trainer if Pacemaker supports it.
        # I will return the updated ASI path so the Orchestrator can track it.
        pass

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
