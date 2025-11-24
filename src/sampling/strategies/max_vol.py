"""MaxVol Sampler Strategy."""

import logging
import subprocess
import numpy as np
import os
from pathlib import Path
from typing import List, Tuple, Optional
from ase import Atoms
from ase.io import iread, write

from src.core.interfaces import Sampler

logger = logging.getLogger(__name__)

class MaxVolSampler(Sampler):
    """Selects high-value structures using MaxVol algorithm via pace_select.

    This sampler works on a dump file containing multiple frames, converts it
    to extxyz in a streaming fashion, runs pace_select to pick the most informative
    structures, and then identifies the most uncertain atom within each selected structure.
    """

    def __init__(self, pace_select_cmd: str = "pace_select"):
        self.cmd = pace_select_cmd

    def sample(self, **kwargs) -> List[Tuple[Atoms, int]]:
        """Select structures using pace_select and identify max gamma atoms.

        Args:
            **kwargs: Must contain:
                - dump_file (str): Path to the LAMMPS dump file.
                - potential_yaml_path (str): Path to potential.yaml.
                - asi_path (str): Path to the current active set index (.asi) file.
                - n_clusters (int): Number of structures to select (passed to pace_select).
                - elements (List[str]): List of chemical symbols corresponding to Type IDs.

        Returns:
            List[Tuple[Atoms, int]]: List of (structure, max_gamma_atom_index).
        """
        dump_file = kwargs.get('dump_file')
        potential_yaml_path = kwargs.get('potential_yaml_path')
        asi_path = kwargs.get('asi_path')
        n_selection = kwargs.get('n_clusters')
        elements = kwargs.get('elements')

        if not all([dump_file, potential_yaml_path, asi_path, n_selection, elements]):
             missing = [k for k, v in [("dump_file", dump_file), ("potential_yaml_path", potential_yaml_path),
                                       ("asi_path", asi_path), ("n_clusters", n_selection), ("elements", elements)] if not v]
             raise ValueError(f"MaxVolSampler missing required arguments: {missing}")

        dump_path = Path(dump_file)
        if not dump_path.exists():
             raise FileNotFoundError(f"Dump file not found: {dump_path}")

        temp_extxyz = dump_path.with_suffix(".temp.extxyz")

        try:
            # --- Pass 1: Streaming Conversion & Accumulation ---
            logger.info("Starting Pass 1: Streaming conversion from LAMMPS dump to extxyz.")
            count = 0
            with open(temp_extxyz, "w") as f_out:
                # Iterate over the dump file frame by frame
                for atoms in iread(dump_path, index=":", format="lammps-dump-text"):
                    # Apply element mapping
                    self._assign_symbols(atoms, elements)
                    # Write to temporary file
                    write(f_out, atoms, format="extxyz")
                    count += 1

            logger.info(f"Pass 1 complete. Processed {count} frames.")

            # --- External Execution: pace_select ---
            # We assume pace_select writes selected indices to stdout
            # The prompt says: "Capture the standard output to get the list of selected frame indices"

            cmd = [
                self.cmd,
                "-p", potential_yaml_path,
                "-a", asi_path,
                "-m", str(n_selection),
                str(temp_extxyz)
            ]

            logger.info(f"Running pace_select: {' '.join(cmd)}")
            try:
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                stdout = result.stdout.strip()
            except subprocess.CalledProcessError as e:
                logger.error(f"pace_select failed: {e.stderr}")
                raise RuntimeError(f"pace_select execution failed: {e.stderr}")

            # Parse selected indices from stdout
            selected_indices = set()
            if stdout:
                for line in stdout.splitlines():
                    try:
                        # Assuming one index per line, or space separated
                        parts = line.split()
                        for p in parts:
                            selected_indices.add(int(p))
                    except ValueError:
                        logger.warning(f"Could not parse line from pace_select output: {line}")

            logger.info(f"pace_select selected {len(selected_indices)} structures.")

            if not selected_indices:
                return []

            # --- Pass 2: Extraction ---
            results = []
            max_index = max(selected_indices)

            logger.info("Starting Pass 2: Extraction of selected frames.")

            # Stream the temporary extxyz file from the beginning
            for i, atoms in enumerate(iread(temp_extxyz, index=":", format="extxyz")):
                if i in selected_indices:
                    # Identify Max Gamma Atom
                    idx = self._find_max_gamma_index(atoms)
                    results.append((atoms, idx))

                    # Optimization (Early Exit)
                    if len(results) == len(selected_indices):
                        logger.info("All selected structures retrieved. Exiting early.")
                        break

                # Safety break if we passed the max required index (though usually len check covers this)
                if i > max_index:
                     break

            return results

        finally:
            # --- Cleanup ---
            if temp_extxyz.exists():
                try:
                    temp_extxyz.unlink()
                    logger.info(f"Deleted temporary file: {temp_extxyz}")
                except OSError as e:
                    logger.warning(f"Failed to delete temporary file {temp_extxyz}: {e}")

    def _assign_symbols(self, atoms: Atoms, elements: List[str]):
        """Assign chemical symbols based on type IDs."""
        # Note: ASE's lammpsrun might consume 'type' column and set atomic numbers.
        # If 'type' is missing, check if we can infer it or rely on existing behavior.
        # However, for correct mapping, we need the original type IDs.

        types = None
        if 'type' in atoms.arrays:
             types = atoms.get_array('type')
        else:
             # Fallback: if 'type' is not in arrays, check if we can use atomic numbers
             # assuming simple 1->1 mapping (H->1, He->2, etc.) which ASE does by default for types 1, 2...
             # This is risky but standard ASE behavior for unknown types.
             logger.debug(f"Type array missing. Current symbols: {atoms.get_chemical_symbols()}. Trying to infer from numbers.")
             # For types 1, 2... ASE assigns atomic numbers 1, 2...
             # So we can treat atomic numbers as types.
             types = atoms.numbers

        if types is not None:
             symbols = []
             for t in types:
                 if 1 <= t <= len(elements):
                     symbols.append(elements[t-1])
                 else:
                     symbols.append("X") # Unknown
             atoms.set_chemical_symbols(symbols)
        else:
             logger.warning("Could not find 'type' array or infer types. Symbols might be incorrect.")

    def _find_max_gamma_index(self, atoms: Atoms) -> int:
        """Find the index of the atom with the highest gamma value."""
        gamma_key = None
        possible_keys = ["f_f_gamma", "f_gamma", "gamma", "MaxVol_Gamma"]

        for key in possible_keys:
            if key in atoms.arrays:
                gamma_key = key
                break

        if not gamma_key:
            logger.warning(f"Gamma not found in structure keys: {atoms.arrays.keys()}. Using index 0.")
            return 0
        else:
            gammas = atoms.get_array(gamma_key)
            if gammas.ndim > 1:
                gammas = gammas.flatten()
            return int(np.argmax(gammas))
