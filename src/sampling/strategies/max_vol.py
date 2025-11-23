"""MaxVol Sampler Strategy."""

import logging
import subprocess
import numpy as np
from pathlib import Path
from typing import List, Tuple
from ase import Atoms
from ase.io import read, write

from src.core.interfaces import Sampler

logger = logging.getLogger(__name__)

class MaxVolSampler(Sampler):
    """Selects high-value structures using MaxVol algorithm via pace_select.

    This sampler works on a dump file containing multiple frames, converts it
    to extxyz, runs pace_select to pick the most informative structures,
    and then identifies the most uncertain atom within each selected structure.
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

        Returns:
            List[Tuple[Atoms, int]]: List of (structure, max_gamma_atom_index).
        """
        dump_file = kwargs.get('dump_file')
        potential_yaml_path = kwargs.get('potential_yaml_path')
        asi_path = kwargs.get('asi_path')
        n_selection = kwargs.get('n_clusters')

        if not all([dump_file, potential_yaml_path, asi_path, n_selection]):
             missing = [k for k, v in [("dump_file", dump_file), ("potential_yaml_path", potential_yaml_path),
                                       ("asi_path", asi_path), ("n_clusters", n_selection)] if not v]
             raise ValueError(f"MaxVolSampler missing required arguments: {missing}")

        dump_path = Path(dump_file)
        if not dump_path.exists():
             raise FileNotFoundError(f"Dump file not found: {dump_path}")

        # 1. Convert LAMMPS dump to extxyz with ASE (handling symbols)
        try:
            frames = read(dump_path, index=":", format="lammps-dump-text")
        except Exception as e:
            logger.error(f"Failed to read dump file with ASE: {e}")
            raise

        elements = kwargs.get('elements')
        if elements:
             for atoms in frames:
                 self._assign_symbols(atoms, elements)

        # Save to temporary extxyz
        extxyz_path = dump_path.with_suffix(".extxyz")
        write(extxyz_path, frames)
        logger.info(f"Converted dump to {extxyz_path} with {len(frames)} frames.")

        # 2. Run pace_select
        selected_file = "selected_structures.extxyz"
        cmd = [
            self.cmd,
            "-p", potential_yaml_path,
            "-a", asi_path,
            "-m", str(n_selection),
            "-o", selected_file,
            str(extxyz_path)
        ]

        logger.info(f"Running pace_select: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"pace_select failed: {e.stderr}")
            raise RuntimeError(f"pace_select execution failed: {e.stderr}")

        if not Path(selected_file).exists():
             logger.warning("pace_select finished but output file not found. Maybe no structures selected?")
             return []

        # 3. Read selected structures
        selected_frames = read(selected_file, index=":")
        logger.info(f"pace_select selected {len(selected_frames)} structures.")

        # 4. Identify Max Gamma Atom in each frame
        results = []
        for atoms in selected_frames:
             gamma_key = None
             possible_keys = ["f_f_gamma", "f_gamma", "gamma", "MaxVol_Gamma"]

             for key in possible_keys:
                if key in atoms.arrays:
                    gamma_key = key
                    break

             if not gamma_key:
                  logger.warning(f"Gamma not found in selected structure keys: {atoms.arrays.keys()}. Using index 0.")
                  idx = 0
             else:
                  gammas = atoms.get_array(gamma_key)
                  if gammas.ndim > 1:
                       gammas = gammas.flatten()
                  idx = np.argmax(gammas)

             results.append((atoms, int(idx)))

        return results

    def _assign_symbols(self, atoms: Atoms, elements: List[str]):
        """Assign chemical symbols based on type IDs."""
        if 'type' in atoms.arrays:
             types = atoms.get_array('type')
             symbols = []
             for t in types:
                 if 1 <= t <= len(elements):
                     symbols.append(elements[t-1])
                 else:
                     symbols.append("X")
             atoms.set_chemical_symbols(symbols)
